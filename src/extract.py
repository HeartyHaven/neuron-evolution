"""Per-checkpoint activation summaries for every MLP neuron.

Activation ("pre") = output of gpt_neox.layers[l].mlp.dense_h_to_4h, before GELU. Hooks fold each batch into
GPU-resident statistics and drop the raw activations. Rows pack several documents; a block-diagonal causal
mask and per-document position_ids make each document's activations identical to a standalone forward pass.
Output: OUT/acts/step{S}/{field}.npy + meta.json + DONE, and OUT/manifest.json.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM

from common import load_config, probe_dir, acts_dir, ckpt_dir, step_order

HERE = os.path.dirname(os.path.abspath(__file__))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


CODE_FILES = ["common.py", "build_probe.py", "extract.py", "load.py", "../config.yaml"]


def load_probe(cfg):
    d = probe_dir(cfg)
    tokens = np.load(os.path.join(d, "tokens.npy"))
    offsets = np.load(os.path.join(d, "doc_offsets.npy"))
    subset = np.load(os.path.join(d, "doc_subset.npy"))
    with open(os.path.join(d, "subsets.json")) as f:
        names = json.load(f)
    return tokens, offsets, subset, [names[str(k)] for k in range(len(names))]


def make_batches(n_tokens, W, budget):
    n_rows = n_tokens // W
    assert n_rows * W == n_tokens
    B = max(1, budget // W)
    return [(r, min(B, n_rows - r)) for r in range(0, n_rows, B)]


INT64_MAX = torch.iinfo(torch.int64).max


@torch.compile(dynamic=True)
def _elementwise(x16, bin_off, H: int):
    x = x16.float()
    gt0 = x > 0
    b = ((torch.asinh(x) + 6.0) / 12.0 * H).floor().clamp(0, H - 1).long() + bin_off
    return x.sum(0), (x * x).sum(0), gt0.sum(0), b, F.gelu(x), gt0.float()


@torch.compile(dynamic=True)
def _first_hit(x16, ext, seg, g):
    return torch.where(x16 == ext[seg], g[:, None], INT64_MAX)


class Stats:
    """All statistics for one checkpoint, kept on the GPU."""

    def __init__(self, L, N, S, cfg, tier_b, dev):
        K, KT, H = cfg["K_DOC"], cfg["K_TOK"], cfg["HIST_BINS"]
        f32, i32 = torch.float32, torch.int32
        self.H, self.N, self.tier_b = H, N, tier_b
        self.pos_val = torch.full((L, K, N), -float("inf"), dtype=f32, device=dev)
        self.pos_idx = torch.full((L, K, N), -1, dtype=i32, device=dev)
        self.neg_val = torch.full((L, K, N), -float("inf"), dtype=f32, device=dev)  # stores -pre
        self.neg_idx = torch.full((L, K, N), -1, dtype=i32, device=dev)
        self.hist = torch.zeros((L, N * H), dtype=torch.int64, device=dev)
        self.s1 = torch.zeros((L, N), dtype=torch.float64, device=dev)
        self.s2 = torch.zeros((L, N), dtype=torch.float64, device=dev)
        self.npos = torch.zeros((L, N), dtype=torch.int64, device=dev)
        self.subset_sum = torch.zeros((L, S, N), dtype=torch.float64, device=dev)
        self.subset_cnt = torch.zeros((L, S, N), dtype=torch.float64, device=dev)
        self.n_tok = 0
        self.bin_off = (torch.arange(N, device=dev, dtype=torch.int64) * H)[None, :]
        if tier_b:
            self.terr_val = torch.full((L, KT, N), -float("inf"), dtype=f32, device=dev)
            self.terr_idx = torch.full((L, KT, N), -1, dtype=i32, device=dev)

    @staticmethod
    def _merge(vals, idxs, new_v, new_i):
        """Merge candidates [M, N] into a sorted top-K [K, N] along dim 0."""
        v = torch.cat([vals, new_v], 0)
        i = torch.cat([idxs, new_i], 0)
        top_v, top_r = v.topk(vals.shape[0], dim=0)
        return top_v, i.gather(0, top_r)

    def update(self, l, pre, ctx):
        """pre: [B, W, N] fp16; packed rows have no padding. Extremes in fp16, sums in fp32."""
        B, T, N = pre.shape
        x16 = pre.reshape(B * T, N)                # [M, N]

        s1, s2, npos, b, gelu, gt0 = _elementwise(x16, self.bin_off, self.H)
        self.s1[l] += s1.double()
        self.s2[l] += s2.double()
        self.npos[l] += npos
        self.hist[l] += torch.bincount(b.view(-1), minlength=N * self.H)
        del b

        # per-subset sum(GELU(pre)) and pre>0 counts
        oh = ctx["onehot"]                         # [S, M] fp32
        self.subset_sum[l] += oh @ gelu
        self.subset_cnt[l] += oh @ gt0
        del gelu, gt0

        # one entry per document: per-document max (min for the negative pole), then merge into top-K
        mx, am = self._segment_ext(x16, ctx, "max")
        self.pos_val[l], self.pos_idx[l] = self._merge(self.pos_val[l], self.pos_idx[l], mx.float(), am)
        mn, an = self._segment_ext(x16, ctx, "min")
        self.neg_val[l], self.neg_idx[l] = self._merge(self.neg_val[l], self.neg_idx[l], -mn.float(), an)

        # Tier B: top-K_TOK over all positions, no per-document dedup
        if self.tier_b:
            g = ctx["gidx"]
            KT = self.terr_val.shape[1]
            if x16.shape[0] > KT:
                v, r = x16.topk(KT, dim=0)
                cand_i = g[r].int()
            else:
                v, cand_i = x16, g[:, None].expand(-1, N).int()
            self.terr_val[l], self.terr_idx[l] = self._merge(
                self.terr_val[l], self.terr_idx[l], v.float(), cand_i)

    @staticmethod
    def _segment_ext(x16, ctx, op):
        """Per-document max or min -> (values [D_b, N], int32 global indices [D_b, N]); ties take the earliest position."""
        seg, g = ctx["seg"], ctx["gidx"]
        M, N = x16.shape
        ext = torch.segment_reduce(x16, op, lengths=ctx["seg_len"], axis=0)
        cand = _first_hit(x16, ext, seg, g)
        am = torch.full((ext.shape[0], N), INT64_MAX, dtype=torch.int64, device=x16.device)
        am.scatter_reduce_(0, seg[:, None].expand(M, N), cand, "amin")
        return ext, am.int()

    def finalize(self):
        n = self.n_tok
        mean = self.s1 / n
        var = (self.s2 / n - mean * mean).clamp_min(0)
        L, N = mean.shape
        out = dict(
            pos_val=self.pos_val.half(), pos_idx=self.pos_idx,
            neg_val=(-self.neg_val).half(), neg_idx=self.neg_idx,
            hist=self.hist.view(L, N, self.H).to(torch.int64),
            mean=mean.float(), std=var.sqrt().float(),
            frac_pos=(self.npos.double() / n).float(),
            subset_sum=self.subset_sum.float(),
            subset_cnt=self.subset_cnt.round().to(torch.int64),
        )
        if self.tier_b:
            out["terr_val"] = self.terr_val.half()
            out["terr_idx"] = self.terr_idx
        res = {}
        for k, v in out.items():
            a = v.cpu().numpy()
            if k in ("hist", "subset_cnt"):
                assert a.max() < 2**32
                a = a.astype(np.uint32)
            res[k] = a
        return res


def run_step(cfg, step, probe, batches, dev, max_batches=None):
    tokens, offsets, subset, names = probe
    L_tier_b = step in set(cfg["TIER_B_STEPS"])
    dtype = getattr(torch, cfg["DTYPE"])
    model = GPTNeoXForCausalLM.from_pretrained(
        ckpt_dir(cfg, step), torch_dtype=dtype, attn_implementation="sdpa").to(dev).eval()
    L = model.config.num_hidden_layers
    N = model.config.intermediate_size
    S = len(names)
    st = Stats(L, N, S, cfg, L_tier_b, dev)
    ctx = {}
    hooks = []
    for l, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.mlp.dense_h_to_4h.register_forward_hook(
            lambda mod, inp, out, l=l: st.update(l, out, ctx)))

    loss_sum = torch.zeros(S, dtype=torch.float64, device=dev)
    loss_cnt = torch.zeros(S, dtype=torch.float64, device=dev)
    W = cfg["MAX_LEN"]
    tok_doc_np = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    tok_doc = torch.from_numpy(tok_doc_np).to(dev)
    offsets_t = torch.from_numpy(offsets).to(dev)
    subset_t = torch.from_numpy(subset.astype(np.int64)).to(dev)
    tokens_t = torch.from_numpy(tokens.astype(np.int64)).to(dev)
    causal = torch.ones(W, W, dtype=torch.bool, device=dev).tril()
    for r0, B in batches[:max_batches]:
        a, b = r0 * W, (r0 + B) * W
        ids_t = tokens_t[a:b].view(B, W)
        gidx = torch.arange(a, b, device=dev)
        doc = tok_doc[a:b]
        seg = doc - doc[0]
        pos = (gidx - offsets_t[doc]).view(B, W)
        dv = doc.view(B, W)
        attn = ((dv[:, :, None] == dv[:, None, :]) & causal)[:, None]
        sub_tok = subset_t[doc]
        onehot = F.one_hot(sub_tok, S).T.float().contiguous()
        seg_len = torch.bincount(seg)
        assert bool((seg_len > 0).all())
        ctx.update(seg=seg, seg_len=seg_len, gidx=gidx, onehot=onehot)
        st.n_tok += b - a

        with torch.inference_mode():
            logits = model(input_ids=ids_t, attention_mask=attn, position_ids=pos).logits
            # next-token loss within the same document
            tgt_mask = dv[:, 1:] == dv[:, :-1]
            lg = logits[:, :-1][tgt_mask].float()
            tgt = ids_t[:, 1:][tgt_mask]
            tl = F.cross_entropy(lg, tgt, reduction="none").double()
            ts = sub_tok.view(B, W)[:, 1:][tgt_mask]
            loss_sum.index_add_(0, ts, tl)
            loss_cnt.index_add_(0, ts, torch.ones_like(tl))
            del logits, lg, attn
    for h in hooks:
        h.remove()

    res = st.finalize()
    res["loss"] = np.array(float(loss_sum.sum() / loss_cnt.sum()), dtype=np.float32)
    res["loss_subset"] = (loss_sum / loss_cnt).float().cpu().numpy()
    del model, st
    torch.cuda.empty_cache()
    return res


def code_hash():
    h = {}
    for f in CODE_FILES:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return h


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_manifest(cfg):
    root = acts_dir(cfg)
    steps = {}
    for meta in glob.glob(os.path.join(root, "step*", "meta.json")):
        d = os.path.dirname(meta)
        if os.path.exists(os.path.join(d, "DONE")):
            with open(meta) as f:
                m = json.load(f)
            steps[m["step"]] = m
    fields = {}
    for m in steps.values():
        for k, v in m["fields"].items():
            fields.setdefault(k, v)
    man = dict(
        config=cfg,
        git_commit=git_commit() or "N/A",
        code_sha256=code_hash(),
        fields=fields,
        notes=dict(
            pre="output of gpt_neox.layers[l].mlp.dense_h_to_4h, before GELU",
            neg_val="raw pre values of the negative pole, ascending",
            hist_edges="edges[i] = sinh(-6 + 12*i/HIST_BINS); end bins also hold values beyond +-sinh(6)",
            idx="global token index g; doc = searchsorted(doc_offsets, g, 'right') - 1",
            loss="mean next-token cross-entropy (nats) on the probe set; loss_subset follows subsets.json",
        ),
        steps={str(s): steps[s] for s in sorted(steps)},
    )
    tmp = os.path.join(cfg["OUT"], "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(cfg["OUT"], "manifest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", default=None, help="comma-separated steps")
    ap.add_argument("--max-batches", type=int, default=None, help="debug: first N batches only, no DONE marker")
    ap.add_argument("--out", default=None, help="debug: alternative output dir")
    args = ap.parse_args()
    cfg = load_config()
    dev = torch.device(args.device)

    probe = load_probe(cfg)
    batches = make_batches(len(probe[0]), cfg["MAX_LEN"], cfg["BATCH_TOKENS"])
    steps = [int(s) for s in args.steps.split(",")] if args.steps else step_order(cfg)
    print(f"{len(probe[1]) - 1} docs, {len(probe[0])} tokens, {len(batches)} batches, "
          f"{len(steps)} checkpoints", flush=True)

    for step in steps:
        d = args.out or acts_dir(cfg, step)
        if args.out:
            d = os.path.join(d, f"step{step}")
        if os.path.exists(os.path.join(d, "DONE")):
            print(f"step{step}: done, skipping", flush=True)
            continue
        t0 = time.time()
        res = run_step(cfg, step, probe, batches, dev, args.max_batches)
        t1 = time.time()
        os.makedirs(d, exist_ok=True)
        for k, v in res.items():
            np.save(os.path.join(d, f"{k}.npy"), v)
        meta = dict(step=step, tier_b=step in set(cfg["TIER_B_STEPS"]),
                    seconds=round(time.time() - t0, 1), compute_seconds=round(t1 - t0, 1),
                    n_tokens=int(len(probe[0])), loss=float(res["loss"]),
                    device=torch.cuda.get_device_name(dev),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    fields={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in res.items()})
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if args.max_batches is None:
            open(os.path.join(d, "DONE"), "w").close()
            write_manifest(cfg)
        print(f"step{step}: {meta['seconds']}s, loss={meta['loss']:.4f}", flush=True)

if __name__ == "__main__":
    main()
