"""Preparation (no LLM calls).
  sample    pick neurons -> OUT/sample.json
  contexts  recompute per-token activations of the sampled neurons -> OUT/contexts/step{S}.json
  metrics   quantitative metrics for all neurons x all extracted steps -> OUT/metrics/all.npz
  ref       similarity to two late reference steps (drift control) -> OUT/metrics/ref.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import GPTNeoXModel

import load
from common import load_config, out_path, ckpt_dir, log


def sample(cfg):
    path = out_path(cfg, "sample.json")
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.default_rng(cfg["SEED"])
    neurons = []
    for l in range(12):
        for n in sorted(rng.choice(3072, cfg["NEURONS_PER_LAYER"], replace=False).tolist()):
            neurons.append([l, n])
    done = set(load.done_steps())
    assert all(s in done for s in cfg["STEPS"]), "some STEPS have not been extracted"
    tokens = load._probe()[0]
    # random-negative pool shared by all neurons and steps
    pool = sorted(rng.choice(len(tokens), cfg["RANDOM_POOL"], replace=False).tolist())
    s = dict(steps=cfg["STEPS"], neurons=neurons, random_pool=pool)
    json.dump(s, open(path, "w"))
    log(f"sample: {len(cfg['STEPS'])} steps x {len(neurons)} neurons")
    return s


def neuron_negatives(cfg, smp, l, n, top_docs):
    """N_RANDOM negatives from the pool, avoiding the neuron's top-64 documents; fixed across steps."""
    rng = np.random.default_rng(cfg["SEED"] * 7919 + l * 10007 + n)
    offsets = load._probe()[1]
    pool = np.array(smp["random_pool"])
    ok = pool[~np.isin(np.searchsorted(offsets, pool, side="right") - 1, list(top_docs))]
    return sorted(rng.choice(ok, cfg["N_RANDOM"], replace=False).tolist())


def contexts(cfg, smp):
    tokens, offsets, subset, names = load._probe()
    dev = torch.device(cfg["DEVICE"])
    neurons = [tuple(x) for x in smp["neurons"]]
    by_layer = {}
    for j, (l, n) in enumerate(neurons):
        by_layer.setdefault(l, []).append((j, n))
    K = cfg["N_TOP"] + cfg["N_HELDOUT"]
    final = max(load.done_steps())
    negs = {}
    for (l, n) in neurons:
        idx = np.asarray(load._field(final, "pos_idx")[l, :, n]).astype(np.int64)
        negs[(l, n)] = neuron_negatives(cfg, smp, l, n, set(np.searchsorted(offsets, idx, side="right") - 1))

    for step in smp["steps"]:
        path = out_path(cfg, "contexts", f"step{step}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        pv = np.asarray(load._field(step, "pos_val"))
        pi = np.asarray(load._field(step, "pos_idx"))
        reqs = []  # (neuron index, kind, rank, global token index)
        for j, (l, n) in enumerate(neurons):
            for r in range(K):
                g = int(pi[l, r, n])
                if g >= 0:
                    reqs.append((j, "top" if r < cfg["N_TOP"] else "heldout", r, g))
            for g in negs[(l, n)]:
                reqs.append((j, "random", -1, int(g)))
        # one forward pass per document, up to the furthest position needed
        need, doc_reqs = {}, {}
        for q in reqs:
            j, kind, r, g = q
            d = int(np.searchsorted(offsets, g, side="right") - 1)
            after = cfg["CTX_AFTER"] if kind == "top" else cfg["DET_AFTER"]
            need[d] = max(need.get(d, 0), min(int(offsets[d + 1]), g + after + 1))
            doc_reqs.setdefault(d, []).append(q)

        model = GPTNeoXModel.from_pretrained(ckpt_dir(cfg, step), torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to(dev).eval()
        cols = {l: torch.tensor([n for _, n in v], device=dev) for l, v in by_layer.items()}
        cap = {}
        hooks = [model.layers[l].mlp.dense_h_to_4h.register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(l, o[..., cols[l]].float())) for l in by_layer]
        col_of = {j: (l, k) for l, v in by_layer.items() for k, (j, _) in enumerate(v)}

        docs = sorted(need, key=lambda d: -(need[d] - offsets[d]))
        results, i = {}, 0
        while i < len(docs):
            T = int(need[docs[i]] - offsets[docs[i]])
            batch = docs[i:i + max(1, cfg["GPU_BATCH_TOKENS"] // T)]
            i += len(batch)
            ids = np.zeros((len(batch), T), dtype=np.int64)
            mask = np.zeros((len(batch), T), dtype=np.int64)
            for b, d in enumerate(batch):
                ln = int(need[d] - offsets[d])
                ids[b, :ln] = tokens[offsets[d]:need[d]]
                mask[b, :ln] = 1
            with torch.inference_mode():
                model(input_ids=torch.from_numpy(ids).to(dev), attention_mask=torch.from_numpy(mask).to(dev))
            acts = {l: v.cpu().numpy() for l, v in cap.items()}
            for b, d in enumerate(batch):
                st = int(offsets[d])
                for j, kind, r, g in doc_reqs[d]:
                    l, k = col_of[j]
                    before, after = (cfg["CTX_BEFORE"], cfg["CTX_AFTER"]) if kind == "top" else (cfg["DET_BEFORE"], cfg["DET_AFTER"])
                    s, e = max(st, g - before), min(int(offsets[d + 1]), g + after + 1)
                    item = dict(rank=r + 1, g=g, doc=d, pos=g - st, subset=names[subset[d]], hit=g - s,
                                tokens=load.token_texts(tokens[s:e]), act_hit=round(float(acts[l][b, g - st, k]), 3))
                    if kind == "top":
                        item["acts"] = [round(float(x), 3) for x in acts[l][b, s - st:e - st, k]]
                        item["stored"] = round(float(pv[l, r, neurons[j][1]]), 3)
                    results.setdefault(j, {}).setdefault(kind, []).append(item)
        for h in hooks:
            h.remove()
        del model
        torch.cuda.empty_cache()

        out = {}
        for j, (l, n) in enumerate(neurons):
            res = results.get(j, {})
            for kind in ("top", "heldout"):
                res.setdefault(kind, []).sort(key=lambda x: x["rank"])
            out[f"{l}_{n}"] = dict(layer=l, neuron=n, step=step, top1=round(float(pv[l, 0, n]), 3),
                                   top=res["top"], heldout=res["heldout"], random=res.get("random", []))
        with open(path + ".tmp", "w") as f:
            json.dump(out, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)
        log(f"contexts step{step}: {len(docs)} docs, {time.time() - t0:.0f}s")


def _sim_helpers():
    """keys_docs(step) -> (neuron x doc keys, neuron x token keys); profile(step) -> centered subset preference."""
    tokens, offsets, subset, names = load._probe()
    tok_doc = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    sub_tokens = np.bincount(subset[tok_doc], minlength=len(names)).astype(np.float64)
    L, N, S = 12, 3072, len(names)
    nid = np.arange(L * N, dtype=np.int64).reshape(L, 1, N)

    def keys_docs(step):
        idx = np.asarray(load._field(step, "pos_idx")).astype(np.int64)
        docs = np.searchsorted(offsets, idx, side="right") - 1
        return nid * (1 << 20) + docs, nid * (1 << 17) + tokens[idx].astype(np.int64)

    def profile(step):
        p = np.asarray(load._field(step, "subset_sum")).astype(np.float64) / sub_tokens[None, :, None]
        return p - p.mean(axis=2, keepdims=True)

    return keys_docs, profile, sub_tokens, L, N, S


def metrics_ref(cfg, refs=(64000, 143000)):
    """Coarse/token/doc similarity of every step to two late references: converging to a fixed target
    gives similar values for both; ongoing drift makes the nearer reference look more similar."""
    path = out_path(cfg, "metrics", "ref.npz")
    if os.path.exists(path):
        return
    steps = load.done_steps()
    keys_docs, profile, _, L, N, S = _sim_helpers()
    out = {k: np.zeros((len(refs), len(steps), L, N), dtype=np.float32) for k in ("coarse", "tok", "doc")}
    for ri, ref in enumerate(refs):
        rk_doc, rk_tok = keys_docs(ref)
        rk_doc_u, rk_tok_u = np.unique(rk_doc), np.unique(rk_tok)
        rprof = profile(ref)
        K = rk_doc.shape[1]
        for si, s in enumerate(steps):
            prof = profile(s)
            out["coarse"][ri, si] = (prof * rprof).sum(1) / np.maximum(np.sqrt((prof ** 2).sum(1) * (rprof ** 2).sum(1)), 1e-12)
            kd, kt = keys_docs(s)
            inter = np.isin(kd, rk_doc_u).sum(1)
            out["doc"][ri, si] = inter / (2 * K - inter)
            out["tok"][ri, si] = np.isin(kt, rk_tok_u).mean(1)
        log(f"metrics ref{ref} done")
    np.savez_compressed(path, steps=np.array(steps), refs=np.array(refs), **out)


def metrics(cfg):
    path = out_path(cfg, "metrics", "all.npz")
    if os.path.exists(path):
        return
    t0 = time.time()
    steps = load.done_steps()
    keys_docs, profile, sub_tokens, L, N, S = _sim_helpers()
    fk_doc, fk_tok = keys_docs(steps[-1])
    fk_doc_u, fk_tok_u = np.unique(fk_doc), np.unique(fk_tok)
    fprof = profile(steps[-1])
    K = fk_doc.shape[1]
    out = {k: np.zeros((len(steps), L, N), dtype=np.float32) for k in
           ("top1", "snr", "frac_pos", "hist_entropy", "selectivity", "coarse_corr_final",
            "doc_jacc_final", "tok_overlap_final", "doc_jacc_prev")}
    prev_doc = None
    for si, s in enumerate(steps):
        pv = np.asarray(load._field(s, "pos_val")).astype(np.float32)
        mean, std = np.asarray(load._field(s, "mean")), np.asarray(load._field(s, "std"))
        out["top1"][si] = pv[:, 0, :]
        out["snr"][si] = (pv.mean(1) - mean) / np.maximum(std, 1e-6)
        out["frac_pos"][si] = load._field(s, "frac_pos")
        h = np.asarray(load._field(s, "hist")).astype(np.float64)
        p = h / h.sum(-1, keepdims=True)
        out["hist_entropy"][si] = -(p * np.log(np.where(p > 0, p, 1))).sum(-1) / np.log(h.shape[-1])
        pp = np.clip(np.asarray(load._field(s, "subset_sum")).astype(np.float64) / sub_tokens[None, :, None], 0, None)
        pp = pp / np.maximum(pp.sum(1, keepdims=True), 1e-12)
        out["selectivity"][si] = 1 + (pp * np.log(np.where(pp > 0, pp, 1))).sum(1) / np.log(S)
        prof = profile(s)
        out["coarse_corr_final"][si] = (prof * fprof).sum(1) / np.maximum(np.sqrt((prof ** 2).sum(1) * (fprof ** 2).sum(1)), 1e-12)
        kd, kt = keys_docs(s)
        inter = np.isin(kd, fk_doc_u).sum(1)
        out["doc_jacc_final"][si] = inter / (2 * K - inter)
        out["tok_overlap_final"][si] = np.isin(kt, fk_tok_u).mean(1)
        if prev_doc is not None:
            inter = np.isin(kd, np.unique(prev_doc)).sum(1)
            out["doc_jacc_prev"][si] = inter / (2 * K - inter)
        prev_doc = kd
        if si % 20 == 0:
            log(f"metrics: {si + 1}/{len(steps)}, {time.time() - t0:.0f}s")
    np.savez_compressed(path, steps=np.array(steps), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sample", "contexts", "metrics", "ref", "all"])
    a = ap.parse_args()
    cfg = load_config()
    smp = sample(cfg)
    if a.what in ("contexts", "all"):
        contexts(cfg, smp)
    if a.what in ("metrics", "all"):
        metrics(cfg)
    if a.what in ("ref", "all"):
        metrics_ref(cfg)


if __name__ == "__main__":
    main()
