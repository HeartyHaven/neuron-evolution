"""Readers for the extraction output.

    from load import topk, trajectory
    topk(143000, 6, 123)                      # [(value, doc, pos, subset, context), ...]
    steps, ys = trajectory(6, 123, "subset_sum")

Global token index g: doc = searchsorted(doc_offsets, g, 'right') - 1, pos = g - doc_offsets[doc].
"""
import functools
import json
import os

import numpy as np
from tokenizers import Tokenizer

from common import load_config, probe_dir, acts_dir, ckpt_dir

CTX_BEFORE, CTX_AFTER = 64, 8


@functools.lru_cache(maxsize=None)
def _cfg():
    return load_config()


@functools.lru_cache(maxsize=None)
def _probe():
    d = probe_dir(_cfg())
    tokens = np.load(os.path.join(d, "tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(d, "doc_offsets.npy"))
    subset = np.load(os.path.join(d, "doc_subset.npy"))
    with open(os.path.join(d, "subsets.json")) as f:
        names = json.load(f)
    return tokens, offsets, subset, [names[str(k)] for k in range(len(names))]


@functools.lru_cache(maxsize=None)
def _tok():
    return Tokenizer.from_file(os.path.join(ckpt_dir(_cfg(), 143000), "tokenizer.json"))


@functools.lru_cache(maxsize=256)
def _field(step, field):
    return np.load(os.path.join(acts_dir(_cfg(), step), f"{field}.npy"), mmap_mode="r")


def _bytes_to_unicode():
    """Byte -> printable character map of GPT-2 / NeoX byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_DEC = {c: b for b, c in _bytes_to_unicode().items()}


def token_texts(ids):
    """Decode token by token; a multi-byte character split across tokens goes to the token that completes it."""
    tok = _tok()
    out, pending = [], b""
    for i in ids:
        t = tok.id_to_token(int(i))
        pending += bytes(BYTE_DEC[c] for c in t) if all(c in BYTE_DEC for c in t) else t.encode()
        try:
            out.append(pending.decode("utf-8"))
            pending = b""
        except UnicodeDecodeError:
            if len(pending) >= 4:
                out.append(pending.decode("utf-8", errors="replace"))
                pending = b""
            else:
                out.append("")
    return out


def subset_names():
    return _probe()[3]


def done_steps(tier_b_only=False):
    root = acts_dir(_cfg())
    steps = sorted(int(n[4:]) for n in os.listdir(root) if os.path.exists(os.path.join(root, n, "DONE")))
    if tier_b_only:
        steps = [s for s in steps if s in set(_cfg()["TIER_B_STEPS"])]
    return steps


def locate(g):
    offsets = _probe()[1]
    doc = np.searchsorted(offsets, g, side="right") - 1
    return doc, np.asarray(g) - offsets[doc]


def context(g, before=CTX_BEFORE, after=CTX_AFTER):
    """Text around token g within its document, with the token marked as [[ ]]."""
    tokens, offsets, _, _ = _probe()
    doc, _ = locate(g)
    s, e = max(offsets[doc], g - before), min(offsets[doc + 1], g + after + 1)
    dec = lambda a, b: _tok().decode([int(t) for t in tokens[a:b]], skip_special_tokens=False)
    return dec(s, g) + "[[" + dec(g, g + 1) + "]]" + dec(g + 1, e)


def topk(step, layer, neuron, pole="pos", k=20):
    """[(value, doc, pos, subset, context)] sorted by strength; pole 'neg' returns raw (negative) pre values."""
    assert pole in ("pos", "neg")
    vals = np.asarray(_field(step, f"{pole}_val")[layer, :k, neuron], dtype=np.float32)
    idxs = np.asarray(_field(step, f"{pole}_idx")[layer, :k, neuron])
    subset, names = _probe()[2], _probe()[3]
    out = []
    for v, g in zip(vals, idxs):
        if g < 0:
            continue
        doc, pos = locate(int(g))
        out.append((float(v), int(doc), int(pos), names[subset[doc]], context(int(g))))
    return out


def trajectory(layer, neuron, field="subset_sum", steps=None):
    """(steps, values) with the layer/neuron axes removed, e.g. subset_sum -> [steps, S]."""
    if steps is None:
        steps = done_steps(tier_b_only=field.startswith("terr_"))
    ys = []
    for s in steps:
        a = _field(s, field)
        ys.append(np.asarray(a[layer, :, neuron] if a.ndim == 3 and field != "hist" else a[layer, neuron]))
    return np.asarray(steps), np.stack(ys)


def hist_edges():
    """Bin edges in raw pre units: edges[i] = sinh(-6 + 12*i/HIST_BINS)."""
    H = _cfg()["HIST_BINS"]
    return np.sinh(-6.0 + 12.0 * np.arange(H + 1) / H)


def hist(step, layer, neuron):
    return hist_edges(), np.asarray(_field(step, "hist")[layer, neuron], dtype=np.int64)


def territory(step, layer, neuron):
    """Tier B: (token indices [K_TOK], values [K_TOK]) over all positions, no per-document dedup."""
    if step not in set(_cfg()["TIER_B_STEPS"]):
        raise ValueError(f"step {step} is not in TIER_B_STEPS")
    idx = np.asarray(_field(step, "terr_idx")[layer, :, neuron])
    val = np.asarray(_field(step, "terr_val")[layer, :, neuron], dtype=np.float32)
    return idx, val


def loss(steps=None):
    """(steps, overall loss [n], per-subset loss [n, S])."""
    steps = steps or done_steps()
    tot = np.array([float(_field(s, "loss")) for s in steps])
    sub = np.stack([np.asarray(_field(s, "loss_subset")) for s in steps])
    return np.asarray(steps), tot, sub
