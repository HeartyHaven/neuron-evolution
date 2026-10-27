"""Build the ~10M-token probe set from the Pile validation split, stratified by pile_set_name.

Output OUT/probe/: tokens.npy, doc_offsets.npy, doc_subset.npy, doc_source.npy, subsets.json, probe_stats.json
"""
import json
import os
import subprocess
import time

import numpy as np
from tokenizers import Tokenizer

from common import load_config, probe_dir, ckpt_dir


def read_val(path):
    proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
    texts, names = [], []
    for line in proc.stdout:
        rec = json.loads(line)
        texts.append(rec["text"])
        names.append(rec["meta"]["pile_set_name"])
    if proc.wait() != 0:
        raise RuntimeError(f"zstd failed on {path}")
    return texts, names


def water_fill(avail, total):
    """Equal quota per subset; subsets that run short give their remainder to the rest."""
    quota, active, remaining = {}, set(avail), total
    while active:
        share = remaining / len(active)
        short = [s for s in active if avail[s] <= share]
        if not short:
            for s in active:
                quota[s] = share
            break
        for s in short:
            quota[s] = avail[s]
            remaining -= avail[s]
            active.remove(s)
    return quota


def pack_rows(chosen, lens, leftover, cfg):
    """Pack documents into rows of exactly MAX_LEN tokens (best-fit decreasing).
    Each row's gap is filled with the prefix of an unused document (round-robin over subsets);
    gaps shorter than MIN_LEN are first widened by trimming the row's longest document."""
    W, MIN = cfg["MAX_LEN"], cfg["MIN_LEN"]
    use_len = {d: int(lens[d]) for d in chosen}
    rows, free = [], []
    by_free = [[] for _ in range(W + 1)]
    for d in sorted(chosen, key=lambda d: -use_len[d]):
        n = use_len[d]
        r = next((r for r in range(n, W + 1) if by_free[r]), None)
        if r is None:
            rows.append([])
            free.append(W)
            i = len(rows) - 1
        else:
            i = by_free[r].pop()
        rows[i].append(d)
        free[i] -= n
        by_free[free[i]].append(i)

    queues = [v for v in leftover.values() if v]
    pool = []
    for j in range(max(len(q) for q in queues)):
        pool.extend(q[j] for q in queues if j < len(q))
    pool_pos = n_fill = n_shrink = 0
    for i, row in enumerate(rows):
        g = free[i]
        if g == 0:
            continue
        if g < MIN:
            need = MIN - g
            for d in sorted(row, key=lambda d: -use_len[d]):
                take = min(need, use_len[d] - MIN)
                use_len[d] -= take
                need -= take
                if need == 0:
                    break
            assert need == 0
            g = MIN
            n_shrink += 1
        while lens[pool[pool_pos]] < g:
            pool_pos += 1
        d = pool.pop(pool_pos)
        use_len[d] = g
        row.append(d)
        free[i] = 0
        n_fill += 1
    return rows, use_len, n_fill, n_shrink


def main():
    cfg = load_config()
    out = probe_dir(cfg)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    texts, names = read_val(cfg["VAL_PATH"])
    print(f"read {len(texts)} documents, {time.time() - t0:.0f}s")

    # All Pythia checkpoints share one tokenizer; no BOS/EOS.
    tok = Tokenizer.from_file(os.path.join(ckpt_dir(cfg, 143000), "tokenizer.json"))
    lens = np.zeros(len(texts), dtype=np.int64)
    ids_all = [None] * len(texts)
    for i in range(0, len(texts), 4096):
        for j, e in enumerate(tok.encode_batch(texts[i:i + 4096], add_special_tokens=False)):
            ids = e.ids[:cfg["MAX_LEN"]]
            ids_all[i + j] = np.asarray(ids, dtype=np.uint16)
            lens[i + j] = len(ids)

    subsets = sorted(set(names))
    sid = {s: k for k, s in enumerate(subsets)}
    doc_sid = np.array([sid[n] for n in names], dtype=np.int64)
    keep = lens >= cfg["MIN_LEN"]
    avail = {k: int(lens[keep & (doc_sid == k)].sum()) for k in range(len(subsets))}
    quota = water_fill(avail, cfg["N_TOKENS"])

    rng = np.random.default_rng(cfg["SEED"])
    chosen, leftover, stats = [], {}, {}
    for k in range(len(subsets)):
        rows = np.flatnonzero(keep & (doc_sid == k))
        rows = rows[rng.permutation(len(rows))]
        cum, picked, rest = 0, [], []
        for r in rows:
            if cum + lens[r] <= quota[k]:
                picked.append(int(r))
                cum += lens[r]
            else:
                rest.append(int(r))
        chosen.extend(picked)
        leftover[k] = rest
        stats[subsets[k]] = dict(available_tokens=avail[k], quota=round(quota[k]), tokens=int(cum), docs=len(picked))

    rows, use_len, n_fill, n_shrink = pack_rows(chosen, lens, leftover, cfg)
    chosen_set = set(chosen)
    for r in rows:
        for d in r:
            st = stats[subsets[doc_sid[d]]]
            if d not in chosen_set:
                st["filler_docs"] = st.get("filler_docs", 0) + 1
            st["tokens_final"] = st.get("tokens_final", 0) + use_len[d]

    doc_source = np.array([d for r in rows for d in r], dtype=np.int64)
    doc_lens = np.array([use_len[d] for d in doc_source], dtype=np.int64)
    doc_offsets = np.zeros(len(doc_source) + 1, dtype=np.int64)
    np.cumsum(doc_lens, out=doc_offsets[1:])
    tokens = np.concatenate([ids_all[d][:use_len[d]] for d in doc_source]).astype(np.uint16)
    W = cfg["MAX_LEN"]
    assert len(tokens) == doc_offsets[-1] == len(rows) * W < 2**31
    assert np.all(doc_offsets[:-1] // W == (doc_offsets[1:] - 1) // W)  # no document crosses a row

    np.save(os.path.join(out, "tokens.npy"), tokens)
    np.save(os.path.join(out, "doc_offsets.npy"), doc_offsets)
    np.save(os.path.join(out, "doc_subset.npy"), doc_sid[doc_source].astype(np.uint8))
    np.save(os.path.join(out, "doc_source.npy"), doc_source)
    with open(os.path.join(out, "subsets.json"), "w") as f:
        json.dump({str(k): s for k, s in enumerate(subsets)}, f, indent=1)
    with open(os.path.join(out, "probe_stats.json"), "w") as f:
        json.dump(dict(source=cfg["VAL_PATH"], n_source_docs=len(texts), n_docs=len(doc_source),
                       n_tokens=int(len(tokens)), row_len=W, n_rows=len(rows), filler_docs=n_fill,
                       shrunk_docs=n_shrink, build_seconds=round(time.time() - t0, 1), subsets=stats), f, indent=1)
    print(f"probe: {len(doc_source)} docs, {len(tokens)} tokens, {len(rows)} rows x {W}, "
          f"{n_fill} filler docs, {n_shrink} shrunk, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
