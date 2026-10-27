"""Semantic map of the explanations: t-SNE of the final-step explanations (sentence embeddings), with every
earlier step placed relative to those final positions. Also stores pairwise cosine similarity per step.
Output: OUT/semantic_map.json {neuron: [[x, y] per step]}, OUT/semantic_sim.json.
"""
import base64
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import load_config, out_path, read_jsonl, log

MODEL = "BAAI/bge-small-en-v1.5"


def texts(cfg, smp):
    expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
    keys, docs = [], []
    for l, n in smp["neurons"]:
        for i, s in enumerate(smp["steps"]):
            e = expl[f"{l}_{n}_{s}"]["explanation"]
            keys.append((f"{l}_{n}", i))
            docs.append(f"{e.get('concept') or ''}. {e.get('description') or ''}")
    return keys, docs


@torch.no_grad()
def embed(docs, dev):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    out = []
    for i in range(0, len(docs), 256):
        b = tok(docs[i:i + 256], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def tsne(X, dev, perplexity=30.0, iters=1000, seed=0):
    """Exact t-SNE on the GPU (all pairs) after PCA to 50 dims."""
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float64)
    X = X - X.mean(0)
    U, S, V = torch.pca_lowrank(X, q=50, center=False)
    X = (X @ V).to(dev)
    N = X.shape[0]
    D = torch.cdist(X, X) ** 2
    # binary-search each point's Gaussian width to match the perplexity
    target = np.log(perplexity)
    beta = torch.ones(N, 1, dtype=torch.float64, device=dev)
    lo = torch.full_like(beta, 1e-20)
    hi = torch.full_like(beta, 1e20)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    for _ in range(64):
        P = torch.exp(-D * beta).masked_fill(eye, 0)
        sP = P.sum(1, keepdim=True).clamp_min(1e-12)
        H = torch.log(sP) + beta * (D * P).sum(1, keepdim=True) / sP
        too_flat = H > target
        lo = torch.where(too_flat, beta, lo)
        hi = torch.where(too_flat, hi, beta)
        beta = torch.where(hi > 1e19, beta * 2, (lo + hi) / 2)
    P = P / sP
    P = (P + P.T) / (2 * N)
    P = P.clamp_min(1e-12).float()
    Y = (torch.randn(N, 2, generator=g) * 1e-4).to(dev)
    vel = torch.zeros_like(Y)
    gains = torch.ones_like(Y)
    for it in range(iters):
        exag = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        d2 = torch.cdist(Y, Y) ** 2
        num = (1.0 / (1.0 + d2)).masked_fill(eye, 0)
        Q = (num / num.sum()).clamp_min(1e-12)
        W = (exag * P - Q) * num
        grad = 4 * ((W.sum(1, keepdim=True) * Y) - W @ Y)
        gains = torch.where(torch.sign(grad) != torch.sign(vel), gains + 0.2, gains * 0.8).clamp_min(0.01)
        vel = mom * vel - 200.0 * gains * grad
        Y = Y + vel
        Y = Y - Y.mean(0)
    return Y.cpu().numpy()


def anchor_to_final(E, keys, final_idx, ref_xy, k=8, tau=0.02):
    """Place each explanation at the softmax(sim / tau)-weighted mean position of its k nearest final-step explanations."""
    R = E[final_idx]
    sims = E @ R.T
    top = np.argsort(-sims, 1)[:, :k]
    w = np.take_along_axis(sims, top, 1) / tau
    w = np.exp(w - w.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return (w[:, :, None] * ref_xy[top]).sum(1)


def save_similarity(cfg, order, keys, E, k, lo=0.4, hi=1.0):
    """Per step, cosine similarity between all neurons' explanations, quantized to uint8 and base64-encoded.
    Decode: cos = lo + q / 255 * (hi - lo); rows/cols follow `order`."""
    idx = {kk: j for j, kk in enumerate(keys)}
    mats = []
    for i in range(k):
        V = E[[idx[(nid, i)] for nid in order]]
        C = np.clip((V @ V.T - lo) / (hi - lo), 0, 1)
        mats.append(base64.b64encode(np.round(C * 255).astype(np.uint8).tobytes()).decode())
    json.dump(dict(lo=lo, hi=hi, order=order, steps=mats), open(out_path(cfg, "semantic_sim.json"), "w"))


def main():
    cfg = load_config()
    smp = json.load(open(out_path(cfg, "sample.json")))
    dev = torch.device(cfg["DEVICE"])
    keys, docs = texts(cfg, smp)
    log(f"embedding {len(docs)} explanations")
    E = embed(docs, dev)
    k = len(smp["steps"])
    final_idx = [j for j, (_, i) in enumerate(keys) if i == k - 1]
    ref = tsne(E[final_idx], dev, perplexity=5.0, seed=cfg["SEED"])
    ref = (ref - ref.min(0)) / (ref.max(0) - ref.min(0))
    Y = anchor_to_final(E, keys, final_idx, ref)
    Y[final_idx] = ref
    out = {}
    for (nid, i), (x, y) in zip(keys, Y):
        out.setdefault(nid, [None] * k)[i] = [round(float(x), 4), round(float(y), 4)]
    json.dump(out, open(out_path(cfg, "semantic_map.json"), "w"))
    np.save(out_path(cfg, "semantic_embeddings.npy"), E)
    save_similarity(cfg, [f"{l}_{n}" for l, n in smp["neurons"]], keys, E, k)
    log("semantic map done")


if __name__ == "__main__":
    main()
