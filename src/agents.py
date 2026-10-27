"""LLM stages (OpenAI Responses API).
  explain  explain each (neuron, step) from its top-10 contexts
  detect   score an explanation by picking held-out activating texts among random ones (balanced accuracy)
  judge    two trajectory judges per neuron: recursive generalization vs superposition, and the denoising hypothesis
Results are appended to OUT/*.jsonl by key; reruns resume.
"""
import argparse
import asyncio
import json
import zlib

import numpy as np

from common import load_config, out_path, read_jsonl, run_jobs, LLM


def clean(t):
    return t.replace("\n", "↵")


def render_top(ctx, scale):
    """Tokens with positive activation become «token»k, k = 1..10 relative to the neuron's top activation."""
    out = []
    for t, a in zip(ctx["tokens"], ctx["acts"]):
        k = min(int(round(10 * a / scale)) if a > 0 else 0, 10)
        out.append(f"«{clean(t)}»{k}" if k >= 1 else clean(t))
    return "".join(out)


def render_det(item):
    toks = [clean(t) for t in item["tokens"]]
    h = item["hit"]
    return "".join(toks[:h]) + "[[" + toks[h] + "]]" + "".join(toks[h + 1:])


EXPLAIN_SYS = """You are a language-model interpretability researcher. Below are the 10 contexts (from the Pile) on which one MLP neuron activates most strongly.
In each context, tokens with positive activation are written «token»k, where k = 1..10 is the relative strength (10 = the neuron's strongest activation on this data); unmarked tokens have activation near 0 or negative. ↵ marks a newline.

Task: describe comprehensively what this neuron responds to. Coverage comes before abstraction:
1. First list every distinct activation pattern, including ones that appear only once or twice and seem unrelated to the main pattern. For each, give the trigger token or substring (quoted from the text), the required context, the domain or genre, and which contexts (by number) it covers.
2. Do not drop contexts that do not fit just to get a neat abstract concept; put contexts you cannot explain in unexplained.
3. Only then give the most general abstract concept you can (write none if there is none), and judge whether the patterns are related.

Output only one JSON object, nothing else:
{"patterns":[{"name":"short name","trigger":"trigger token/substring (verbatim)","context":"context condition","domain":"domain/genre","segments":[numbers]}],
 "unexplained":[numbers],
 "description":"complete description covering all patterns, at most 60 words",
 "concept":"abstract concept, at most 8 words, or none",
 "relation":"single|related|unrelated|unclear",
 "clarity":integer 1-5 (5 = all contexts clearly share one pattern; 1 = no visible regularity)}"""


def explain_jobs(cfg, smp):
    jobs = []
    for step in smp["steps"]:
        ctxs = json.load(open(out_path(cfg, "contexts", f"step{step}.json")))
        for l, n in smp["neurons"]:
            c = ctxs[f"{l}_{n}"]
            scale = max(c["top1"], 1e-6)
            segs = [f"[{i + 1}] ({x['subset']}, max activation {x['stored']:.2f})\n{render_top(x, scale)}"
                    for i, x in enumerate(c["top"])]
            jobs.append((f"{l}_{n}_{step}", dict(layer=l, neuron=n, step=step, prompt="\n\n".join(segs))))
    return jobs


async def explain_worker(llm, p):
    res, u = await llm.json_call(EXPLAIN_SYS, p["prompt"])
    return dict(layer=p["layer"], neuron=p["neuron"], step=p["step"], explanation=res,
                usage=dict(prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens")))


DETECT_SYS = """Below is a description of a language-model neuron and 10 texts, each with one target token marked [[ ]]. ↵ marks a newline.
Based on the description, decide on which texts the neuron fires strongly at the target token. Any number from 0 to 10 may fire.
Output only JSON: {"active":[text numbers]}"""


def detect_jobs(cfg, smp, expl):
    jobs = []
    for step in smp["steps"]:
        ctxs = json.load(open(out_path(cfg, "contexts", f"step{step}.json")))
        for l, n in smp["neurons"]:
            key = f"{l}_{n}_{step}"
            if key not in expl:
                continue
            c = ctxs[f"{l}_{n}"]
            items = [(x, 1) for x in c["heldout"]] + [(x, 0) for x in c["random"]]
            order = np.random.default_rng(zlib.crc32(key.encode())).permutation(len(items)).tolist()
            items = [items[i] for i in order]
            e = expl[key]["explanation"]
            desc = e.get("description", "")
            if e.get("concept") and e.get("concept") != "none":
                desc = f"{desc} (in short: {e['concept']})"
            body = f"Neuron description: {desc}\n\n" + "\n".join(f"[{i + 1}] {render_det(x)}" for i, (x, _) in enumerate(items))
            jobs.append((key, dict(layer=l, neuron=n, step=step, prompt=body, labels=[y for _, y in items],
                                   acts=[x["act_hit"] for x, _ in items], top1=c["top1"])))
    return jobs


async def detect_worker(llm, p):
    res, u = await llm.json_call(DETECT_SYS, p["prompt"])
    pred = set(int(i) for i in res.get("active", []) if str(i).isdigit() or isinstance(i, int))
    y = np.array(p["labels"])
    yhat = np.array([1 if i + 1 in pred else 0 for i in range(len(y))])
    tpr = (yhat[y == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (yhat[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return dict(layer=p["layer"], neuron=p["neuron"], step=p["step"], score=float(np.nanmean([tpr, tnr])),
                tpr=float(tpr), tnr=float(tnr), labels=p["labels"], pred=yhat.tolist(), acts=p["acts"],
                top1=p["top1"], usage=dict(prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens")))


TRAJ_INTRO = """Below are observations of one MLP neuron of Pythia-160M at {k} training checkpoints (step 0 is the random initialization; loss falls fast until about step 1000, then slowly, with a slight rise after step 77000).
Each checkpoint gives the explanation an explainer wrote from the top-10 contexts (pattern list, full description, abstract concept, whether the patterns are related, clarity 1-5) and quantitative metrics:
- det: detection score (balanced accuracy at telling held-out strong activations from random text using only the explanation; 0.5 = chance, 1 = perfect), i.e. how sharp the explanation is
- snr: z-score of the mean top-64 activation against the full distribution; sel: domain selectivity (0 = same on all subsets, 1 = fires on one subset only)
- coarse: correlation of the subset preference with the final checkpoint (coarse grain); tok: overlap of top tokens with the final checkpoint; doc: Jaccard of top documents with the final checkpoint (fine grain)
×k after a pattern means it covers k of the 10 contexts.
coarse, tok and doc all compare with the last checkpoint (step 143000), so the last checkpoint is 1 by definition; what matters is when earlier checkpoints approach the final state.

When reading the evidence:
- The explainer was asked to be comprehensive and lists one-off patterns separately. Focus on patterns covering several contexts; ×1 patterns are weak evidence and cannot alone establish unrelated co-existing patterns.
- Each checkpoint has different top-10 samples, so wording varies. Judge whether the neuron really changed concept from tok, doc and coarse, not from wording differences.
- Calibrate confidence: above 0.8 only for clear evidence; about 0.5 when two readings are equally plausible.
"""

RG_SYS = """
Decide whether this neuron's evolution shows "recursive generalization" or "superposition". Definitions (from Goodfire's "Neural networks are already clean"):
- recursive_generalization: there is a core pattern, and later patterns are branches grown out of the core or an existing branch, related in form, meaning or function (e.g. "iy" -> "diy" -> "do it yourself"). It ends up looking polysemantic, but the branch chain traces back to the core.
- superposition: unrelated patterns are put into the same neuron with no shared core, each appearing independently.
- stable_monosemantic: from some checkpoint on it stays one single pattern, with no clear branches.
- replacement: an early concept is replaced by an unrelated one (swapped, not branched).
- uninterpretable: no regularity at most checkpoints.
- mixed: several of the above are clearly present (say which).
Note: steps 0-128 are close to random initialization; "patterns" there are mostly token frequency or token identity. Do not treat them as the core unless they actually persist.

Output only JSON:
{"label":"recursive_generalization|superposition|stable_monosemantic|replacement|uninterpretable|mixed",
 "confidence":0 to 1,
 "core":"core pattern (none if there is none)",
 "core_first_step":step where the core first appears, or null,
 "branches":[{"step":step it appears,"pattern":"branch pattern","grows_from":"which pattern it grows from and how they relate"}],
 "unrelated_patterns":["patterns unrelated to the core (evidence for superposition)"],
 "evidence":"reasons, at most 80 words",
 "summary":"one-sentence summary of this neuron's evolution"}"""

DIFF_SYS = """
Test the "diffusion / denoising" hypothesis of evolution. A diffusion model starts from noise and denoises step by step, fixing low-frequency coarse structure first and high-frequency detail later, getting sharper over time. For a neuron, the hypothesis predicts:
1. Early activation patterns are blurry, broad and unselective (noisy); later they converge gradually to the final concept, rather than appearing suddenly or switching to another concept midway.
2. Coarse before fine: coarse properties (domain / genre / broad category, i.e. coarse) approach the final state first; specific tokens and contexts (tok, doc) later.
3. Clarity (det, clarity, snr) rises roughly monotonically and saturates; the saturation level is limited by model and data scale and need not be high.
Alternatives: phase_transition (the final concept appears abruptly with no lead-up), early_fixed (fixed very early, barely changes, no denoising), drift (switches concept midway before settling), late_degradation (gets blurrier late), unclear.

Output only JSON:
{"label":"denoising|phase_transition|early_fixed|drift|late_degradation|unclear",
 "confidence":0 to 1,
 "commit_step":first step at which the final concept is recognizable, or null,
 "saturation_step":step after which clarity stops improving, or null,
 "coarse_before_fine":"yes|no|unclear",
 "coarse_evidence":"evidence on coarse vs fine order, at most 40 words",
 "clarity_trend":"monotonic_up|up_then_plateau|non_monotonic|flat|down",
 "final_clarity":"low|medium|high",
 "evidence":"reasons, at most 80 words",
 "summary":"one-sentence summary"}"""


def traj_intro(smp):
    return TRAJ_INTRO.format(k=len(smp["steps"]))


def traj_table(cfg, smp, l, n, expl, det, met, loss):
    steps_all = list(met["steps"])
    rows = []
    for s in smp["steps"]:
        e = expl.get(f"{l}_{n}_{s}", {}).get("explanation", {})
        d = det.get(f"{l}_{n}_{s}", {})
        si = steps_all.index(s)
        pats = "; ".join(f"{p.get('name', '')} [{p.get('trigger', '')}] ×{len(p.get('segments', []))}"
                         for p in e.get("patterns", []))[:300]
        q = {k: float(met[k][si, l, n]) for k in ("snr", "selectivity", "coarse_corr_final", "tok_overlap_final", "doc_jacc_final")}
        rows.append(
            f"## step {s} (loss {loss[s]:.2f})\n"
            f"det={d.get('score', float('nan')):.2f} clarity={e.get('clarity', '?')} snr={q['snr']:.2f} sel={q['selectivity']:.2f} "
            f"coarse={q['coarse_corr_final']:.2f} tok={q['tok_overlap_final']:.2f} doc={q['doc_jacc_final']:.2f}\n"
            f"concept: {e.get('concept', '?')}; relation: {e.get('relation', '?')}; unexplained: {len(e.get('unexplained', []))}/10\n"
            f"patterns: {pats}\ndescription: {e.get('description', '')}")
    return "\n\n".join(rows)


async def judge_worker(llm, p):
    rg, u1 = await llm.json_call(p["intro"] + RG_SYS, p["prompt"])
    df, u2 = await llm.json_call(p["intro"] + DIFF_SYS, p["prompt"])
    return dict(layer=p["layer"], neuron=p["neuron"], recursive=rg, diffusion=df,
                usage=dict(prompt=u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
                           completion=u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0)))


def load_metrics(cfg):
    met = np.load(out_path(cfg, "metrics", "all.npz"))
    return {k: met[k] for k in met.files}


def loss_by_step(met):
    import load
    ls, tot, _ = load.loss(list(met["steps"]))
    return dict(zip(ls.tolist(), tot.tolist()))


async def main_async(stage, limit):
    cfg = load_config()
    smp = json.load(open(out_path(cfg, "sample.json")))
    if limit:
        smp["neurons"] = smp["neurons"][:: max(1, len(smp["neurons"]) // limit)][:limit]
    llm = LLM(cfg)
    try:
        if stage in ("explain", "all"):
            await run_jobs(explain_jobs(cfg, smp), lambda p: explain_worker(llm, p),
                           out_path(cfg, "explanations.jsonl"), "explain")
        if stage in ("detect", "all"):
            expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
            await run_jobs(detect_jobs(cfg, smp, expl), lambda p: detect_worker(llm, p),
                           out_path(cfg, "detections.jsonl"), "detect")
        if stage in ("judge", "all"):
            expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
            det = read_jsonl(out_path(cfg, "detections.jsonl"))
            met = load_metrics(cfg)
            loss = loss_by_step(met)
            jobs = [(f"{l}_{n}", dict(layer=l, neuron=n, intro=traj_intro(smp),
                                      prompt=traj_table(cfg, smp, l, n, expl, det, met, loss)))
                    for l, n in smp["neurons"] if all(f"{l}_{n}_{s}" in expl for s in smp["steps"])]
            await run_jobs(jobs, lambda p: judge_worker(llm, p), out_path(cfg, "judgments.jsonl"), "judge")
    finally:
        await llm.close()
        llm.report(cfg, stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["explain", "detect", "judge", "all"])
    ap.add_argument("--limit", type=int, default=0, help="trial run on a few neurons")
    a = ap.parse_args()
    asyncio.run(main_async(a.stage, a.limit))


if __name__ == "__main__":
    main()
