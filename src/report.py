"""Summary: label statistics, coarse-to-fine test on all neurons, figures, a per-neuron catalog, a Claude audit
of the trajectory judges, and a Claude-written findings section.
Output: OUT/summary.json, OUT/figs/*.png, OUT/neuron_catalog.md, OUT/claude_audit.json, OUT/report.md
"""
import json
import os
import sys
from collections import Counter

import matplotlib
import numpy as np
from scipy.stats import spearmanr

import agents
from common import load_config, out_path, read_jsonl, parse_json, log

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = {"shallow 0-3": range(0, 4), "middle 4-7": range(4, 8), "deep 8-11": range(8, 12)}
DRIFT_STEPS = [256, 1000, 2000, 4000, 8000, 16000, 32000]


def first_cross(r, thr=0.5):
    idx = np.argmax(r >= thr, axis=0)
    idx[~(r >= thr).any(axis=0)] = r.shape[0] - 1
    return idx


def quantitative(met):
    """Median curves and, per layer group, when coarse/token/doc similarity to the final step first passes halfway."""
    steps = met["steps"]
    res = dict(steps=steps.tolist(), median_curves={}, convergence={})
    for k in ["snr", "selectivity", "hist_entropy", "frac_pos", "coarse_corr_final", "tok_overlap_final", "doc_jacc_final", "doc_jacc_prev"]:
        res["median_curves"][k] = np.median(met[k], axis=2).round(4).tolist()
    conv = {}
    for k, name in (("coarse_corr_final", "coarse"), ("tok_overlap_final", "token"), ("doc_jacc_final", "doc")):
        m = met[k].reshape(len(steps), -1).astype(np.float64)
        conv[name] = first_cross((m - m[0]) / np.maximum(1 - m[0], 1e-6), 0.5)
    L, N = met["snr"].shape[1:]
    for g, layers in GROUPS.items():
        sel = np.zeros((L, N), bool)
        sel[list(layers)] = True
        sel = sel.reshape(-1)
        c = {name: dict(median_step=int(steps[int(np.median(v[sel]))]), q25=int(steps[int(np.percentile(v[sel], 25))]),
                        q75=int(steps[int(np.percentile(v[sel], 75))])) for name, v in conv.items()}
        c["frac_coarse_before_doc"] = float((conv["coarse"][sel] < conv["doc"][sel]).mean())
        c["frac_token_before_doc"] = float((conv["token"][sel] < conv["doc"][sel]).mean())
        c["frac_coarse_before_token"] = float((conv["coarse"][sel] < conv["token"][sel]).mean())
        res["convergence"][g] = c
    res["coarse_corr_step0_median"] = float(np.median(met["coarse_corr_final"][0]))
    rng = np.random.default_rng(0)
    snr = met["snr"].reshape(len(steps), -1)
    rk = np.apply_along_axis(lambda c: spearmanr(np.log1p(steps), c).statistic, 0,
                             snr[:, rng.choice(snr.shape[1], 3000, replace=False)])
    res["snr_spearman_logstep"] = dict(median=float(np.nanmedian(rk)), frac_positive=float(np.nanmean(rk > 0)))
    return res


def drift_control(cfg):
    path = out_path(cfg, "metrics", "ref.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    steps, refs = d["steps"].tolist(), d["refs"].tolist()
    i64, i143 = refs.index(64000), refs.index(143000)
    res = {}
    for g, layers in GROUPS.items():
        res[g] = {}
        for m in ("coarse", "tok", "doc"):
            a = d[m][:, :, list(layers), :].reshape(len(refs), len(steps), -1)
            res[g][m] = dict(steps=DRIFT_STEPS,
                             sim_to_64000=[float(np.median(a[i64, steps.index(s)])) for s in DRIFT_STEPS],
                             sim_to_143000=[float(np.median(a[i143, steps.index(s)])) for s in DRIFT_STEPS],
                             step64000_vs_143000=float(np.median(a[i143, steps.index(64000)])))
    return res


def figures(cfg, q, det_curve, clar_curve, judg, smp):
    fd = os.path.dirname(out_path(cfg, "figs", "x"))
    x = np.array(q["steps"]) + 1
    cmap = plt.get_cmap("viridis")
    keys = [("snr", "SNR (top-64 z-score)"), ("selectivity", "Domain selectivity"),
            ("coarse_corr_final", "Coarse: subset preference vs final"), ("tok_overlap_final", "Mid: top tokens vs final"),
            ("doc_jacc_final", "Fine: top documents vs final"), ("doc_jacc_prev", "Top documents vs previous step")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, t) in zip(axs.flat, keys):
        arr = np.array(q["median_curves"][k])
        for l in range(arr.shape[1]):
            ax.plot(x, arr[:, l], color=cmap(l / 11), lw=1.2, label=f"L{l}" if k == "snr" else None)
        ax.set_xscale("log")
        ax.set_title(t)
        ax.set_xlabel("step + 1")
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Median over all 36,864 neurons, by layer")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "quantitative_curves.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos, ticks = 0, []
    for gi, (g, cv) in enumerate(q["convergence"].items()):
        for ni, nm in enumerate(["coarse", "token", "doc"]):
            c = cv[nm]
            ax.errorbar(pos, c["median_step"] + 1, yerr=[[c["median_step"] - c["q25"]], [c["q75"] - c["median_step"]]],
                        fmt="o", color=["#e76f51", "#2a9d8f", "#264653"][ni], capsize=4, label=nm if gi == 0 else None)
            pos += 1
        ticks.append(pos - 2)
        pos += 1
    ax.set_yscale("log")
    ax.set_xticks(ticks, list(q["convergence"]))
    ax.set_ylabel("step where similarity to final passes halfway")
    ax.legend()
    ax.set_title("Coarse before fine?")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "coarse_to_fine.png"), dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    sx = np.array(smp["steps"]) + 1
    for g in GROUPS:
        axs[0].plot(sx, det_curve[g], marker="o", ms=3, label=g)
        axs[1].plot(sx, clar_curve[g], marker="o", ms=3, label=g)
    for ax, t in zip(axs, ["Detection score (balanced accuracy, 0.5 = chance)", "Explainer clarity (1-5)"]):
        ax.set_xscale("log")
        ax.set_title(t)
        ax.set_xlabel("step + 1")
        ax.legend()
    axs[0].axhline(0.5, color="gray", ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "clarity_curves.png"), dpi=120)
    plt.close(fig)

    dc = q.get("drift_control")
    if dc:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, m in zip(axs, ["coarse", "tok", "doc"]):
            for gi, g in enumerate(GROUPS):
                c, col = dc[g][m], plt.get_cmap("tab10")(gi)
                ax.plot(np.array(c["steps"]) + 1, c["sim_to_64000"], "--", color=col, marker="o", ms=3, label=f"{g} vs 64000")
                ax.plot(np.array(c["steps"]) + 1, c["sim_to_143000"], "-", color=col, marker="o", ms=3, label=f"{g} vs 143000")
            ax.set_xscale("log")
            ax.set_title(m)
            ax.set_xlabel("step + 1")
        axs[0].legend(fontsize=7)
        fig.suptitle("Drift control: similarity to two late references (solid ~ dashed: converging; dashed higher: still drifting)")
        fig.tight_layout()
        fig.savefig(os.path.join(fd, "drift_control.png"), dpi=120)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, field in ((axs[0], "recursive"), (axs[1], "diffusion")):
        bottom = np.zeros(12)
        for i, lab in enumerate(sorted({j[field].get("label", "?") for j in judg.values()})):
            cnt = np.array([sum(1 for j in judg.values() if j["layer"] == l and j[field].get("label") == lab) for l in range(12)])
            ax.bar(range(12), cnt, bottom=bottom, label=lab, color=plt.get_cmap("tab10")(i))
            bottom += cnt
        ax.set_xticks(range(12))
        ax.set_xlabel("layer")
        ax.set_title(field)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "labels_by_layer.png"), dpi=120)
    plt.close(fig)


def catalog(cfg, smp, expl, det, judg):
    lines = ["# Neuron catalog", ""]
    for l, n in smp["neurons"]:
        j = judg.get(f"{l}_{n}", {})
        rg, df = j.get("recursive", {}), j.get("diffusion", {})
        lines += [f"## L{l} / N{n}", "",
                  f"- **Evolution:** {rg.get('label')} ({rg.get('confidence')}); core: {rg.get('core')}; {rg.get('summary', '')}",
                  f"- **Denoising:** {df.get('label')} ({df.get('confidence')}); commit step {df.get('commit_step')}; "
                  f"coarse before fine: {df.get('coarse_before_fine')}; {df.get('summary', '')}"]
        if rg.get("branches"):
            lines.append("- **Branches:** " + "; ".join(f"step{b.get('step')} {b.get('pattern')} <- {b.get('grows_from')}" for b in rg["branches"]))
        lines += ["", "| step | det | clarity | concept | description |", "|---|---|---|---|---|"]
        for s in smp["steps"]:
            e = expl.get(f"{l}_{n}_{s}", {}).get("explanation", {})
            d = det.get(f"{l}_{n}_{s}", {})
            desc = str(e.get("description", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {s} | {d.get('score', float('nan')):.2f} | {e.get('clarity', '')} | "
                         f"{str(e.get('concept', '')).replace('|', '/')} | {desc} |")
        lines.append("")
    open(out_path(cfg, "neuron_catalog.md"), "w").write("\n".join(lines))


def claude(cfg, prompt):
    import anthropic
    r = anthropic.Anthropic().beta.messages.create(
        model=cfg["CLAUDE_MODEL"], max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt}])
    if r.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    return "".join(b.text for b in r.content if b.type == "text"), r.usage


AUDIT_SYS = """You are auditing another model's judgments. Below are several neurons, each with observations at {k} checkpoints. For each neuron, make two independent judgments using the criteria below.

[Judgment 1: evolution type]
{rg}

[Judgment 2: denoising hypothesis]
{df}

Give only labels and confidences. Output only one JSON object:
{{"results":[{{"id":"L<layer>/N<index>","rg_label":"...","rg_confidence":0 to 1,"diff_label":"...","coarse_before_fine":"yes|no|unclear","diff_confidence":0 to 1,"note":"at most 20 words"}}]}}"""


def claude_audit(cfg, smp, expl, det, met, judg, n_per_layer=2, batch=8):
    """Claude re-judges a random subset with the same criteria, to estimate how reliable the judges are."""
    path = out_path(cfg, "claude_audit.json")
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.default_rng(cfg["SEED"] + 2)
    picks = []
    for l in range(12):
        cand = [(a, b) for a, b in smp["neurons"] if a == l and f"{a}_{b}" in judg]
        picks += [cand[i] for i in sorted(rng.choice(len(cand), min(n_per_layer, len(cand)), replace=False))]
    loss = agents.loss_by_step(met)
    strip = lambda s: s.split("Output only JSON")[0].strip()
    sys_prompt = AUDIT_SYS.format(k=len(smp["steps"]), rg=strip(agents.RG_SYS), df=strip(agents.DIFF_SYS))
    results, tokens = [], [0, 0]
    for i in range(0, len(picks), batch):
        body = "\n\n".join(f"# Neuron L{l}/N{n}\n" + agents.traj_table(cfg, smp, l, n, expl, det, met, loss)
                           for l, n in picks[i:i + batch])
        text, u = claude(cfg, sys_prompt + "\n\n" + agents.traj_intro(smp) + "\n\n" + body)
        tokens[0] += u.input_tokens
        tokens[1] += u.output_tokens
        results += parse_json(text)["results"]
        log(f"Claude audit {min(i + batch, len(picks))}/{len(picks)}")
    agree = {"rg": [], "diff": [], "cbf": []}
    for r in results:
        l, n = r["id"].replace("L", "").replace("N", "").split("/")
        j = judg.get(f"{int(l)}_{int(n)}")
        if j:
            agree["rg"].append(r.get("rg_label") == j["recursive"].get("label"))
            agree["diff"].append(r.get("diff_label") == j["diffusion"].get("label"))
            agree["cbf"].append(r.get("coarse_before_fine") == j["diffusion"].get("coarse_before_fine"))
    out = dict(n=len(results), input_tokens=tokens[0], output_tokens=tokens[1], results=results,
               agreement={k: float(np.mean(v)) if v else None for k, v in agree.items()},
               claude_rg_labels=dict(Counter(r.get("rg_label") for r in results)),
               claude_diff_labels=dict(Counter(r.get("diff_label") for r in results)),
               judge_rg_labels_same_neurons=dict(Counter(judg[f"{l}_{n}"]["recursive"].get("label") for l, n in picks)),
               judge_diff_labels_same_neurons=dict(Counter(judg[f"{l}_{n}"]["diffusion"].get("label") for l, n in picks)))
    json.dump(out, open(path, "w"), indent=1)
    return out


REPORT_PROMPT = """You are an ML interpretability researcher. Below are automated analysis results on the evolution of Pythia-160M MLP neurons over training. Write the "Findings" section of an English report.

Research questions:
1. Is "recursive generalization" (Goodfire, "Neural networks are already clean": a neuron grows related branches out of a core, so it looks polysemantic but is traceable) common here, or does it look more like superposition (unrelated features coexisting)? How does it vary with depth?
2. Hypothesis: neuron evolution resembles diffusion denoising - blurry at first, gradually sharper, coarse structure before details, with final sharpness limited by model and data scale. Does the evidence support or contradict it?
Method: {per_layer} random neurons per layer x {k} checkpoints; an LLM explainer reads top-10 contexts; detection scoring uses 8 held-out strong activations + 8 random texts to measure predictive power; two judge agents give an evolution type and a denoising verdict per neuron. Quantitative metrics (no LLM) cover all 36,864 neurons.

Requirements:
- Conclusions first, then evidence; separate agent judgments from quantitative metrics and note where they agree or conflict.
- Cite specific neurons (L<layer>/N<index>) to illustrate branch chains or denoising.
- When reading the coarse-to-fine evidence, account for two confounds: (a) any drifting neuron becomes more similar to the final checkpoint near the end; summary.quantitative.drift_control compares each step with step64000 and step143000 - similar values mean convergence to a fixed target, much higher similarity to 64000 means ongoing drift; (b) subset preference averages over 10M tokens and is inherently more stable than top-64 extremes, so part of "coarse first" may come from statistic stability.
- State limitations: labeling-model ability, sample size, detection noise, judges possibly led by the prompt (e.g. calling anything related a "branch"), the late-training loss rise of Pythia-160M.
- Propose 3-5 follow-up experiments that would separate the hypotheses.
- Markdown, start at level-2 headings, at most 800 words.

Data:
"""


def main():
    no_claude = "--no-claude" in sys.argv
    cfg = load_config()
    smp = json.load(open(out_path(cfg, "sample.json")))
    expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
    det = read_jsonl(out_path(cfg, "detections.jsonl"))
    judg = read_jsonl(out_path(cfg, "judgments.jsonl"))
    met = agents.load_metrics(cfg)
    need = len(smp["neurons"]) * len(smp["steps"])
    if "--force" not in sys.argv and (len(expl) < need or len(det) < need or len(judg) < len(smp["neurons"])):
        raise SystemExit(f"incomplete: explanations {len(expl)}/{need}, detections {len(det)}/{need}, judgments {len(judg)}")

    q = quantitative(met)
    q["drift_control"] = drift_control(cfg)
    det_curve, clar_curve = {}, {}
    for g, layers in GROUPS.items():
        det_curve[g] = [float(np.nanmean([v["score"] for v in det.values() if v["step"] == s and v["layer"] in layers] or [np.nan]))
                        for s in smp["steps"]]
        clar_curve[g] = [float(np.nanmean([float(v["explanation"].get("clarity", np.nan)) for v in expl.values()
                                           if v["step"] == s and v["layer"] in layers] or [np.nan])) for s in smp["steps"]]
    mono = []
    for l, n in smp["neurons"]:
        ys = [det.get(f"{l}_{n}_{s}", {}).get("score", np.nan) for s in smp["steps"]]
        if not np.isnan(ys).all():
            mono.append(spearmanr(np.log1p(smp["steps"]), ys, nan_policy="omit").statistic)
    commit = [j["diffusion"].get("commit_step") for j in judg.values() if isinstance(j["diffusion"].get("commit_step"), (int, float))]
    summary = dict(
        n_neurons=len(smp["neurons"]), steps=smp["steps"],
        recursive_overall=dict(Counter(j["recursive"].get("label") for j in judg.values())),
        recursive_by_layer={l: dict(Counter(j["recursive"].get("label") for j in judg.values() if j["layer"] == l)) for l in range(12)},
        diffusion_overall=dict(Counter(j["diffusion"].get("label") for j in judg.values())),
        diffusion_by_layer={l: dict(Counter(j["diffusion"].get("label") for j in judg.values() if j["layer"] == l)) for l in range(12)},
        coarse_before_fine=dict(Counter(j["diffusion"].get("coarse_before_fine") for j in judg.values())),
        clarity_trend=dict(Counter(j["diffusion"].get("clarity_trend") for j in judg.values())),
        commit_step_quartiles=np.percentile(commit, [25, 50, 75]).tolist() if commit else None,
        detection_curve=det_curve, clarity_curve=clar_curve,
        detection_spearman_logstep=dict(median=float(np.nanmedian(mono)), frac_positive=float(np.nanmean(np.array(mono) > 0))),
        quantitative=dict(convergence=q["convergence"], snr_spearman_logstep=q["snr_spearman_logstep"],
                          coarse_corr_step0_median=q["coarse_corr_step0_median"],
                          median_snr_by_layer_final=q["median_curves"]["snr"][-1],
                          median_selectivity_by_layer_final=q["median_curves"]["selectivity"][-1],
                          drift_control=q["drift_control"]))
    figures(cfg, q, det_curve, clar_curve, judg, smp)
    catalog(cfg, smp, expl, det, judg)

    compact = []
    for l, n in smp["neurons"]:
        j = judg.get(f"{l}_{n}")
        if j:
            rg, df = j["recursive"], j["diffusion"]
            compact.append(dict(id=f"L{l}/N{n}", rg=rg.get("label"), rg_conf=rg.get("confidence"), core=rg.get("core"),
                                branches=[f"{b.get('step')}:{b.get('pattern')}" for b in (rg.get("branches") or [])][:5],
                                rg_summary=rg.get("summary"), diff=df.get("label"), commit=df.get("commit_step"),
                                cbf=df.get("coarse_before_fine"), trend=df.get("clarity_trend"), diff_summary=df.get("summary")))
    findings, audit = "(Claude not called)", None
    if not no_claude:
        audit = claude_audit(cfg, smp, expl, det, met, judg)
        summary["claude_audit"] = {k: v for k, v in audit.items() if k != "results"}
        prompt = REPORT_PROMPT.format(per_layer=cfg["NEURONS_PER_LAYER"], k=len(smp["steps"]))
        findings, _ = claude(cfg, prompt + json.dumps(dict(summary=summary, neurons=compact)))
    json.dump(summary, open(out_path(cfg, "summary.json"), "w"), indent=1)

    figs = ["quantitative_curves", "coarse_to_fine", "clarity_curves", "drift_control", "labels_by_layer"]
    agree = (f" Agreement with the judges: evolution type {audit['agreement']['rg']:.0%}, denoising {audit['agreement']['diff']:.0%}, "
             f"coarse-before-fine {audit['agreement']['cbf']:.0%}." if audit else "")
    lines = ["# Neuron evolution: recursive generalization and the denoising hypothesis", "",
             f"Pythia-160M, {len(smp['steps'])} checkpoints, {len(smp['neurons'])} sampled neurons; explainer and judges: "
             f"{cfg['OPENAI_MODEL']} (effort {cfg['OPENAI_EFFORT']}); audit and findings: {cfg['CLAUDE_MODEL']}.{agree}", "",
             "## Findings (written by Claude)", "", findings, ""] + [f"![{f}](figs/{f}.png)\n" for f in figs]
    open(out_path(cfg, "report.md"), "w").write("\n".join(lines) + "\n")
    log("report written")


if __name__ == "__main__":
    main()
