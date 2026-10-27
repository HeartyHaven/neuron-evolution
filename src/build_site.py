"""Build the static explorer site: OUT/site/{index.html, data/meta.json, data/L<layer>.json, data/sim.json}.

    python src/build_site.py            # build
    python src/build_site.py --serve    # build and serve on 127.0.0.1:8766
"""
import argparse
import asyncio
import json
import os
import shutil

from common import load_config, out_path, read_jsonl, LLM, log
from lineage import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = ["Subword / character fragments", "Specific words or phrases", "Function words & syntax",
              "Semantic / topical concepts", "Domain / genre", "Formatting, punctuation & code symbols",
              "Numbers & math", "Non-English languages", "Mixed / hard to summarize"]

CAT_SYS = """You are sorting language-model neurons. Each neuron comes with its automatic explanations at the end of training (and two checkpoints before).
For each neuron:
1. Pick the single best "final activation type" from: """ + "; ".join(CATEGORIES) + """
2. Write a final concept label of at most 6 words (source tokens may be kept).
Output only JSON: {"items":[{"id":"L<layer>/N<index>","category":"...","label":"..."}]}"""

RG = {"recursive_generalization": "Grew out from a core (recursive generalization)", "superposition": "Unrelated things coexist (superposition)",
      "replacement": "Replaced by something else", "stable_monosemantic": "Always one thing", "mixed": "Mixed",
      "uninterpretable": "No clear pattern"}
DIFF = {"denoising": "Sharpened gradually (like denoising)", "drift": "Changed content midway", "phase_transition": "Appeared suddenly",
        "early_fixed": "Settled early", "late_degradation": "Got blurrier late", "unclear": "Unclear"}
DOMINANT = {"branch": "Mostly branches (recursive generalization)", "coexist": "Mostly unrelated coexistence",
            "replace": "Mostly replacement", "sporadic": "Mostly sporadic", "none": "No clear majority", "unknown": "Not analyzed"}


async def categorize(cfg, neurons, expl, judg, last3):
    path = out_path(cfg, "categories.json")
    if os.path.exists(path):
        return json.load(open(path))
    llm = LLM(cfg)

    async def batch(b):
        body = []
        for l, n in b:
            rows = [f"step{s}: concept={expl.get(f'{l}_{n}_{s}', {}).get('explanation', {}).get('concept')}; "
                    f"description={expl.get(f'{l}_{n}_{s}', {}).get('explanation', {}).get('description')}" for s in last3]
            core = judg.get(f"{l}_{n}", {}).get("recursive", {}).get("core", "")
            body.append(f"# L{l}/N{n}\n" + "\n".join(rows) + f"\ncore of its evolution: {core}")
        res, _ = await llm.json_call(CAT_SYS, "\n\n".join(body))
        return res["items"]

    try:
        out = await asyncio.gather(*(batch(neurons[i:i + 20]) for i in range(0, len(neurons), 20)))
    finally:
        await llm.close()
        llm.report(cfg, "categories")
    cats = {}
    for items in out:
        for it in items:
            l, n = it["id"].replace("L", "").replace("N", "").split("/")
            cat = it.get("category") if it.get("category") in CATEGORIES else CATEGORIES[-1]
            cats[f"{int(l)}_{int(n)}"] = dict(category=cat, label=it.get("label", ""))
    json.dump(cats, open(path, "w"), indent=1)
    return cats


def write_json(path, obj):
    # U+FFFD (an undecodable byte) is shown as a box
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("�", "▯"))


def build(cfg):
    import load
    smp = json.load(open(out_path(cfg, "sample.json")))
    neurons = [tuple(x) for x in smp["neurons"]]
    steps = smp["steps"]
    expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
    det = read_jsonl(out_path(cfg, "detections.jsonl"))
    judg = read_jsonl(out_path(cfg, "judgments.jsonl"))
    ctxs = {s: json.load(open(out_path(cfg, "contexts", f"step{s}.json"))) for s in steps}
    _, tot, _ = load.loss(steps)
    cats = asyncio.run(categorize(cfg, neurons, expl, judg, steps[-3:]))
    lin = analyze(cfg, smp, expl, ctxs)
    sem_path = out_path(cfg, "semantic_map.json")
    sem = json.load(open(sem_path)) if os.path.exists(sem_path) else {}

    nlist = []
    for l, n in neurons:
        k = f"{l}_{n}"
        rg, df = judg.get(k, {}).get("recursive", {}), judg.get(k, {}).get("diffusion", {})
        mk = lin.get(k, {})
        nlist.append(dict(
            id=k, layer=l, neuron=n,
            mech=mk.get("birth_shares"), mech_late=mk.get("birth_late"), dominant=mk.get("birth_dominant", "unknown"),
            core_summary=mk.get("core_summary"), lineage_shares=mk.get("lineage_shares", []), lineages=mk.get("lineages", []),
            category=cats.get(k, {}).get("category", CATEGORIES[-1]), label=cats.get(k, {}).get("label", ""),
            rg=rg.get("label", "unclear"), rg_conf=rg.get("confidence"), core=rg.get("core"),
            rg_summary=rg.get("summary"), rg_evidence=rg.get("evidence"),
            branches=rg.get("branches") or [], unrelated=rg.get("unrelated_patterns") or [],
            diff=df.get("label", "unclear"), diff_conf=df.get("confidence"), diff_summary=df.get("summary"),
            diff_evidence=df.get("evidence"), commit=df.get("commit_step"), saturation=df.get("saturation_step"),
            cbf=df.get("coarse_before_fine"), trend=df.get("clarity_trend"),
            det=[det.get(f"{k}_{s}", {}).get("score") for s in steps], sem=sem.get(k),
            cz=[expl.get(f"{k}_{s}", {}).get("explanation", {}).get("concept", "") for s in steps]))
    item = lambda d: {key: dict(name=v, desc="") for key, v in d.items()}
    groups = dict(
        category=dict(title="What it responds to by the end of training (AI-sorted)", order=CATEGORIES, items=item({c: c for c in CATEGORIES})),
        rg=dict(title="How it evolved (AI judgment)", order=list(RG), items=item(RG)),
        diff=dict(title="Did it sharpen gradually, like denoising? (AI judgment)", order=list(DIFF), items=item(DIFF)),
        dominant=dict(title="Which mechanism its strongest late responses mostly come from", order=list(DOMINANT), items=item(DOMINANT)),
        layer=dict(title="Layer (0 = closest to the input)", order=[str(l) for l in range(12)],
                   items={str(l): dict(name=f"Layer {l}", desc="") for l in range(12)}))

    root = os.path.join(cfg["OUT"], "site")
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    write_json(os.path.join(root, "data", "meta.json"),
               dict(steps=steps, loss=[round(float(x), 4) for x in tot], groups=groups, neurons=nlist))
    for L in range(12):
        data = {}
        for l, n in neurons:
            if l != L:
                continue
            k = f"{l}_{n}"
            pmap = lin.get(k, {}).get("pattern_lineage", {})
            rows = []
            for ti, s in enumerate(steps):
                e = expl.get(f"{k}_{s}", {}).get("explanation", {})
                c = ctxs[s][k]
                t1 = max(c["top1"], 1e-6)
                rows.append(dict(
                    step=s, det=det.get(f"{k}_{s}", {}).get("score"), clarity=e.get("clarity"),
                    concept=e.get("concept"), description=e.get("description"), relation=e.get("relation"),
                    patterns=[dict(name=p.get("name"), trigger=p.get("trigger"), context=p.get("context"), domain=p.get("domain"),
                                   segments=p.get("segments", []), lineage=pmap.get(f"t{ti}.{pi}"))
                              for pi, p in enumerate(e.get("patterns", []), 1)],
                    unexplained=e.get("unexplained", []), top1=c["top1"],
                    contexts=[dict(subset=x["subset"], doc=x["doc"], pos=x["pos"], value=x["stored"], hit=x["hit"],
                                   tokens=x["tokens"], a=[int(round(100 * v / t1)) for v in x["acts"]]) for x in c["top"]]))
            data[k] = rows
        write_json(os.path.join(root, "data", f"L{L}.json"), data)
    if os.path.exists(out_path(cfg, "semantic_sim.json")):
        shutil.copy(out_path(cfg, "semantic_sim.json"), os.path.join(root, "data", "sim.json"))
    body = open(os.path.join(HERE, "site.html"), encoding="utf-8").read()
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as f:
        f.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
                + body + "\n</body>\n</html>\n")
    log(f"site written to {root}")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    root = build(load_config())
    if a.serve:
        import functools
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        log(f"http://127.0.0.1:{a.port}/")
        ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(SimpleHTTPRequestHandler, directory=root)).serve_forever()


if __name__ == "__main__":
    main()
