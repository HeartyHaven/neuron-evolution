"""Group each neuron's per-step patterns into lineages (one concept persisting across steps) and derive
how each lineage was born:
  branch    derived from an earlier lineage (any lineage can be a parent)
  sporadic  appears at a single step, no parent and no children
  replace   born right after a non-sporadic lineage died, taking its place
  coexist   everything else: arrives from outside and coexists
LLM output goes to OUT/lineages.jsonl; analyze() turns it into activation shares.
"""
import asyncio
import json
import sys

import numpy as np

from common import load_config, out_path, read_jsonl, run_jobs, LLM

BIRTH = ["branch", "coexist", "replace", "sporadic"]
LATE_STEP = 16000

LINEAGE_SYS = """Below are explanations of one MLP neuron of Pythia-160M at several training checkpoints. Each checkpoint lists the patterns found by the explainer; each pattern has an id (e.g. t5.2 = pattern 2 at checkpoint 5), a name, a trigger token, and which of that checkpoint's top-10 strongest-activation texts it covers (×k = covers k texts). Step 0 is the random initialization.

Task:
1. Chain all patterns into lineages, where a lineage is the same concept continuing across checkpoints. Every pattern id must be assigned to a lineage.
2. For each lineage, decide whether it derives from an earlier lineage. The parent may be any lineage that appeared earlier, not only the core.
   It counts as derived whenever you can name a concrete relation: form (e.g. "iy" -> "diy"), meaning (e.g. "in" -> "within" -> "outside"), or function/usage (e.g. meeting agendas -> parliamentary votes and speeches).
   Belonging to the same kind of thing also counts as a semantic relation (e.g. London -> Amsterdam, both city names; weaken -> specific things being weakened); pick the most direct parent, usually the most recent lineage of the same kind.
   If you can name the relation, fill parent and how_related; set parent to null only when no relation can be named (a new concept coming from outside).
3. Mark the core lineage: the one that stably carries the main concept late in training; at most one, none if there is no stable main concept late. The core may itself have a parent.

Output only JSON:
{"core_summary":"core concept, or none",
 "lineages":[{"id":"A","name":"lineage name","is_core":false,"parent":"parent lineage id or null","how_related":"concrete relation to the parent, at most 20 words; empty when parent is null","first_step":step of first appearance}],
 "assign":{"t0.1":"A","t0.2":"B"}}"""


def build_prompt(smp, l, n, expl, ctxs):
    """Prompt text plus {pattern id: (step, covered segments)}."""
    lines, pid = [], {}
    for ti, s in enumerate(smp["steps"]):
        e = expl[f"{l}_{n}_{s}"]["explanation"]
        nseg = len(ctxs[s][f"{l}_{n}"]["top"])
        rows = [f"## t{ti}: step {s} (concept: {e.get('concept', '')})"]
        for k, p in enumerate(e.get("patterns", []), 1):
            segs = [int(x) for x in p.get("segments", []) if str(x).isdigit() or isinstance(x, int)]
            segs = [x for x in segs if 1 <= x <= nseg]
            pid[f"t{ti}.{k}"] = (s, segs)
            rows.append(f"t{ti}.{k} {p.get('name', '')} [{p.get('trigger', '')}] ×{len(segs)}; {p.get('context', '')}")
        if e.get("unexplained"):
            rows.append(f"(unexplained texts: {len(e['unexplained'])})")
        lines.append("\n".join(rows))
    return "\n\n".join(lines), pid


def clean_lineages(res):
    ids = {g.get("id") for g in res.get("lineages", [])}
    out = []
    for g in res.get("lineages", []):
        p = g.get("parent")
        out.append(dict(id=g.get("id"), name=g.get("name"), parent=p if p in ids and p != g.get("id") else None,
                        how=g.get("how_related") or "", first_step=g.get("first_step"), core=bool(g.get("is_core"))))
    return out


def lineage_shares(smp, l, n, ctxs, pid, assign):
    """Per step, each lineage's share of the top-10 activation mass. A text covered by several lineages is split
    evenly; uncovered texts count as unexplained (the remainder)."""
    out = []
    for s in smp["steps"]:
        vals = [max(x["stored"], 0.0) for x in ctxs[s][f"{l}_{n}"]["top"]]
        seg = [[] for _ in vals]
        for key, (st, segs) in pid.items():
            if st == s and assign.get(key):
                for x in segs:
                    seg[x - 1].append(assign[key])
        mass = {}
        for v, ids in zip(vals, seg):
            for g in ids:
                mass[g] = mass.get(g, 0.0) + v / len(ids)
        tot = sum(vals) or 1.0
        out.append({g: m / tot for g, m in mass.items()})
    return out


def birth_mechanisms(lineages, lin_shares):
    k = len(lin_shares)
    ids = {g["id"] for g in lineages}
    first, last = {}, {}
    for g in lineages:
        pres = [j for j, sh in enumerate(lin_shares) if sh.get(g["id"], 0) > 0]
        first[g["id"]] = pres[0] if pres else None
        last[g["id"]] = pres[-1] if pres else None
    parent_of = {g["id"]: g["parent"] for g in lineages
                 if g["parent"] in ids and first[g["id"]] is not None and first.get(g["parent"]) is not None}
    has_child = set(parent_of.values())
    sporadic = {g for g in ids if g not in parent_of and g not in has_child and first[g] is not None and first[g] == last[g]}
    out, pending = {}, []
    for g in lineages:
        gid, f = g["id"], first[g["id"]]
        info = dict(first=f, parent=None, replaces=None, birth=None)
        if f is None:
            pass
        elif gid in parent_of:
            info.update(birth="branch", parent=parent_of[gid])
        elif gid in sporadic:
            info["birth"] = "sporadic"
        elif f == 0:
            info["birth"] = "coexist"
        else:
            pending.append(gid)
        out[gid] = info

    def depends_on(x, target):
        """Whether x's line depends on target via derived-from or replaced links (prevents drawing cycles)."""
        stack, seen = [x], set()
        while stack:
            y = stack.pop()
            if y in seen:
                continue
            seen.add(y)
            for z in (parent_of.get(y), out.get(y, {}).get("replaces")):
                if z == target:
                    return True
                if z is not None:
                    stack.append(z)
        return False

    for f in range(1, k):
        # pair newborns with lineages that died at the previous step, largest share first; ties broken by id
        newborn = sorted([g for g in pending if first[g] == f], key=lambda g: (-lin_shares[f].get(g, 0), g))
        dead = sorted([g for g in ids if last[g] == f - 1 and g not in sporadic], key=lambda g: (-lin_shares[f - 1].get(g, 0), g))
        for nb in newborn:
            dd = next((d for d in dead if not depends_on(d, nb)), None)
            if dd is None:
                out[nb]["birth"] = "coexist"
            else:
                dead.remove(dd)
                out[nb].update(birth="replace", replaces=dd)
    return out


def birth_shares(lin_shares, births):
    out = {m: [] for m in BIRTH + ["unexplained"]}
    for sh in lin_shares:
        acc = {m: 0.0 for m in BIRTH}
        for gid, v in sh.items():
            b = births.get(gid, {}).get("birth")
            if b in acc:
                acc[b] += v
        for m in BIRTH:
            out[m].append(round(acc[m], 4))
        out["unexplained"].append(round(max(0.0, 1.0 - sum(acc.values())), 4))
    return out


def analyze(cfg, smp, expl, ctxs):
    """{neuron id: lineages with birth mechanism, per-step shares, late-training averages}."""
    raw = read_jsonl(out_path(cfg, "lineages.jsonl"))
    late = [i for i, s in enumerate(smp["steps"]) if s >= LATE_STEP]
    out = {}
    for l, n in smp["neurons"]:
        k = f"{l}_{n}"
        if k not in raw:
            continue
        res = raw[k]["raw"]
        assign = res.get("assign", {})
        _, pid = build_prompt(smp, l, n, expl, ctxs)
        lins = clean_lineages(res)
        lsh = lineage_shares(smp, l, n, ctxs, pid, assign)
        births = birth_mechanisms(lins, lsh)
        bsh = birth_shares(lsh, births)
        blate = {m: round(float(np.mean([bsh[m][i] for i in late])), 3) for m in BIRTH + ["unexplained"]}
        top = max(BIRTH, key=lambda m: blate[m])
        for g in lins:
            b = births[g["id"]]
            g.update(birth=b["birth"], birth_parent=b["parent"], replaces=b["replaces"])
        out[k] = dict(core_summary=res.get("core_summary"), lineages=lins, pattern_lineage=assign,
                      lineage_shares=[{g: round(v, 3) for g, v in x.items()} for x in lsh],
                      birth_shares=bsh, birth_late=blate, birth_dominant=top if blate[top] > 0.5 else "none")
    return out


async def main_async(limit=0):
    cfg = load_config()
    smp = json.load(open(out_path(cfg, "sample.json")))
    expl = read_jsonl(out_path(cfg, "explanations.jsonl"))
    ctxs = {s: json.load(open(out_path(cfg, "contexts", f"step{s}.json"))) for s in smp["steps"]}
    llm = LLM(cfg)

    async def worker(p):
        prompt, pid = build_prompt(smp, p["layer"], p["neuron"], expl, ctxs)
        res, u = await llm.json_call(LINEAGE_SYS, prompt)
        missing = [k for k in pid if k not in res.get("assign", {})]
        if len(missing) > 0.2 * len(pid):
            raise ValueError(f"{len(missing)}/{len(pid)} patterns not assigned to a lineage")
        return dict(layer=p["layer"], neuron=p["neuron"], raw=res, n_patterns=len(pid), n_unassigned=len(missing),
                    usage=dict(prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens")))

    neurons = smp["neurons"][:limit] if limit else smp["neurons"]
    jobs = [(f"{l}_{n}", dict(layer=l, neuron=n)) for l, n in neurons]
    try:
        await run_jobs(jobs, worker, out_path(cfg, "lineages.jsonl"), "lineage")
    finally:
        await llm.close()
        llm.report(cfg, "lineage")


if __name__ == "__main__":
    asyncio.run(main_async(int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0))
