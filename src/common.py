"""Config, paths, JSONL checkpointing and the OpenAI client shared by all scripts."""
import asyncio
import json
import os
import re
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.environ.get("NEVO_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("CKPT_ROOT", "VAL_PATH", "OUT"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))
    cfg["N_TOKENS"] = int(cfg["N_TOKENS"])
    os.makedirs(cfg["OUT"], exist_ok=True)
    return cfg


def out_path(cfg, *parts):
    p = os.path.join(cfg["OUT"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def probe_dir(cfg):
    return os.path.join(cfg["OUT"], "probe")


def acts_dir(cfg, step=None):
    d = os.path.join(cfg["OUT"], "acts")
    return d if step is None else os.path.join(d, f"step{step}")


def ckpt_dir(cfg, step):
    return os.path.join(cfg["CKPT_ROOT"], f"step{step}")


def available_steps(cfg):
    return sorted(int(n[4:]) for n in os.listdir(cfg["CKPT_ROOT"]) if n.startswith("step") and n[4:].isdigit())


def step_order(cfg):
    """TIER_B_STEPS first, then every other downloaded checkpoint."""
    avail = available_steps(cfg)
    tier_b = [s for s in cfg["TIER_B_STEPS"] if s in avail]
    return tier_b + [s for s in avail if s not in set(tier_b)]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["key"]] = r
    return rows


def parse_json(text):
    """First JSON object in a model reply; tolerates code fences and stray backslashes."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    s = text[text.find("{"):]
    dec = json.JSONDecoder(strict=False)
    try:
        return dec.raw_decode(s)[0]
    except json.JSONDecodeError:
        return dec.raw_decode(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s))[0]


class LLM:
    """Async OpenAI Responses client returning parsed JSON, with a concurrency cap and usage totals.
    Transport errors (429, 5xx, timeouts) are retried by the SDK; malformed JSON counts toward LLM_MAX_TRIES."""

    def __init__(self, cfg):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(max_retries=8, timeout=600)
        self.model, self.effort = cfg["OPENAI_MODEL"], cfg["OPENAI_EFFORT"]
        self.max_tries = cfg["LLM_MAX_TRIES"]
        self.sem = asyncio.Semaphore(cfg["LLM_CONCURRENCY"])
        self.usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, failures=0)

    async def json_call(self, system, user):
        for attempt in range(1, self.max_tries + 1):
            async with self.sem:
                r = await self.client.responses.create(
                    model=self.model, instructions=system, input=user, reasoning={"effort": self.effort})
            u = dict(prompt_tokens=r.usage.input_tokens, completion_tokens=r.usage.output_tokens) if r.usage else {}
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
            try:
                return parse_json(r.output_text), u
            except (ValueError, IndexError) as e:
                if attempt == self.max_tries:
                    self.usage["failures"] += 1
                    raise RuntimeError(f"no valid JSON after {attempt} tries: {e}")
                log(f"bad JSON (try {attempt}): {str(e)[:120]}; retrying")

    async def close(self):
        await self.client.close()

    def report(self, cfg, stage):
        u = self.usage
        log(f"{stage}: {u['calls']} calls, {u['prompt_tokens']:,} in / {u['completion_tokens']:,} out tokens, {u['failures']} failures")
        with open(out_path(cfg, "llm_usage.jsonl"), "a") as f:
            f.write(json.dumps(dict(stage=stage, **u)) + "\n")


async def run_jobs(jobs, worker, path, desc):
    """jobs: [(key, payload)]. Keys already in `path` are skipped; results are appended as they finish."""
    done = read_jsonl(path)
    todo = [(k, p) for k, p in jobs if k not in done]
    log(f"{desc}: {len(jobs)} total, {len(done)} done, {len(todo)} to do")
    if not todo:
        return done
    lock = asyncio.Lock()
    n_ok = n_fail = 0
    t0 = time.time()

    with open(path, "a") as f:
        async def one(k, p):
            nonlocal n_ok, n_fail
            try:
                res = await worker(p)
            except Exception as e:
                n_fail += 1
                log(f"{desc} {k} failed: {e}")
                return
            res["key"] = k
            async with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                done[k] = res
                n_ok += 1
                if n_ok % 100 == 0 or n_ok == len(todo):
                    el = time.time() - t0
                    log(f"{desc}: {n_ok}/{len(todo)} ({n_fail} failed), {el:.0f}s, ~{el / n_ok * (len(todo) - n_ok):.0f}s left")

        await asyncio.gather(*(one(k, p) for k, p in todo))
    if n_fail:
        raise SystemExit(f"{desc}: {n_fail} jobs failed; rerun to resume")
    return done
