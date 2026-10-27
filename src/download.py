"""Download Pythia checkpoints (one HF revision per step) and the Pile validation split.

    python src/download.py              # the analysis STEPS only
    python src/download.py --all        # all 154 checkpoints (needed for the full metric curves)
"""
import argparse
import os

from huggingface_hub import hf_hub_download, list_repo_refs, snapshot_download

from common import load_config, ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", help="comma-separated steps")
    a = ap.parse_args()
    cfg = load_config()
    if a.steps:
        steps = [int(s) for s in a.steps.split(",")]
    elif a.all:
        steps = sorted(int(b.name[4:]) for b in list_repo_refs(cfg["MODEL"]).branches if b.name[4:].isdigit())
    else:
        steps = sorted(set(cfg["STEPS"]) | {143000})
    for s in steps:
        d = ckpt_dir(cfg, s)
        if not os.path.exists(os.path.join(d, "pytorch_model.bin")):
            snapshot_download(cfg["MODEL"], revision=f"step{s}", local_dir=d,
                              allow_patterns=["*.json", "pytorch_model.bin"])
        print(f"step{s} ok", flush=True)
    os.makedirs(os.path.dirname(cfg["VAL_PATH"]), exist_ok=True)
    hf_hub_download("monology/pile-uncopyrighted", "val.jsonl.zst", repo_type="dataset",
                    local_dir=os.path.dirname(cfg["VAL_PATH"]))
    print(cfg["VAL_PATH"], "ok")


if __name__ == "__main__":
    main()
