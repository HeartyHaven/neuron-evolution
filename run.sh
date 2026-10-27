#!/bin/bash
# Full pipeline. Every stage skips finished work, so rerunning resumes.
set -e -o pipefail
cd "$(dirname "$0")"
python src/download.py --all      # 154 checkpoints (~58 GB) + Pile val
python src/build_probe.py         # 10M-token probe set
python src/extract.py             # activation summaries, all neurons x all checkpoints (~14 GB)
python src/prep.py all            # sample neurons, contexts, metrics
python src/agents.py explain
python src/agents.py detect
python src/agents.py judge
python src/lineage.py
python src/semantic_map.py
python src/report.py              # needs ANTHROPIC_API_KEY; add --no-claude to skip
python src/build_site.py
