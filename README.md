# Pythia Neuron Evolution

How MLP neurons of Pythia-160M change over training. An LLM explains 240 neurons at 40 checkpoints; the explanations are scored, then grouped into concept lineages (branch, replacement, unrelated coexistence, sporadic).

- **Blog post:** https://heartyhaven.github.io/blog/neuron-evolution/
- **Explorer:** https://heartyhaven.github.io/pythia-neuron-explorer/
- **Dataset:** https://huggingface.co/datasets/LangGao/pythia-160m-neuron-evolution

## Setup

```bash
pip install -r requirements.txt        # plus the `zstd` CLI and a CUDA GPU
export OPENAI_API_KEY=...              # explanations, scoring, judges, lineages
export ANTHROPIC_API_KEY=...           # report.py only (audit + findings)
```

Models and settings are in `config.yaml` (`OPENAI_MODEL`, `OPENAI_EFFORT`, `CLAUDE_MODEL`, `STEPS`, ...).

## Reproduce

`./run.sh` runs everything. Each stage resumes if interrupted.

| Step | Command | Output (`out/`) |
|---|---|---|
| Download checkpoints + Pile val | `python src/download.py --all` | `models/`, `data/` |
| Build probe set (10M tokens) | `python src/build_probe.py` | `probe/` |
| Activation summaries | `python src/extract.py` | `acts/` |
| Sample, contexts, metrics | `python src/prep.py all` | `sample.json`, `contexts/`, `metrics/` |
| Explain | `python src/agents.py explain` | `explanations.jsonl` |
| Detection score | `python src/agents.py detect` | `detections.jsonl` |
| Trajectory judges | `python src/agents.py judge` | `judgments.jsonl` |
| Lineages | `python src/lineage.py` | `lineages.jsonl` |
| Semantic map | `python src/semantic_map.py` | `semantic_map.json` |
| Report | `python src/report.py` | `report.md`, `figs/` |
| Explorer site | `python src/build_site.py --serve` | `site/` |

`download.py` without `--all` fetches only the 40 analysis steps (~15 GB). That is enough for the explanations, but the metric curves then cover only those steps.

Cost: about 20k OpenAI calls (9,600 explain + 9,600 detect + 480 judge + 240 lineage).

## Notes

- Use `--limit N` on `agents.py` and `lineage.py` for a trial run.
- The released dataset was produced with Chinese versions of these prompts, then machine-translated. The prompts here are English, so a rerun gives English text directly, worded differently.
- In the release, detection scores and judges cover 20 of the 40 steps.

## References

- Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, 2023.
- Gao et al., *The Pile*, 2020 ([monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)).
