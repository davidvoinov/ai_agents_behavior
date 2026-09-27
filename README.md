# Reaction, Recollection, or Script? Investment game

Data and code for the investment game design of *Reaction, Recollection, or Script? Trust and Promise Keeping in Language Model Agents*. The Promise Switch design is on the branch `promise-switch`.

## Layout

| Path | Content |
|---|---|
| `berg/berg_trust_game.py`, `berg/reskin.py` | game engine and the seven surfaces |
| `berg/providers.py`, `berg/anthropic_chat.py` | model configurations and API clients |
| `berg/run_full_cycle.py` | main design, open-weight configurations |
| `berg/batch_runner.py`, `berg/run_batch.sh` | main design, GPT-5.6 and Claude Sonnet 5 through the Batch APIs |
| `berg/run_incentive_control.py` | positive control, enforced minimum return |
| `berg/run_probe.py` | recognition probe |
| `berg/run_judge.py`, `berg/run_kappa.sh`, `berg/make_kappa_sample.py`, `berg/score_kappa.py` | coding of communication transcripts and Cohen's kappa |
| `berg/COMM_CODEBOOK.md` | communication codebook |
| `berg/analysis.py`, `berg/figures.py`, `berg/toggle_check.py`, `berg/lit_reference.py`, `berg/bargaining_proxy.py`, `berg/power_analysis.py` | estimates, figures, reasoning toggle classifier, paradigm reference rate, bargaining proxy, power |
| `berg/dedup_runs.py`, `berg/reclassify_attrition.py`, `berg/purge_truncated.py`, `berg/purge_unusable.py`, `berg/reparse.py`, `berg/redo_truncated_batch.py` | data preparation |
| `berg/data/full/`, `berg/data/batch/` | dyad records, attrition records and removed dyads, fifteen configurations |
| `berg/data/incentive/` | positive control, fourteen arms |
| `berg/data/probe/` | recognition probe, twelve configurations |
| `berg/data/coding/` | coded transcripts and validation samples |
| `berg/data/pilot/` | pilot behind the power calculation |
| `zeckhauser/` | betrayal aversion elicitation after Bohnet and Zeckhauser (2004) |

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

API keys go in `.env` at the repository root: `NEBIUS_API_KEY` for the open-weight configurations, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` for the closed ones.

## Reproduce

From `berg/`, on the stored records, with no API calls:

```
python3 analysis.py
python3 figures.py
python3 toggle_check.py
python3 run_probe.py --score-only
./run_kappa.sh score
python3 -m pytest -q test_reskin.py
```

`figures.py` writes to `paper/figures/`.
