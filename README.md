# Promise Switch

This branch contains the data, code, and publication outputs for the Promise
Switch experiment. The companion Investment Game experiment is maintained on
the `investment-game` branch.

## Repository layout

- `promise_switch/`: experimental design, execution, validation, and analysis code;
- `runs/promise_switch/`: completed Promise Switch run artifacts;
- `notes/promise_switch/semantic_coding/public_semantic_input_v1/`: portable semantic labels joined to raw trials by `source_run` and `trial_id`;
- `notes/promise_switch/final_analysis_article15/`: article tables, reports, and figures;
- `notes/promise_switch/validation/promise_switch_message_codebook_v1/public/`: public human-validation labels and reliability summary;
- `scripts/`: offline reproduction entry points;
- `tests/`: offline unit and publication checks.

## Environment

The publication workflow was verified with Python 3.12.5. From the repository
root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-publication.txt
```

The constraints file records the directly used package versions in the verified
publication environment.

## Reproduction

The commands are offline and do not call model-provider APIs. Git metadata and
remote refs are not required.

```bash
python3 -B scripts/analyze_promise_switch_final.py --profile article_15
python3 -B scripts/analyze_promise_switch_validation.py
python3 -m unittest discover -s tests
```

The article analysis writes to
`notes/promise_switch/final_analysis_article15/`. Human-validation output is
written to
`notes/promise_switch/validation/promise_switch_message_codebook_v1/public/`.

`article_15` is the 15-configuration publication profile. The optional
`extended_17` profile adds the excluded Qwen reasoning-ON and GPT-OSS/Groq
configurations and writes to
`notes/promise_switch/final_analysis_extended17/`; it does not overwrite the
article outputs.

Detailed commands and output definitions are in
`docs/experiments/promise_switch_article15_reproduction.md`.
