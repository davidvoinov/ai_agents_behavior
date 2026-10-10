# Promise Switch: publication analysis (`article_15`)

The publication analysis is fully offline and uses only the completed Promise Switch runs already stored with the project. It does not contact model-provider APIs or collect new observations. It was verified with Python 3.12.5 and does not require `.git`, Git history, or remote refs.

Create an isolated environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-publication.txt
```

From the repository root, run:

```bash
python3 -B scripts/analyze_promise_switch_final.py --profile article_15
```

The command fails closed unless it recovers the exact 15-configuration publication profile, in article order, with 9,584 assigned trials, 9,022 valid final decisions, 562 missing outcomes, 90 configuration-by-condition cells, no duplicate trial identities or trial-level rows, and the documented model/provider/reasoning identities. It also verifies that Qwen reasoning ON and GPT-OSS through Groq are absent.

Outputs are written to `notes/promise_switch/final_analysis_article15/`. They include:

- `analysis_manifest.json`: source-run identities, validation results, methods, and file inventory;
- `trial_level_analysis_dataset.csv` and `condition_counts.csv`: the analysis sample and six-condition cooperation rates;
- `confirmatory_h7_h9.csv`/`.json`: model-level H7–H9 risk differences, Newcombe–Wilson intervals, raw p-values, and within-model Holm corrections;
- `voluntary_itt.csv`, `promise_conditioned_status.json`, `undertaking_robustness.csv`, and `missingness_bounds.csv`: secondary and sensitivity analyses;
- `promise_switch_variance_decomposition.csv`: Type III sum-contrast LPM and Shapley logit variance/deviance results;
- `fig_PS_condition_heatmap`, `fig_PS_confirmatory_forest`, and `fig_PS_voluntary_itt` in PDF and PNG formats.

The broader historical profile remains available explicitly as:

```bash
python3 -B scripts/analyze_promise_switch_final.py --profile extended_17
```

That command writes to `notes/promise_switch/final_analysis_extended17/`; it does not overwrite the `article_15` outputs.

Run the publication-specific regression checks with:

```bash
python3 -m unittest tests.test_promise_switch_article15
```

Reproduce the public human-validation statistics with:

```bash
python3 -B scripts/analyze_promise_switch_validation.py
```

Run the complete offline test suite with:

```bash
python3 -m unittest discover -s tests
```
