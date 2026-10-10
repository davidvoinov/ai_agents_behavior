# Promise Switch human-validation reliability

This directory contains the minimal de-identified labels needed to reproduce the primary human-versus-designated-judge reliability table for `promise_switch_message_codebook_v1`.

Run from the repository root:

```bash
python3 -B scripts/analyze_promise_switch_validation.py
```

The command validates 130 unique matched public IDs, recalculates raw agreement and Cohen's kappa for the seven published fields, and deterministically rewrites `reliability_summary_n130.csv`.

`human_labels_n130.csv` and `judge_labels_n130.csv` contain only `PSV001`–`PSV130` and the seven categorical labels used in the reliability calculation. Messages and experimental provenance are intentionally excluded. The local mapping to source records is not published.

Cohen's kappa is reported as undefined, not zero, when both coders use one identical category and expected agreement is therefore one. This occurs for `explicit_promise` and `refusal_present`. For `ambiguous_response`, kappa is defined as zero: the human coder marked two messages ambiguous while the judge marked none, despite agreement on 128 of 130 items.

The separate challenge sample and diagnostic per-judge materials are outside the primary reliability result reported here and are not included.
