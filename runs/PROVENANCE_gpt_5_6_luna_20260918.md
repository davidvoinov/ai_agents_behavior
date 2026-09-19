# GPT-5.6 Luna Promise Switch runs — provenance note

Executed by: David Voinov (luk_star2@icloud.com), on request, on a local macOS machine.
Date: 2026-09-18 / 2026-09-19 UTC.
Code state: `f5831fe17c7c7ad95ff1cddde5029f3127a415fe` (as specified).

This note is supplementary. It was written after the runs finished and is NOT part of
the audited artifacts — nothing inside the run directories was modified, so every
recorded hash still verifies.

## Runs

| Run ID | Mode | Cap | Status |
|---|---|---|---|
| `gpt_5_6_luna_off_confirmatory_20260918` | off / no thinking | 256 | COMPLETED |
| `gpt_5_6_luna_on_medium_confirmatory_20260918` | on / medium reasoning | 4096 | COMPLETED |

Preflights (both PASSED before the corresponding run started):

| Preflight ID | Verification |
|---|---|
| `gpt_5_6_luna_off_minimal_preflight_20260918` | VERIFIED_MATCH / VERIFIED_OFF / truncation NONE |
| `gpt_5_6_luna_on_medium_minimal_preflight_20260918` | VERIFIED_MATCH / VERIFIED_ON (20 reasoning tokens) / truncation NONE |

## Provider batches

OFF:  `batch_6aad8eb8d8848190953fed985e44c9a1` (message, 200), `batch_6aadbff3a83881908c221d1dbf2fb4af` (final_decision, 600)
ON:   `batch_6aade4337ab08190b215fb334cb6ed9b` (message, 200), `batch_6aade50e49dc819095134155a326342c` (final_decision, 600)

All four submitted through the OpenAI Batch API. No synchronous API call was made.

## Audit results

Both runs: 800/800 attempts `success`, all attempt number 1 (no retries), `finish_reasons`
`{stop: 800}` (no truncation), 6 conditions x 100 completed, 600/600 prompts matching the
frozen templates, 600/600 decisions and 200/200 messages parsed, 0 errors, 0 pause events,
0 top-up events, `invalid_concurrent_execution: false`, `minimum_completed_target_met: true`.

Artifact integrity re-verified after completion: 8/8 recorded `input_artifact_hash` /
`result_artifact_hash` values match the files on disk.

## Caveat: `working_tree_dirty: true`

Both manifests record a dirty working tree. The cause is pre-existing untracked data
directories under `berg/` that were present before this work began
(`berg/_archive/`, `berg/data/_archive/`, `berg/data/batch/`, `berg/data/full/`,
`berg/data/incentive/`, `berg/data/coding_sample.txt`, `berg/data/smoke_new_models.jsonl`).

No tracked file was modified: `git diff HEAD` over the experiment code was empty at run time.
Prompts, seed (20260808), temperature (1.0), n=100, reasoning settings and the
Switch/No-Switch assignment logic were not touched.

## Environment note

The repository's `python3` on this machine is 3.9.6, which cannot run this code
(`promise_switch/schemas.py` uses 3.10+ union syntax). The runs were executed with a
local Python 3.13 virtualenv (`.venv-ps/`, excluded locally, not committed).

## Headline outcome (not analysed, stated as recorded)

Final actions, off:    COOPERATE 599 / DEFECT 1
Final actions, medium: COOPERATE 278 / DEFECT 322

The off condition sits near the cooperation ceiling; worth a look before analysis.
