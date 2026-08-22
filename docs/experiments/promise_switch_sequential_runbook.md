# Promise Switch sequential confirmatory runbook

Promise Switch configurations may be collected one at a time. Each completed
provider/model/reasoning configuration is an independent confirmatory run and
an independent primary analysis unit. There is no technical or scientific
requirement to launch all 14 configurations together or to wait until all are
available. A descriptive cross-model summary is produced later from multiple
completed run directories; it does not replace any model-specific result.

This runbook does not authorize provider calls. The exact-setting technical
preflight is a future live step and must pass before the corresponding
confirmatory run is initialized.

## One-configuration workflow

1. Verify the repository identity and tracked status:

   ```bash
   git branch --show-current
   git rev-parse HEAD
   git status --short --branch
   git diff --exit-code origin/main -- berg/
   ```

   Use `integrate/main-into-llm-trust-experiments` at the approved frozen HEAD.
   The known untracked `audit_promise_switch/` directory is not run input and
   must not be staged.

2. Select one exact provider, model alias, and reasoning mode from the frozen
   registry. Assign a unique path-safe `<RUN_ID>`. Never reuse a run ID for a
   different configuration.

3. Perform one future exact-setting technical preflight. It is not a pilot and
   makes exactly one request with no retry:

   ```bash
   python3 -B scripts/preflight_promise_switch.py \
     --preflight-id <PREFLIGHT_ID> \
     --provider <PROVIDER> \
     --model <MODEL_ALIAS> \
     --reasoning-mode <REASONING_MODE> \
     --output-dir preflights/promise_switch
   ```

   The command uses the production registry, temperature 1.0, the production
   cap, provider-specific reasoning kwargs, and the frozen final-decision
   prompt. It writes only
   `preflights/promise_switch/<PREFLIGHT_ID>/preflight.json`; this ignored
   technical artifact is not a run and cannot enter scientific analysis.
   Require `result=PASS`, `request_count=1`, normal `finish_reason=stop`, no
   truncation, a valid mechanical action, and the expected reasoning
   verification. `REASONING_STATE_UNVERIFIABLE` and
   `REASONING_STATE_MISMATCH` are failures. Do not repeat a passed preflight
   unless the configuration or a relevant provider behavior changes.

   After a failed preflight, inspect `result`, `failure_category`,
   `http_status`, `provider_failure_category`, `finish_reason`, and
   `reasoning_verification` in the artifact. Do not immediately rerun a failed
   provider preflight. First distinguish authentication, billing/payment,
   permission, rate limit, model/endpoint availability, provider server, and
   transport/connection failures. A provider failure before usable generation
   records reasoning verification as `NOT_EVALUATED`; it is not a reasoning
   failure. This diagnosis-first rule prevents unnecessary paid calls.

4. After a PASS, initialize the confirmatory run without transport:

   ```bash
   python3 -B scripts/run_promise_switch.py \
     --run-id <RUN_ID> \
     --provider <PROVIDER> \
     --model <MODEL_ALIAS> \
     --reasoning-mode <REASONING_MODE> \
     --dataset-role confirmatory \
     --n-per-condition 100 \
     --seed 20260808 \
     --output-dir runs/promise_switch \
     --initialize-only
   ```

   Omit `--reasoning-mode` only when the registry default is the intentionally
   selected frozen mode. Do not pass temperature or token-cap overrides for a
   standard confirmatory run.

5. Before the first provider call, inspect the durable manifest and state:

   ```bash
   python3 -B scripts/inspect_promise_switch_run.py \
     runs/promise_switch/<RUN_ID>
   ```

   Verify `dataset_role=confirmatory`, exact provider/model/reasoning fields,
   temperature `1.0`, production message/final caps, prompt version
   `promise_switch_v1`, frozen prompt hash, target `100`, seed `20260808`, 600
   initial plans, and status `CREATED`.

6. Begin or resume only that initialized run:

   ```bash
   python3 -B scripts/run_promise_switch.py \
     --resume <RUN_ID> \
     --output-dir runs/promise_switch
   ```

   Resume reads the immutable manifest. Do not supply changed scientific or
   technical settings.

7. Inspect status between invocations without executing pending work:

   ```bash
   python3 -B scripts/inspect_promise_switch_run.py \
     runs/promise_switch/<RUN_ID>
   ```

   Respect `PAUSED_PROVIDER_QUOTA`, `PAUSED_INFRASTRUCTURE_REVIEW`, and
   `PAUSED_BEHAVIORAL_PARSER_REVIEW`. Review the recorded cause before resume;
   never change provider, prompt, temperature, cap, or replacement policy.
   For a behavioral parser pause, copy the exact `parser_failure_class` from
   `run_state.json` into the explicit approval:

   ```bash
   python3 -B scripts/run_promise_switch.py \
     --resume <RUN_ID> \
     --output-dir runs/promise_switch \
     --approve-parser-review-class '<PARSER_FAILURE_CLASS>'
   ```

   The class identity is stage plus mechanical response-contract mechanism, not
   semantic failure reason. The approval applies only to that class and model
   configuration. Identical later failures stay missing and are not retried,
   but do not pause; a new class produces a new review pause.
   Under QC v3, malformed JSON is divided into narrow syntactic mechanisms.
   `other_malformed_json` is a conservative residual: its explicit approval is
   occurrence-scoped, so any later residual failure pauses for review again.

8. Never launch a second writer for the same run directory. The per-run OS lock
   rejects concurrent writers. Distinct run IDs have separate manifests,
   journals, state, and lock files and may coexist, although this runbook uses
   deliberate sequential execution.

9. Let outcome-blind top-ups proceed. The run completes only when every one of
   the six cells has at least 100 mechanically valid binary decisions. Retain
   all failures and voluntary overshoot.

10. Audit the completed artifact:

    ```bash
    python3 -B scripts/audit_promise_switch_artifacts.py \
      runs/promise_switch/<RUN_ID>
    ```

    Require a completed target, frozen prompt/config identity, valid links,
    matching prompts, unique IDs, and six qualifying cells.

11. Produce the standalone model-specific primary analysis:

    ```bash
    python3 -B scripts/analyze_promise_switch_runs.py \
      --dataset-role confirmatory \
      runs/promise_switch/<RUN_ID>
    ```

    Archive the run ID, directory, Git commit, provider/model/reasoning identity,
    audit output, and analysis output before moving to the next configuration.

12. After multiple configurations are complete, pass their distinct run
    directories to one analysis invocation:

    ```bash
    python3 -B scripts/analyze_promise_switch_runs.py \
      --dataset-role confirmatory \
      runs/promise_switch/<RUN_ID_1> \
      runs/promise_switch/<RUN_ID_2>
    ```

    `PRIMARY MODEL-SPECIFIC RESULTS` remains the inferential output. The
    `DESCRIPTIVE CROSS-MODEL SUMMARY` displays heterogeneity and contains no
    pooled observation-level effect or pooled significance test.

## Current future-preflight classification

| Configuration | Required future preflight |
|---|---|
| Qwen3-32B ON, Nebius | `ONE_FINAL_PREFLIGHT_RECOMMENDED` |
| Qwen3-32B OFF, Nebius | `ONE_FINAL_PREFLIGHT_RECOMMENDED` |
| GLM-5.2 ON/OFF, Nebius UK | `LIVE_TOGGLE_PREFLIGHT_REQUIRED` |
| MiniMax M3 ON/OFF, Nebius | `LIVE_TOGGLE_PREFLIGHT_REQUIRED` |
| DeepSeek V4 Pro ON/OFF, Nebius | `LIVE_TOGGLE_PREFLIGHT_REQUIRED` |
| DeepSeek V4 Flash ON/OFF, Nebius | `LIVE_TOGGLE_PREFLIGHT_REQUIRED` |
| Kimi K3, Nebius | `ONE_FINAL_PREFLIGHT_RECOMMENDED` |
| Llama 3.3 70B, Nebius | `ONE_FINAL_PREFLIGHT_RECOMMENDED` |
| GPT-OSS-120B, Nebius | `ONE_FINAL_PREFLIGHT_RECOMMENDED` |
| GPT-OSS-120B, Groq | `PROVIDER_ACCESS_PREFLIGHT_REQUIRED` |

Historical successful probes used settings that are not fully equivalent to
the final temperature-1 configuration. No row is currently classified as an
equivalent exact-final preflight.
