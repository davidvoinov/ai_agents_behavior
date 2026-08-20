# Promise Switch Foundation and Execution Layer

Promise Switch is an autonomous package. It has no runtime dependency on Berg.
The frozen scientific foundation is separated from the crash-safe execution
layer. Offline model-specific analysis and codebook validation are available;
they do not perform model calls or replace mechanical decision parsing.

## Package boundaries

```text
promise_switch/
├── __init__.py
├── version.py
├── design.py
├── prompts.py
├── randomization.py
├── schemas.py
├── providers.py
├── client.py
├── parser.py
├── artifacts.py
├── analysis.py
├── codebook.py
├── power.py
└── execution/
    ├── planner.py
    ├── state.py
    ├── storage.py
    ├── scheduler.py
    ├── rate_limit.py
    ├── retry.py
    ├── transport.py
    ├── attrition.py
    └── runner.py
```

`version.py` owns `EXPERIMENT_VERSION`, `PROMPT_VERSION`, and the deterministic
hash of all model-facing templates. `design.py` owns the six frozen researcher
conditions and payoffs. Condition labels are not exposed to evaluated models.

## Confirmatory prompt policy

Primary H8-H10 confirmatory collection uses the single frozen Promise Switch v1
prompt set. No prompt-variant experiment is currently planned, and prompt
variants are not part of the primary or secondary analysis plan. The frozen v1
templates, prompt version, and prompt hash remain unchanged.

Any future prompt-robustness experiment, if one is later justified, would be a
separate post-hoc or follow-up study outside the current Promise Switch
pre-confirmatory scope. Alternative prompt formulations must not be generated
or incorporated into the present collection or analysis plan.

## Voluntary boundary

Voluntary trials begin unassigned. A sees the one-shot task, payoffs, B1, and the
fact that B1 might not remain the final partner. A returns only `{"message": ...}`.
The balanced allocator accepts only recorded `MessageEvent` records and assigns
Switch/No-Switch 1:1 from the seed without using response content. Exactly one
assigned final prompt is then available to a future runner.

Stable internal identifiers retain `voluntary_promise_*` for artifact
compatibility. Scientific/user-facing labels are **Voluntary Message--No-Switch**
and **Voluntary Message--Switch** because message opportunity is not evidence
that the resulting content is a promise.

## Model-facing output

Final decisions must use:

```json
{"decision": "COOPERATE", "justification": "one concise sentence"}
```

The parser accepts only exact `COOPERATE` or `DEFECT` values from `decision`. It
requires the documented string `justification`, rejects duplicate JSON keys,
does not repair actions from justification, and does not classify promises.

## Provider foundation

`promise_switch/providers.py` independently defines provider endpoints, model
aliases, API-key environment-variable names, reasoning capabilities, request
kwargs, and rate-limit placeholders. The current registry distinguishes Nebius
Studio, Nebius UK `uk-south1`, and Groq as separate immutable provider
identities. Nebius identities use `NEBIUS_API_KEY`; Groq uses `GROQ_API_KEY`.
No key value is serializable, and there is no automatic provider fallback.

The current intended pre-GPT/Claude matrix is recorded authoritatively in
`promise_switch_production_readiness.md`. All 14 active rows are executable
offline. DeepSeek V4 Pro and Flash each have explicit ON/OFF configurations;
Groq Llama is excluded from the active matrix because its endpoint is
decommissioned for the available Free/Developer access.
The explicit GLM arms use boolean `enable_thinking` on the Nebius UK endpoint;
the explicit MiniMax arms use `thinking_mode="enabled"` or `"disabled"` on
Nebius Studio. Qwen ON and OFF share a 4,096-token output ceiling; each GLM and
MiniMax pair shares a 2,048-token ceiling. DeepSeek Pro and Flash each use a
provisional shared 2,048-token pair ceiling and serialize
`thinking.type="enabled"/"disabled"`; Nebius compatibility remains pending live
validation after billing restoration.

`ProviderAgnosticClient` builds safe request specifications and requires an
explicit transport. With no transport it cannot make an API call.

## Durable execution

Before any transport call, the runner writes an immutable manifest and the
complete initial pre-API plan. Append-only, fsync-backed JSONL logs then record
every top-up event, appended plan, pause, API attempt, message event,
post-message assignment, terminal trial, and error.
`run_state.json` is replaced atomically after each state transition.

New runs use the `promise_switch_execution_v3` schema and must declare exactly
one dataset role: `technical`, `pilot`, `confirmatory`, or `synthetic`. Resume
rejects legacy runs, changed prompt hashes, changed experiment versions, and
incompatible parameter overrides.
Durable successful responses are reused after a crash instead of being called
again. Behavioral failures remain original missing-action ITT observations and
are never retried; retry is limited to infrastructure failures.

The second durable parser failure for one provider/model/reasoning
configuration pauses the run as `PAUSED_BEHAVIORAL_PARSER_REVIEW`. Per-config
message/final parser-failure counts and the review flag are surfaced to the
operator. This is an operational repeated-event trigger for manual review, not
a significance threshold or automatic incompatibility decision; observations
remain behavioral ITT attrition and are never deleted or replaced.

Exhausting the bounded infrastructure path pauses the batch for manual review
before another trial. Current-schema resume and analysis also validate the
deterministic plan and reject duplicate or cross-linked journals instead of
silently collapsing them.

The default rate policy is deliberately conservative: 30 requests/minute, one
sequential request at a time, and a two-second minimum interval. A provider
`Retry-After` value is a lower bound for exponential backoff. Explicit quota
exhaustion atomically pauses the run as `PAUSED_PROVIDER_QUOTA` for later resume.

Future primary experimental-agent requests use temperature `1.0` for voluntary
messages and final decisions. A different runtime temperature requires explicit
authorization, and the immutable manifest records the effective value and
authorization. Historical temperature-0 probes/pilots remain unchanged and
non-confirmatory. A later temperature-range robustness analysis is secondary,
outside H8-H10 and the six-condition matrix, with exact values still undecided.

## Minimum completed-cell stopping rule

Confirmatory `n=100` means a minimum of 100 mechanically valid completed binary
decisions in every condition for every model configuration. Confirmatory runs
enable outcome-blind sequential top-up automatically; no guessed attrition
percentage is used. A top-up decision can use completion/missingness and
operational pause state, but cannot read COOPERATE/DEFECT direction, message
content, cooperation rates, effects, confidence intervals, or p-values.

Every failed initial or top-up trial remains immutable in the same collection.
Fixed-cell deficits receive new seed-derived trials with new IDs. Voluntary
deficits receive deterministic blocks of two message trials; after both
messages are durable, seeded content-blind allocation assigns exactly one to
No-Switch and one to Switch. If pre-assignment infrastructure attrition prevents
a block from obtaining its required message-complete candidates, a new
seed-derived recovery trial is recorded in that same allocation group. Valid
observations are never discarded. Consequently a voluntary arm may exceed 100
when the other arm first reaches the minimum; analysis uses every eligible
observation and reports actual cell sizes.

`top_up_events.jsonl`, `planned_trials.jsonl`, and `pause_events.jsonl` preserve
the expansion sequence, derivation inputs, initial/top-up identity, block
membership, creation reason, and pause history. Scientific analysis admits a
completed confirmatory run only after integrity validation proves that all six
cells meet the minimum. Audit and inspection tools can still examine incomplete
or paused artifacts.

## Offline dry run

The following exercises planning, prompts, scheduling, serialization,
post-message allocation, and state transitions without network access:

```bash
python3 -B scripts/run_promise_switch.py \
  --run-id local-dry-run \
  --provider nebius \
  --model qwen3_32b \
  --reasoning-mode off \
  --n-per-condition 10 \
  --seed 20260808 \
  --output-dir /private/tmp/promise-switch-dry-runs \
  --dry-run
```

Inspect a run without executing pending work:

```bash
python3 -B scripts/inspect_promise_switch_run.py \
  /private/tmp/promise-switch-dry-runs/local-dry-run
```

## Offline analysis and codebook validation

`scripts/analyze_promise_switch_runs.py` accepts only current-schema execution
runs that are complete, integrity-valid, and explicitly match one requested
`--dataset-role pilot` or `--dataset-role confirmatory`. Technical, synthetic,
legacy pre-role, mixed-role, and invalid runs cannot enter this loader. It
reports primary results independently for
each provider/model/reasoning configuration. The confirmatory family is H8
(Induced No-Switch vs No Communication), H9 (Induced No-Switch vs Induced
Switch), and H10 (Induced No-Switch vs Expectation Without Own Promise), with
Holm correction across those three tests. Voluntary Message--No-Switch versus
Voluntary Message--Switch is a secondary randomized ITT analysis; the
explicit-promiser subset is a further
selected-subgroup secondary analysis.
Primary contrast estimates use mechanically valid completed actions and report
terminal/completed attrition denominators plus worst-case binary-outcome risk-
difference bounds for missing actions. Behavioral failures remain in the
assigned-arm provenance and are not replaced or inferred from ambiguous text.

The output labels these independent analyses `PRIMARY MODEL-SPECIFIC RESULTS`.
After them, `DESCRIPTIVE CROSS-MODEL SUMMARY` presents one row per
provider/model/reasoning configuration, including actual cell sizes, condition
rates, model-specific H8-H10 estimates and intervals, raw p-values, and the
within-configuration Holm values. Direction counts and ranges/medians of the
model-specific risk differences are descriptive only. The summary contains no
pooled observation-level estimate, new significance test, or claim about all
language models.

`codebook.py` validates semantic coding records and derives compliance from the
mechanically parsed final action. It never rewrites that action. Actual human
coding, selection of an independent LLM judge after discussion with David, a
meaningful Cohen-kappa estimate, and the final semantic codebook freeze remain
deferred. The future independent judge will use temperature `0` for coding
reproducibility, without claiming absolute backend determinism; its model is
still undecided and no judging is implemented. The future confirmatory target
is a minimum `n=100` completed observations per condition per model configuration; this
does not authorize collection. No real provider call is part of the offline
test suite.

## Offline power planning

`scripts/power_promise_switch.py` emits a deterministic JSON grid for the
equal-arm binary comparisons underlying H8-H10. It uses the large-sample,
two-sided z test for two independent proportions, with pooled variance under the
null and unpooled variance under the alternative. It does not calculate Fisher
exact power. The default grid covers reference rates 0.10-0.70, absolute MDEs
0.15/0.20/0.25/0.30 in both valid directions, completed n=50/75/100 per arm,
and alpha 0.05 plus 0.05/3. The latter is a conservative Bonferroni planning
sensitivity; it is not identical to and does not replace confirmatory Holm.

The same output reports minimum equal-arm n for 80% and 90% approximate power,
planning-only fixed-percentage attrition illustrations, and exact base calls
for eight- and eleven-configuration scenarios. Production collection instead
uses outcome-blind sequential top-up and does not select an attrition percentage.

```bash
python3 -B scripts/power_promise_switch.py
```

## Minimal confirmatory runbook

The operational one-model-at-a-time procedure, including zero-call durable
initialization and manifest inspection before collection, is documented in
`docs/experiments/promise_switch_sequential_runbook.md`.

1. From a clean frozen revision, choose one validated model alias/mode and a
   unique run ID. Create it with `--dataset-role confirmatory`,
   `--n-per-condition 100`, the frozen seed, and the intended output directory.
   Confirmatory construction automatically enables `minimum_completed_binary_v1`.

   ```bash
   python3 -B scripts/run_promise_switch.py \
     --run-id confirmatory-qwen-off \
     --provider nebius --model qwen3_32b --reasoning-mode off \
     --dataset-role confirmatory --n-per-condition 100 --seed 20260808 \
     --output-dir runs/promise_switch --initialize-only
   ```
2. Inspect the persisted manifest before transport. Review prompt version/hash,
   temperature 1.0, model ID, reasoning mode, cap, initial 600 trials, and base
   800 calls. Retries/top-ups are unknown.
3. Begin collection with `scripts/run_promise_switch.py --resume RUN_ID
   --output-dir OUTPUT_DIR`. The plan is already durable and expansions are
   appended as deterministic events.
4. Inspect without execution using `scripts/inspect_promise_switch_run.py
   RUN_DIR`; audit with `scripts/audit_promise_switch_artifacts.py RUN_DIR`.
5. Resolve quota/infrastructure/parser pauses without changing experimental
   settings, then resume with `scripts/run_promise_switch.py --resume RUN_ID
   --output-dir OUTPUT_DIR`. Paused invocations do not expand automatically.
6. After current trials are terminal, the runner appends only fixed deficits or
   balanced voluntary blocks. Crash/resume remains idempotent during top-up.
7. Collection is complete only when `completion_target_met` is true and all six
   `completed_binary` cell counts are at least 100; voluntary n may be larger.
8. Audit integrity and retain initiated, completed, behavioral, infrastructure,
   retry, pause, and top-up counts for each cell/configuration.
9. Run `scripts/analyze_promise_switch_runs.py --dataset-role confirmatory
   RUN_DIR`; the loader refuses any confirmatory cell below 100.
10. Report actual analyzed n, all attrition counts, and worst-case sensitivity
    bounds. Never trim voluntary overshoot.

## Production-readiness audit

The Stage 6E stop/pause matrix, dataset-isolation rules, `n=100` offline plan
validation, sampling audit, output-token evidence, nomenclature mapping, and
deferred decisions are recorded in
`docs/experiments/promise_switch_production_readiness.md`.
