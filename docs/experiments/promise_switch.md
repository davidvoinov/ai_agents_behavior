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
does not repair actions from justification and does not classify promises.

## Provider foundation

`promise_switch/providers.py` independently defines provider endpoints, model
aliases, API-key environment-variable names, reasoning capabilities, request
kwargs, and rate-limit placeholders. The initial offline configuration supports
Nebius Token Factory through `NEBIUS_API_KEY`; no key value is serializable.

`ProviderAgnosticClient` builds safe request specifications and requires an
explicit transport. With no transport it cannot make an API call.

## Durable execution

Before any transport call, the runner writes an immutable manifest and the
complete pre-API plan. Append-only, fsync-backed JSONL logs then record every API
attempt, message event, post-message assignment, terminal trial, and error.
`run_state.json` is replaced atomically after each state transition.

New runs use the `promise_switch_execution_v2` schema and must declare exactly
one dataset role: `technical`, `pilot`, `confirmatory`, or `synthetic`. Resume
rejects legacy runs, changed prompt hashes, changed experiment versions, and
incompatible parameter overrides.
Durable successful responses are reused after a crash instead of being called
again. Behavioral failures remain original missing-action ITT observations and
are never retried; retry is limited to infrastructure failures.

Exhausting the bounded infrastructure path pauses the batch for manual review
before another trial. Current-schema resume and analysis also validate the
deterministic plan and reject duplicate or cross-linked journals instead of
silently collapsing them.

The default rate policy is deliberately conservative: 30 requests/minute, one
sequential request at a time, and a two-second minimum interval. A provider
`Retry-After` value is a lower bound for exponential backoff. Explicit quota
exhaustion atomically pauses the run as `PAUSED_PROVIDER_QUOTA` for later resume.

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

`codebook.py` validates semantic coding records and derives compliance from the
mechanically parsed final action. It never rewrites that action. Actual human
coding, selection of an independent LLM judge after discussion with David, a
meaningful Cohen-kappa estimate, and the final semantic codebook freeze remain
deferred. The future confirmatory target is `n=100` completed observations per
condition per model configuration; this does not authorize collection. No real
provider call is part of the offline test suite.

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
planning-only attrition inflation, and exact base calls for eight- and
eleven-configuration scenarios. It does not select final n or alter the actual
attrition/replacement policy.

```bash
python3 -B scripts/power_promise_switch.py
```

## Production-readiness audit

The Stage 6E stop/pause matrix, dataset-isolation rules, `n=100` offline plan
validation, sampling audit, output-token evidence, nomenclature mapping, and
deferred decisions are recorded in
`docs/experiments/promise_switch_production_readiness.md`.
