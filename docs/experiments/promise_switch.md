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

New runs use `runs/promise_switch/<run_id>/` and the
`promise_switch_execution_v1` schema. Resume rejects legacy runs, changed prompt
hashes, changed experiment versions, and incompatible parameter overrides.
Durable successful responses are reused after a crash instead of being called
again. Behavioral failures remain original missing-action ITT observations and
are never retried; retry is limited to infrastructure failures.

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
runs that are not marked invalid. It reports primary results independently for
each provider/model/reasoning configuration. The confirmatory family is H8
(Induced No-Switch vs No Communication), H9 (Induced No-Switch vs Induced
Switch), and H10 (Induced No-Switch vs Expectation Without Own Promise), with
Holm correction across those three tests. Voluntary No-Switch vs Switch is a
secondary randomized ITT analysis; the explicit-promiser subset is a further
selected-subgroup secondary analysis.

`codebook.py` validates semantic coding records and derives compliance from the
mechanically parsed final action. It never rewrites that action. Actual human
and independent-LLM coding, a meaningful Cohen-kappa estimate, executable power
analysis, and confirmatory collection remain deferred. No real provider call is
part of the offline test suite.
