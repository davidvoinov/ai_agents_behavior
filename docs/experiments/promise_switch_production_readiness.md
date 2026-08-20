# Promise Switch Production-Readiness Safeguards

This document records the zero-call Stage 6E audit. It does not authorize a
confirmatory run. The future target is a minimum 100 completed binary
observations per condition per model configuration. Attrition uses outcome-blind
sequential top-up, not a guessed fixed inflation rate.

## Collection stop and pause criteria

The operational categories are:

- **Catastrophic run-integrity failure:** stop the entire batch before another
  provider request. Do not resume until the artifact is reviewed; invalidate the
  batch when the frozen treatment or assignment history cannot be established.
- **Pause and investigate:** preserve durable state and resume only after a
  documented technical review. A pause does not silently change provider,
  model, prompt, sampling, token limits, or replacement policy.
- **Retryable infrastructure failure:** apply the immutable bounded retry policy
  (at most two retries after the first attempt). If it is exhausted, the trial
  becomes terminal infrastructure attrition and the batch pauses for review.
- **Terminal behavioral failure:** retain the original ITT observation as a
  terminal behavioral failure. Do not retry or replace it automatically.

| Event | Required handling | Implemented safeguard |
|---|---|---|
| Prompt hash mismatch | Catastrophic stop | Resume and every fresh request compare the immutable manifest with the current template hash. |
| Experiment or prompt version mismatch | Catastrophic stop | Current-schema resume and artifact validation reject mismatches. |
| Provider/model/reasoning/config mismatch | Catastrophic stop | Immutable normalized parameters, deterministic plan reconstruction, and per-request config checks. |
| Unexpected condition imbalance | Catastrophic stop | Integrity validation reconstructs and exactly compares the seeded six-condition plan; voluntary allocation requires exact 1:1 balance. |
| Duplicate trial or journal IDs | Catastrophic stop | Integrity validation rejects duplicate plan, message, assignment, attempt, and terminal-trial identities before resume or analysis. |
| Duplicate execution / multiple writers | Catastrophic stop | One OS-level writer lock covers the complete run lifecycle; invalid concurrent artifacts remain excluded. |
| Voluntary assignment before durable message save | Catastrophic stop | State transitions and integrity validation require a durable, matching `MessageEvent`. |
| Both voluntary branches for one trial | Catastrophic stop | One assignment per trial is enforced; duplicate assignment rows are rejected rather than collapsed. |
| Isolated parser-invalid model output | Terminal behavioral failure | No retry; preserved as behavioral attrition under ITT. |
| A repeated parser-failure pattern | Pause and investigate | The second durable parser failure for one provider/model/reasoning configuration triggers `PAUSED_BEHAVIORAL_PARSER_REVIEW`, exposes per-configuration counts, and requires manual response-contract review before resume. |
| Malformed provider envelope | Retryable infrastructure failure | Bounded retry; exhaustion records infrastructure attrition and pauses the batch. |
| `finish_reason=length` / truncation | Retryable infrastructure failure | Bounded retry at the immutable limit; exhaustion pauses. The limit cannot change on resume. |
| Quota exhaustion | Pause and investigate | Automatic `PAUSED_PROVIDER_QUOTA`; provider never changes automatically. |
| Provider/model unavailable or authentication rejected | Pause and investigate | Non-retryable infrastructure failure becomes terminal for the affected trial and pauses the batch before another trial. |
| Repeated timeout/connection/provider failures | Retry, then pause | Bounded retry; exhaustion pauses before further collection. |
| Corrupted/incomplete state or journal cross-links | Catastrophic stop | Current-schema integrity validation rejects unknown IDs, broken message links, plan divergence, and run-state mismatch. |
| Resume/idempotence invariant failure | Catastrophic stop | Durable successful responses are reused; duplicate rows and incompatible resume parameters are rejected. |
| Researcher condition-label leakage | Catastrophic stop | User-facing and internal condition labels are checked before transport and in stored attempt prompts. |
| Frozen prompt modified during a live run | Catastrophic stop | Prompt hash is recomputed immediately before each fresh transport request. |

The second occurrence is an operational definition of “repeated,” not a
statistical-significance threshold or an automatic declaration that a model is
incompatible. Every parser-invalid response remains behavioral ITT attrition:
it is retained, not retried, not replaced, and never reclassified as
infrastructure. The review rule does not inspect whether any valid action was
COOPERATE or DEFECT and never changes the frozen prompt automatically.

## Dataset isolation

Execution schema `promise_switch_execution_v3` requires every new run manifest
to identify exactly one `dataset_role`: `technical`, `pilot`, `confirmatory`, or
`synthetic`.

- `synthetic` is allowed only with `dry_run=true`, and every dry run must be
  synthetic.
- `confirmatory` is allowed only at the approved minimum
  `n_per_condition=100`, automatically enables top-up, and may not be a dry run.
- The scientific analysis CLI requires an explicit `--dataset-role pilot` or
  `--dataset-role confirmatory`; every input must match that one role and be a
  complete, integrity-valid current-schema run.
- Technical and synthetic runs cannot enter the scientific analysis CLI.
- Pre-role `promise_switch_execution_v1` artifacts, including existing pilots
  and smokes, are legacy development evidence. They cannot enter the v2
  scientific analysis loader and may be inspected only with the audit CLI's
  explicit `--include-legacy` switch.
- Invalid concurrent runs require the separate audit-only `--include-invalid`
  switch and never enter scientific analysis.
- Standalone probes lack a complete execution manifest/trial journal and are
  rejected. Legacy `promise_switch_high_limit` data are also rejected by schema.

Existing pilots are not relabeled, migrated, pooled, or reused as confirmatory
observations.

## Offline `n=100` planning validation

With one generic model configuration and a fixed seed, the current deterministic
planner produces exactly 600 final-trial plans: 100 in each of four fixed arms
and 200 initially unassigned voluntary-message trials. After all 200 message
events are durably represented, the content-blind allocator assigns exactly 100
to each voluntary final branch. Replanning under the same seed is identical;
changing message content does not change assignments. Every trial has one final
assignment, and the scheduler/state representation safely holds all 600 IDs.

This validation is an in-memory/test-fixture exercise only. It creates no
production run directory and performs no provider call.

## Completed-cell top-up and analysis admission

The immutable manifest freezes `minimum_completed_binary_v1`. Once all current
work is terminal, completion-status-only logic appends exact fixed-cell deficits
and balanced two-trial voluntary blocks. Original failures remain immutable;
new trials have new deterministic IDs and explicit top-up provenance.
COOPERATE/DEFECT values, message content, effect estimates, confidence
intervals, and p-values are absent from expansion inputs.

Voluntary allocation remains after durable message storage, seeded,
content-blind, and exactly one No-Switch plus one Switch per block. Differential
attrition may produce voluntary overshoot, and all valid observations remain
eligible. A pre-assignment infrastructure failure is retained and receives a
new deterministic message-recovery candidate rather than retrying or
overwriting the failed trial. Pause states prevent expansion until safe resume.

Final confirmatory analysis requires at least 100 mechanically valid binary
actions in all six cells for every configuration. Audit/inspection of paused or
incomplete artifacts remains available. Reports preserve initiated, completed,
behavioral missing, terminal infrastructure, retry, pause, and top-up counts
plus worst-case attrition bounds.

## Sampling and reasoning policy

Future primary/confirmatory experimental-agent generation uses temperature
`1.0` for both voluntary messages and final decisions. The foundation config,
runner, and CLI share this default. A different runtime value requires
`--authorize-temperature-override`; the immutable manifest records the
effective temperature and authorization state. No `top_p`, `top_k`, or
equivalent field is set.

Historical probes and pilots retain their actual temperature `0.0` metadata and
remain non-confirmatory development evidence; they are not rerun, rewritten, or
relabeled. Temperature `1.0` is intended to sample behavioral variation across
repeated trials rather than use near-deterministic decoding.

| Alias | Exact model ID | Provider | Reasoning interpretation | Explicit parameter | Requested temp / cap | Validation status | Outstanding validation | Scientific role |
|---|---|---|---|---|---|---|---|---|
| `qwen3_32b` ON | `Qwen/Qwen3-32B` | Nebius Studio | thinking on | `chat_template_kwargs.enable_thinking=true` | 1.0 / 4,096 | Historically validated | None for toggle mechanics | Clean Qwen ON/OFF pair |
| `qwen3_32b` OFF | `Qwen/Qwen3-32B` | Nebius Studio | thinking off | `chat_template_kwargs.enable_thinking=false` | 1.0 / 4,096 | Historically validated | None for toggle mechanics | Clean Qwen ON/OFF pair |
| `glm_5_2_nebius_thinking_on` | `zai-org/GLM-5.2` | Nebius UK `uk-south1` | thinking on | `chat_template_kwargs.enable_thinking=true` | 1.0 / 2,048 | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | One successful ON probe | Clean GLM ON/OFF pair |
| `glm_5_2_nebius_thinking_off` | `zai-org/GLM-5.2` | Nebius UK `uk-south1` | thinking off | `chat_template_kwargs.enable_thinking=false` | 1.0 / 2,048 | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | One successful OFF probe | Clean GLM ON/OFF pair |
| `minimax_m3_nebius_thinking_on` | `MiniMaxAI/MiniMax-M3` | Nebius Studio | thinking on | `chat_template_kwargs.thinking_mode="enabled"` | 1.0 / 2,048 | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | One successful ON probe | Clean MiniMax ON/OFF pair |
| `minimax_m3_nebius_thinking_off` | `MiniMaxAI/MiniMax-M3` | Nebius Studio | thinking off | `chat_template_kwargs.thinking_mode="disabled"` | 1.0 / 2,048 | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | One successful OFF probe | Clean MiniMax ON/OFF pair |
| `deepseek_v4_pro_nebius_thinking_on` | `deepseek-ai/DeepSeek-V4-Pro` | Nebius Studio | thinking on | `thinking.type="enabled"` | 1.0 requested / 2,048 provisional | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | Verify Nebius preserves toggle and normal finish | Clean V4 Pro ON/OFF pair |
| `deepseek_v4_pro_nebius_thinking_off` | `deepseek-ai/DeepSeek-V4-Pro` | Nebius Studio | thinking off | `thinking.type="disabled"` | 1.0 requested / 2,048 provisional | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | Verify absent reasoning and normal finish | Clean V4 Pro ON/OFF pair |
| `deepseek_v4_flash_nebius_thinking_on` | `deepseek-ai/DeepSeek-V4-Flash` | Nebius Studio | thinking on | `thinking.type="enabled"` | 1.0 requested / 2,048 provisional | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | Verify Nebius preserves toggle and normal finish | Clean V4 Flash ON/OFF pair |
| `deepseek_v4_flash_nebius_thinking_off` | `deepseek-ai/DeepSeek-V4-Flash` | Nebius Studio | thinking off | `thinking.type="disabled"` | 1.0 requested / 2,048 provisional | `PENDING_NEBIUS_LIVE_TOGGLE_VALIDATION` | Verify absent reasoning and normal finish | Clean V4 Flash ON/OFF pair |
| `kimi_k3_nebius` | `moonshotai/Kimi-K3` | Nebius Studio | current fixed/default configuration | none | 1.0 / 2,048 | Historically validated | None currently | Single model-family configuration |
| `llama_3_3_70b` | `meta-llama/Llama-3.3-70B-Instruct` | Nebius Studio | fixed off | none | 1.0 / 256 | Historically validated | None currently | Single model-family configuration |
| `gpt_oss_120b_nebius` | `openai/gpt-oss-120b` | Nebius Studio | fixed on, low effort | `reasoning_effort=low` | 1.0 / 2,048 | Historically validated | None currently | Same-model/two-provider control |
| `gpt_oss_120b_groq` | `openai/gpt-oss-120b` | Groq | fixed on, low effort | `reasoning_effort=low` | 1.0 / 2,048 | `GROQ_PERMISSION_STILL_BLOCKED`: exact metadata HTTP 403 | Resolve console permissions, then one generation probe | Same-model/two-provider control |

Provider identity is immutable and no automatic fallback exists. Same-model
provider comparisons are robustness controls, not claims of identical serving,
quantization, chat templates, or bitwise outputs.

Before a configuration's first confirmatory run, its exact-final technical
preflight uses `scripts/preflight_promise_switch.py`. The command makes exactly
one request, never retries, and records safe configuration/finish/parser/
reasoning evidence under the non-scientific `technical_preflight` role. PASS is
required before `--initialize-only`; unverifiable or mismatched reasoning state
is a preflight failure rather than inferred compatibility. The additive v1
artifact fields `http_status`, `provider_failure_category`, and whitelisted
`safe_error_metadata` preserve safe failure provenance from that single call.
Do not immediately rerun a failed provider preflight: diagnose authentication,
billing/payment, permission, rate limit, model/endpoint availability, provider
server, or transport failure first. When no usable generation exists,
reasoning verification is `NOT_EVALUATED` rather than the primary failure.

`LLAMA_GROQ_SKIPPED_PROVIDER_DEPRECATION`: Groq Llama is not an active matrix
row because the Llama 3.3 70B endpoint is decommissioned for the available
Free/Developer access. Historical consideration remains documented; no
replacement model or provider-control substitute is introduced.

For DeepSeek thinking-mode requests, the experiment continues to request and
manifest temperature `1.0`, but upstream behavior may ignore it or be model-controlled by
sampling parameters. This caveat does not alter the global temperature policy.
Nebius-visible effective behavior must be reported transparently after live
toggle validation. V4 Pro versus V4 Flash is model-variant heterogeneity, not a
clean reasoning contrast; the clean contrasts are ON versus OFF within Pro and
within Flash.

A temperature-range robustness analysis is intended as future secondary work,
not part of primary H8-H10 inference or the current six-condition matrix. Its
exact temperatures remain undecided; no robustness cells or runs exist.

## Output-token evidence

Stage 6F freezes the current production maximum output budget per validated
model configuration. These are non-binding caps, not requested generation
lengths: a model that completes in 50 tokens should stop normally at about 50
tokens. The scientific purpose is to prevent differential truncation from
confounding treatment outcomes.

| Configuration | Production cap | Observed completion use | Rationale |
|---|---|---|---|
| Qwen off | 4,096 | pilot max 37; no length finishes | Raised from 256 to match Qwen ON and remove ceiling asymmetry. |
| Qwen on | 4,096 | pilot max 1,402; prior truncation at 128 and 256, plus one invalid-run truncation at 2,560 | 4,096 is the validated non-binding setting. |
| Llama | 256 | pilot max 49; no length finishes | 256 was ample across the pilot. |
| GPT-OSS-120B | 2,048 | 51 in one technical probe; stop | Evidence is insufficient to lower the successful probe cap. |
| DeepSeek V4 Pro ON/OFF | 2,048 shared, provisional | historical default-mode probe used 33 and stopped | Pair parity is frozen; tomorrow's ON probe determines whether a higher shared cap is necessary. |
| DeepSeek V4 Flash ON/OFF | 2,048 shared, provisional | no generation evidence | Pair parity is frozen; tomorrow's ON probe determines whether a higher shared cap is necessary. |
| Kimi K3 | 2,048 | 361 in one technical probe; stop | 256 is known to be too low. |
| MiniMax M3 ON/OFF | 2,048 shared | historical default-mode probe used 495 and stopped; new explicit calls returned HTTP 402 before generation | Pair parity is frozen; live ON/OFF sufficiency remains unverified. |
| GLM 5.2 ON/OFF | 2,048 shared | historical default-mode probe used 540 and stopped; new explicit calls returned HTTP 402 before generation | Pair parity is frozen; live ON/OFF sufficiency remains unverified. |
| GPT-OSS-120B via Groq | 2,048 | no completion; one probe returned HTTP 403 | Matches the Nebius GPT-OSS ceiling if access becomes available. |

The model configuration is authoritative when CLI cap arguments are omitted.
A differing `--max-tokens` or `--message-max-tokens` value requires the explicit
`--authorize-output-cap-override` flag. The immutable manifest records both the
effective message/final caps and whether an override was authorized. A run may
contain multiple configurations only when their validated caps match, because
the current runner has one immutable pair of caps per run.

`finish_reason=length` remains retryable infrastructure/token-truncation failure
under the existing bounded retry policy; it is never a normal behavioral
observation. GPT and Claude caps remain deferred until their own technical
probes. This output-cap decision does not resolve temperature or sampling.

## Scientific nomenclature and consistency

Stable internal identifiers `voluntary_promise_no_switch` and
`voluntary_promise_switch` are retained for artifact compatibility. Scientific
and user-facing output maps them to **Voluntary Message--No-Switch** and
**Voluntary Message--Switch**. Treatment assignment records an opportunity to
message, not evidence that a promise was made; actual promise status requires
semantic coding.

The repository remains consistent on six conditions; payoffs `(8,8)` versus
`(10,0)`; exact induced promise and expectation wording; B1/B2 switch logic;
durable voluntary message before assignment; exact decision JSON; H8, induced
H9, and H10 with Holm correction; voluntary ITT as secondary; the explicit-
promiser subset as further selected-subgroup secondary; model-configuration-
specific primary inference; and the future minimum completed-cell target `n=100`.

## Deferred work

Independent configurations may be collected sequentially. Each uses a unique
run ID, directory, manifest, state, journals, and writer lock, and remains a
standalone model-specific primary analysis unit. Multiple completed run
directories are combined only for the later descriptive cross-model display.
The exact terminal workflow is in `promise_switch_sequential_runbook.md`.

- Codebook software mechanics are validated. Human coding, selection of an
  independent LLM judge after discussion with David, actual Cohen's kappa, and
  the final semantic codebook freeze remain pending. No new Qwen data are needed
  solely for kappa. The future independent judge will use temperature `0` for
  coding reproducibility, but the judge model remains undecided; temperature
  zero is not claimed to guarantee absolute backend determinism.
- Proprietary GPT and Claude remain intended, expensive, and deferred.
- Groq is the approved GPT-OSS second-provider identity. Groq Llama is skipped
  for provider deprecation and is not an active matrix row. No provider fallback
  or substitute model is allowed.
- The current exact Groq GPT-OSS metadata recheck returned HTTP 403 with no safe
  provider error code/type or scope-specific message category. No generation
  call followed. Project permission may be saved while organization-level
  permission still supersedes it; this requires user-side console inspection,
  not random-model probing.
- Nebius live validation is deferred until billing restoration. The exact
  required probes are GLM ON/OFF, MiniMax ON/OFF, DeepSeek V4 Pro ON/OFF, and
  DeepSeek V4 Flash ON/OFF: one successful probe per arm, no pilot. Each pair
  must show reasoning present for ON, absent for OFF, valid final JSON, normal
  finish, no truncation, and one shared cap. Only a length-truncated ON arm may
  receive one targeted higher-cap retry, after which that cap applies to both
  arms.
- The exact secondary temperature-robustness grid remains unresolved. Output
  caps are specified for all 14 active current-matrix configurations; DeepSeek
  pair caps remain provisional until live validation, while future proprietary
  GPT/Claude caps remain unresolved.
