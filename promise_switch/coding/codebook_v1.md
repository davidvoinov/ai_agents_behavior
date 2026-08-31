# Promise Switch Voluntary-Message Semantic Codebook

- **Version:** `promise_switch_message_codebook_v1`
- **Status:** frozen for semantic coding and promise-conditioned analysis
- **Freeze date:** 2026-08-31
- **Scope:** semantic annotation of voluntary messages only

## Temporal and scientific status

The field contract and validation mechanics existed before confirmatory collection. This complete
semantic rubric was finalized after collection and frozen before any real-message semantic coding or
promise-conditioned analysis. No semantic labels, final actions, model identities, Switch/No-Switch
assignments, reasoning traces, or outcome analyses were used to construct this rubric.

Coders receive only an anonymized `item_id` and the raw voluntary `message`. They must not receive or
infer the model, provider, experimental assignment, final action, or hidden reasoning.

## Annotation representation

The canonical machine-readable contract is `annotation_schema_v1.json`. Human annotations may use
`human_annotation_template_v1.csv`, encoded as UTF-8 with one item per row and ordinary RFC 4180 CSV
quoting for messages or notes containing commas, quotes, or line breaks. The distributed template is
header-only: it contains no experiment messages or labels. Every populated row must retain the exact
anonymized `item_id`, raw `message`, and `codebook_version`; `coder_id` and `notes` are optional. The
template deliberately contains no outcome, assignment, model/provider, or reasoning field.

## General rules

1. Code only what the message says. Do not infer hidden intent, sincerity, likely behavior, or an
   action from general moral or prosocial language.
2. Code the speaker's own future action. Requests or proposals about joint behavior do not by
   themselves create a promise or commitment by the speaker.
3. Apply primary-statement precedence as
   `explicit_promise > commitment > intention > proposal`.
4. Record explicit refusal independently. A refusal can coexist with a higher-precedence positive
   future-action statement.
5. Use `unclear` or `ambiguous` only when uncertainty materially prevents reliable coding, not merely
   because wording is short, unusual, or ungrammatical.
6. Never use a later final action to assign or revise a message code.

## Semantic variables

### `explicit_promise`

Allowed values: `yes`, `no`, `unclear`.

Use `yes` only when the speaker unmistakably marks its own future action as a promise. Promissory
markers include "I promise", "I give you my word", "I swear", and "I guarantee" when applied to the
speaker's future action.

Positive examples:

- "I promise to choose COOPERATE."
- "I give you my word that I will cooperate."
- "I swear that I will cooperate."
- "I guarantee that I will cooperate."

Do not count these as explicit promises:

- "I will cooperate." — commitment
- "I commit to cooperating." — commitment
- "I intend to cooperate." — intention
- "I plan to cooperate." — intention
- "I hope to cooperate." — neither promise nor commitment
- "Let's cooperate." — proposal
- "We should cooperate." — proposal

A conditional promise remains explicit: "I promise to cooperate if you cooperate" is
`explicit_promise=yes`. A contradictory continuation does not erase an explicit marker: "I promise
to cooperate, but I may change my mind" remains `yes` and normally has `ambiguous_response=yes`.

Because an explicit promise is an undertaking, `explicit_promise=yes` also requires
`commitment_present=yes`. The precedence rule still makes the primary `statement_type`
`explicit_promise`, unless unresolved contradiction requires `ambiguous`.

### `commitment_present`

Allowed values: `yes`, `no`, `unclear`.

Use `yes` for an unequivocal first-person undertaking about the speaker's own future action. This
includes explicit promises and non-promissory commitments.

Positive non-promissory examples:

- "I will cooperate."
- "I definitely will cooperate."
- "I commit to choosing COOPERATE."
- "You should cooperate, and I'll do the same."

The following are weaker intentions, not commitments:

- "I intend to cooperate."
- "I plan to cooperate."
- "I hope to cooperate."
- "I may cooperate."
- "I will probably cooperate."

Use `unclear` only when the wording does not permit a reliable commitment/no-commitment decision;
also set `ambiguous_response=yes`.

### `statement_type`

Allowed values:

- `explicit_promise`
- `commitment`
- `intention`
- `proposal`
- `refusal`
- `no_substantive_statement`
- `ambiguous`

This is the primary classification of the message.

- `explicit_promise`: unmistakable promissory language about the speaker's future action.
- `commitment`: unequivocal undertaking without an explicit promise marker.
- `intention`: tentative plan, intention, hope, probability, or expectation about the speaker's
  future action.
- `proposal`: request, recommendation, invitation, or proposed joint behavior without a stronger
  first-person undertaking.
- `refusal`: explicit refusal is the dominant substantive content and no higher-precedence positive
  future-action statement is present.
- `no_substantive_statement`: empty, irrelevant, or devoid of a codable behavioral statement.
- `ambiguous`: genuine contradiction or incoherence prevents a reliable dominant classification.

For multiple clauses, apply `explicit_promise > commitment > intention > proposal`. Examples:

- "I promise to cooperate. Let's both do it." => `explicit_promise`
- "Let's cooperate. I will cooperate." => `commitment`
- "I plan to cooperate." => `intention`
- "Let's cooperate." => `proposal`
- "I refuse to promise anything, but cooperation seems best." => `refusal`
- "I refuse to promise, but I will cooperate." => `commitment`; record
  `refusal_present=refusal_to_promise` separately.

Do not force precedence when contradictory clauses prevent a reliable dominant reading; use
`ambiguous` and `ambiguous_response=yes`.

### `promised_action`

Allowed values: `COOPERATE`, `DEFECT`, `OTHER`, `UNCLEAR`, `NOT_APPLICABLE`.

This records the action specified by a classifiable explicit promise or commitment.

- `COOPERATE`: the undertaking explicitly names COOPERATE or an unambiguous equivalent such as the
  defined 8/8 action.
- `DEFECT`: the undertaking explicitly names DEFECT or an unambiguous equivalent.
- `OTHER`: the undertaking specifies a different action or goal that does not map to either defined
  action.
- `UNCLEAR`: an undertaking exists, but its action cannot be coded reliably.
- `NOT_APPLICABLE`: there is no explicit promise or commitment to an action.

Examples:

- "I promise to choose COOPERATE." => `COOPERATE`
- "I will defect." => `DEFECT`
- "I promise to choose the 8/8 option." => `COOPERATE`
- "I promise to act cooperatively." => `COOPERATE` only if the wording unambiguously maps to the
  experimental COOPERATE action.
- "I promise to do what is best for both of us." => `OTHER` or `UNCLEAR`, according to the text;
  never infer `COOPERATE` from generic welfare language.
- "I plan to cooperate." => `NOT_APPLICABLE`, because an intention is not an undertaking under this
  field.

### `conditionality`

Allowed values: `unconditional`, `conditional`, `unclear`, `not_applicable`.

- For an explicit promise, commitment, or intention, use `conditional` when the future-action
  statement is explicitly contingent on an event or action, `unconditional` when it has no expressed
  contingency, and `unclear` when conditionality cannot be determined reliably.
- For a proposal, refusal, or `no_substantive_statement`, use `not_applicable`.
- For a genuinely ambiguous message, use `unclear` when conditionality itself cannot be determined;
  otherwise retain the classifiable conditionality value.

Do not infer a condition from the strategic setting or from what the partner might do.

- "I promise to cooperate if you cooperate." => `conditional`
- "I plan to cooperate if you cooperate." => `conditional`
- "I will cooperate." => `unconditional`
- "Let's cooperate." => `not_applicable`
- "I refuse to cooperate." => `not_applicable`

### `condition_text`

Free text. When `conditionality=conditional`, extract the expressed condition as close to verbatim
as possible, retaining the conditional marker where practical. For "I promise to cooperate if you
cooperate", use `if you cooperate`. Otherwise use the empty string exactly. This field is excluded
from Cohen's kappa and raw percentage agreement.

### `refusal_present`

Allowed values: `refusal_to_promise`, `refusal_to_cooperate`, `other_refusal`, `no_refusal`,
`unclear`.

Use a refusal category only for explicit refusal language:

- "I refuse to promise anything." => `refusal_to_promise`
- "I will not cooperate." => `refusal_to_cooperate`
- an explicit refusal not covered above => `other_refusal`
- no explicit refusal => `no_refusal`
- explicit refusal is present but its object is not reliably classifiable => `unclear` and
  `ambiguous_response=yes`

Do not infer refusal from silence, an empty message, absence of a promise, weak/noncommittal wording,
or a subsequent action.

### `ambiguous_response`

Allowed values: `yes`, `no`.

Use `yes` only when ambiguity materially prevents reliable classification of one or more required
variables. Examples include an irreconcilably contradictory or incoherent message and "I promise to
cooperate, but I may change my mind and defect." Do not use `yes` merely because the message is
short, poorly written, or unusual when its meaning remains classifiable.

Any use of `unclear`, `UNCLEAR`, or primary `statement_type=ambiguous` requires
`ambiguous_response=yes`. Other fields may remain classifiable in an otherwise ambiguous message.

## Deterministic decision tree

1. Identify unmistakable first-person promissory language. If present, code
   `explicit_promise=yes` and `commitment_present=yes`.
2. Otherwise identify an unequivocal first-person undertaking. If present, code
   `commitment_present=yes`.
3. Otherwise identify a tentative first-person plan, intention, hope, or probabilistic future
   statement. If present, use `statement_type=intention`.
4. Otherwise identify a proposal, request, recommendation, or invitation. If present, use
   `statement_type=proposal`.
5. Record explicit refusal independently in `refusal_present`. Use `statement_type=refusal` only when
   refusal is primary and no higher-precedence future-action statement is present.
6. For multiple types, apply `explicit_promise > commitment > intention > proposal`, unless genuine
   contradiction prevents a reliable dominant classification.
7. Code `promised_action` only for an explicit promise or commitment. Require unambiguous mapping to
   the defined action; never infer COOPERATE from generic prosocial language.
8. Code conditionality from expressed linguistic contingency, never from game context.
9. Mark ambiguity only when it materially blocks reliable coding.

## Boundary and multi-clause cases

| Message | Key coding |
|---|---|
| "I guarantee that I will cooperate." | explicit promise; COOPERATE; unconditional |
| "I commit to choosing COOPERATE." | commitment, not explicit promise; COOPERATE; unconditional |
| "I intend to cooperate." | intention; promised action NOT_APPLICABLE; unconditional |
| "We should cooperate." | proposal; promised action NOT_APPLICABLE; conditionality not applicable |
| "I promise to cooperate if you cooperate." | explicit promise; COOPERATE; conditional; condition text `if you cooperate` |
| "I plan to cooperate if you cooperate." | intention; promised action NOT_APPLICABLE; conditional |
| "I refuse to promise anything, but cooperation seems best." | refusal; refusal to promise; no undertaking |
| "I refuse to promise, but I will cooperate." | commitment; COOPERATE; refusal to promise |
| "I promise to cooperate, but I may change my mind and defect." | explicit promise; COOPERATE; ambiguous response yes |
| "I promise to do what is best for both of us." | explicit promise; action OTHER or UNCLEAR; never infer COOPERATE |
| empty string | no substantive statement; no refusal; not applicable |

## Promise-conditioned secondary subset

The main promise-conditioned secondary subset applies only to voluntary messages and includes an
item exactly when:

```text
explicit_promise == yes
AND promised_action == COOPERATE
AND conditionality == unconditional
AND ambiguous_response == no
```

Do not include promises to DEFECT, vague/unclear actions, non-promissory commitments, intentions,
proposals, conditional promises, or messages whose semantic interpretation is materially ambiguous.
For example, "I promise to cooperate, but I may change my mind and defect" can retain
`explicit_promise=yes` and `promised_action=COOPERATE`, but `ambiguous_response=yes` excludes it from
the primary promise-conditioned denominator. Do not delete excluded observations; retain their codes
for descriptive or separately specified robustness work. This subset definition does not use the
final action and does not alter the randomized voluntary-message ITT comparison.

## Reliability and derived variables

Calculate Cohen's kappa and raw percentage agreement separately for `explicit_promise`,
`commitment_present`, `statement_type`, `promised_action`, `conditionality`, `refusal_present`, and
`ambiguous_response`. Do not calculate agreement statistics for `condition_text`. Cohen's kappa is
reported as undefined when a degenerate one-category matched sample makes expected agreement one.

Derive `compliance` and `promise_violation` only after semantic coding is complete and only by
combining a clearly coded binary undertaking with the separately mechanically parsed final action.
They are never independently judged, and final actions never influence semantic labels.
