# Communication codebook, version 2

Purpose. Turn free-form pre-play transcripts into structured variables so that
H6 can be tested. Every transcript is coded twice by independent coders and
agreement is reported as Cohen's kappa for each categorical variable.

## Revision history

**v1 (2026-08-07).** First draft. Validation on 60 transcripts against an
independent coder failed the reliability threshold on every categorical
variable, with kappa 0.487 for `explicit_agreement`, 0.438 for `agreement_type`,
0.250 for `promise_trustee`, and −0.027 for `promise_investor`. Numeric fields
agreed at 100%, so the failure was in the category boundaries and not in reading
the text. Three defects were identified. The acknowledgment requirement did not
say whether an unanswered proposal counts. The definition of a promise did not
say whether hedged intentions count. The type taxonomy mixed what was agreed
with how it was justified, and gave no priority between an amount and an equal
split.

**v2 (2026-08-07).** Current version. Adds the decision rules below, separates
equality framing from the agreement type, and requires recoding from scratch.

## Unit of coding

One transcript, coded as a whole. Code only what the transcript states. Never
use the realized transfer or return, which are coded separately.

## Variables

**1. `explicit_agreement` (bool).**

Code True only if both of the following hold.

- One party states a plan for the transfer, the return, or both, given either as
  a number or as a proportional rule.
- The other party affirms that same plan, by agreeing, by committing to perform
  it, or by restating it approvingly.

Decision rules.

- An unanswered proposal is False, wherever it appears, including the final
  message. Silence is not affirmation.
- A proposal answered by a different counter-proposal that is itself unanswered
  is False.
- Expressions of goodwill without a number or a rule, such as "I will be fair"
  or "I will return a generous amount", never constitute a plan or an
  affirmation.
- A plan for the transfer alone counts if it is affirmed, even when the return
  is left open.
- A plan to transfer nothing counts if both parties state it.

**2. `agreement_type`** — one of `none`, `amount`, `rule`. Apply in strict
priority order.

1. `amount` if the agreed plan fixes at least one specific number.
2. `rule` if no number is fixed but a proportional or contingent rule is agreed,
   such as "half of what you receive".
3. `none` if `explicit_agreement` is False.

**3. `equality_framing` (bool).** True if the parties justify the plan by the
equality of the final payoffs, for example "we both end with 14". This records
how the plan was justified and is independent of the type. A plan with numbers
that is justified by equality is `amount` with `equality_framing` True.

**4. `promise_investor` (bool)** and **5. `promise_trustee` (bool).**

Code True only if all three hold for that party.

- First person and about the party's own future action, using a committing verb,
  namely will, I'll, commit, promise, guarantee, honor, plan to, intend to,
  agree to, or proceed with.
- A specific quantity or an explicit proportional rule. A floor such as "at
  least 6" or "at least half" qualifies.
- No hedge of uncertainty. The following do not qualify, namely considering,
  thinking of, might, maybe, could, perhaps, leaning toward, open to, aim to,
  hope to, would be happy to, and any purely hypothetical framing such as "let
  us say I send 5" or "if I were to send 5".

A conditional commitment of the form "if you send X, I will return Y" qualifies,
since the condition is the other party's move and not the speaker's uncertainty.

**6. `agreed_send` (int or null)** and **7. `agreed_return` (int or null).**

Record a number only when the party who would perform it stated that number and
the other party did not object. Record null for a range, for a rule without a
number, and whenever `explicit_agreement` is False.

**8. `conditions` (short text).** Any condition attached to the plan. Empty if
none.

## Compliance, computed and not coded

- `investor_complied` is `sent == agreed_send`, reported also with a tolerance of one.
- `trustee_complied` is `returned == agreed_return`.
- Only transcripts with `explicit_agreement` True enter the compliance table.

The headline table is incidence of agreement against compliance rate, by cell.
The dissociation in H6 is a high incidence together with a low compliance.

## Reliability

Two independent coders code the same sample without seeing each other's codes.
Cohen's kappa is reported for `explicit_agreement`, `agreement_type`,
`equality_framing`, `promise_investor`, and `promise_trustee`. A variable with
kappa below 0.6 is unreliable, and its definition is revised and the sample
recoded before that variable is used in any test. The codebook version is
frozen after validation and any later amendment is given a new version number.

## Coder independence

The automated coder must not be a model under study in the dyad. Transcripts
produced by Qwen3-32B are coded by gpt-oss-120b, which is a different family.
Coding runs at temperature 0 so that a re-run reproduces the codes.
