# Communication codebook (Berg trust game, communication condition)

Purpose. Turn free-form pre-play transcripts into structured variables so that
H6 (communication raises agreements, and formation dissociates from execution)
can be tested. Every communication dyad is coded twice, once by a human and once
by an LLM judge, and agreement between the two is reported as Cohen's kappa
(PI directive 2026-07-22, point 6). Coding is done on the transcript ONLY, never
on the private reasoning or the realized transfer, so a coder cannot be biased by
the outcome.

## Unit of coding

One transcript = the full message exchange of one dyad before the private
decisions. Code the transcript as a whole, not message by message.

## Variables

1. `explicit_agreement` (bool). True only if the two parties arrive at a stated
   joint plan for the transfer and/or the return that both acknowledge. A single
   party stating an intention is not an agreement. Vague goodwill
   ("let's both be fair") is not an agreement unless it fixes an amount or a rule.

2. `agreement_type` (one of `none`, `amount`, `rule`, `equal_split`).
   - `amount`  a specific transfer and/or return is named (e.g. send 10, return 15).
   - `rule`    a contingent rule is agreed (e.g. "return double what you receive").
   - `equal_split` the plan divides the final pie equally.
   - `none`    no explicit agreement.

3. `promise_investor` (bool). The first mover states a commitment about the
   amount it will send.

4. `promise_trustee` (bool). The second mover states a commitment about the
   amount it will return.

5. `agreed_send` (int or null). The transfer the plan specifies, if any.

6. `agreed_return` (int or null). The return the plan specifies, if any.

7. `conditions` (short text). Any conditions attached to the plan
   ("I will return 15 only if you send at least 8"). Empty if none.

## Compliance (computed, not coded)

Compliance is derived by comparing the coded plan to the realized decisions.

- `investor_complied` = `sent == agreed_send` (within 0 tolerance; report also a
  within-1 variant).
- `trustee_complied` = `returned == agreed_return`.
- A dyad enters the compliance table only if `explicit_agreement` is True.

The headline table is **incidence of agreement → compliance rate**, by cell
(surface × condition × model pair). Incidence = share of dyads with
`explicit_agreement`. Compliance = share of agreeing dyads whose realized
behavior matched the plan. The dissociation in H6 is a high incidence with a
low compliance.

## Double coding and reliability

- Human codes a random sample (target ≥ 20% of transcripts, minimum 100).
- The LLM judge codes all transcripts.
- Report Cohen's kappa on the overlapping sample for each categorical variable
  (`explicit_agreement`, `agreement_type`, `promise_investor`, `promise_trustee`).
- Kappa below 0.6 on a variable means the definition is unreliable; revise this
  codebook and recode before using that variable in a test.

## Judge independence

The judge model must not be one of the models under study for that dyad, to avoid
a model grading its own transcripts. Use a cheap neutral judge (e.g. DeepSeek V4
Flash) for the full pass and a stronger model on the human-overlap sample as a
second check.
