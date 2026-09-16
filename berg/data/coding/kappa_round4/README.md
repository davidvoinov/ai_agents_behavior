# Human validation round 4, procedure

75 transcripts over 15 configurations and 7 surfaces, drawn
with seed 20260913. Every transcript an earlier round put in front of a person is
excluded, so the codebook is tested against material it was not tuned on.

This round replaces the earlier ones rather than adding to them. Rounds one
and two were read by a human who saw the label "party" on both sides of every
exchange, because the renderer looked for a field the engine never wrote. The
two variables defined per role were therefore coded by a person inferring who
spoke, and those two are the ones that failed, at 0.467 and 0.286, while the
three that need no role held between 0.648 and 0.781. The labels are repaired
here and the judge reads the same repaired transcripts, so one kappa now covers
every configuration in the estimates on material both coders could read.

## Procedure

1. Open `berg/COMM_CODEBOOK.md` and keep it open. Code to the codebook as
   written, without adjusting it as you go, since a codebook revised mid-sample
   makes the kappa a fit to the sample.
2. Read `transcripts.md` top to bottom and fill one row of `coding_form.csv`
   per item. Leave a cell empty only when the codebook says the variable does
   not apply.
3. Do not open `ANSWERS_DO_NOT_OPEN_FIRST.jsonl`. It carries the identity of
   each item and exists so the scoring can join your rows to the judge's.
4. When the form is complete, run `python3 score_kappa.py --round 4`.

## What the numbers will mean

`explicit_agreement` carries the compliance results and is the variable the
paper most depends on. `promise_investor` and `promise_trustee` are the open
question of this round. They were withdrawn from every estimate after the
earlier validation, where a high agreement between two machines sat next to a
poor agreement with a person, and the labels those coders were denied are
exactly what the two variables need. If they recover here, the earlier failure
was a rendering fault and the withdrawal can be revisited. If they stay low on
labelled transcripts, the conjunctive definition is what fails and the
withdrawal stands on firmer ground than before.

Code without trying to serve either outcome.
