# Human validation round, procedure

60 transcripts over 10 configurations and 7 surfaces, drawn with
seed 20260904, six surfaces from each. Every transcript used in the first validation
is excluded, so the codebook is tested against material it was not tuned on.

The two Qwen arms are left out of this round on purpose. Round one drew almost
all of their communication transcripts, which leaves them one and two usable
surfaces, and they are also the only configurations whose coding has already been
checked. Placing the fresh round on the ten untested configurations balances the
draw exactly and puts the check where no check has run.

## What to do

1. Open `COMM_CODEBOOK.md` and read the definitions and the decision rules.
2. Open `transcripts.md`. Code the items IN ORDER, one at a time, filling
   `coding_form.csv` as you go. Do not skip ahead and do not revise an earlier
   item after reading a later one, since both break the independence the kappa
   assumes.
3. Leave `ANSWERS_DO_NOT_OPEN_FIRST.jsonl` closed until the form is complete.
   It carries the judge codes and the realized decisions, and seeing either
   while coding contaminates the comparison.
4. Run `python3 score_kappa.py` when finished.

## Columns

| column | values |
|---|---|
| explicit_agreement | TRUE / FALSE |
| agreement_type | amount / rule / vague / none |
| equality_framing | TRUE / FALSE |
| promise_investor | TRUE / FALSE |
| promise_trustee | TRUE / FALSE |
| agreed_send | integer, or blank when no amount was agreed |
| agreed_return | integer, or blank when no amount was agreed |

Time to allow is roughly 2 to 4 minutes per transcript.

## What the result licenses

A kappa at or above 0.70 on the categorical variables supports reporting the
automated coding as validated against a human reading. Below that the codebook
needs another revision, and the revision must then be tested on a further fresh
sample, since a codebook repaired against a sample cannot be validated on it.
