# Promise recode, procedure

Sixty transcripts, the same ones as the first pass, reshuffled with seed 20260905
and stripped down to the two promise variables.

## Why all sixty

Recoding only the disputed items would compare a careful reading against a
hurried one on exactly the cases where they differed, which manufactures
agreement. The kappa from a partial recode would not mean anything.

## What to do

1. Keep the checklist at the top of `transcripts.md` visible while coding.
2. Code in order, filling `promise_form.csv` by the item number shown.
3. For every False, put 1, 2, or 3 in the failed_rule column for which condition
   decided it. That column is what makes the next disagreement diagnosable.
4. Do not open `MAPPING_DO_NOT_OPEN_FIRST.jsonl`, which links these items back to
   the first pass.
5. Run `python3 score_promise_recode.py` when finished.

## The two likely outcomes

A kappa at or above 0.70 confirms the first pass drifted from the written rule
and the coding is usable once corrected. A kappa that stays low says the
definition itself does not carry between readers, which is a finding about the
instrument and belongs in the paper as one.

Roughly 1 to 2 minutes per item, since only two judgements are needed.
