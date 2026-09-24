# Coding records

`coder_b_v1.jsonl`, `coder_b_v2.jsonl`
: Independent coding of the 60-transcript validation sample (`../coding_sample.txt`),
  under codebook v1 and v2. Coder B is an automated coder from a model family
  distinct from both the models under study and the judge. Coded before any judge
  output existed, so the two records are independent.

`judge_v1_gpt-oss-120b-nebius.jsonl`, `judge_gpt-oss-120b-nebius.jsonl`
: Full-set coding of all 775 communication transcripts by gpt-oss-120b at
  temperature 0, under codebook v1 and v2 respectively.

PENDING: human coding of the same 60-transcript sample. Once available, save as
`coder_human_v2.jsonl` and rerun `compare_coders.py` to replace the interim
machine-to-machine reliability figures with human-to-machine kappa.
