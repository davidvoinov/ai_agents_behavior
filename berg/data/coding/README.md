# Coding records

Communication transcripts are coded under `../../COMM_CODEBOOK.md`.

| File | Content |
|---|---|
| `judge_comm.jsonl` | gpt-oss-120b judge at temperature 0 over every communication transcript of the fifteen configurations, speakers named |
| `judge_comm.jsonl.pre_rolefix` | the same judge over the same transcripts, speakers unmarked |
| `kappa_round4/` | human validation sample, 75 transcripts at 5 per configuration: transcripts, coding form, key |
| `kappa_round2/` | human validation sample coded on transcripts with speakers unmarked |
| `promise_recode/` | human recode of the two promise variables |
| `judge_gpt-oss-120b-nebius.jsonl`, `judge_v1_gpt-oss-120b-nebius.jsonl` | judge codes under codebook v2 and v1 on the first 775 transcripts |
| `coder_b_v1.jsonl`, `coder_b_v2.jsonl` | second machine coder on the first 60-transcript sample; `make_kappa_sample.py` excludes these transcripts from later draws |

Kappa for the current round: `./run_kappa.sh score` from `berg/`.
