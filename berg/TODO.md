# TODO

## Done, 2026-09-14

One kappa now covers all fifteen configurations. The judge recoded 10,725
transcripts with the speakers named, 4 came back empty, and a human sample of 75
drawn at 5 per configuration was coded against it. `run_kappa.sh` runs the whole
sequence again if it has to be repeated, and `_transcripts.py` keeps the two
storage layouts resolved in one place.

    explicit_agreement   0.578    agreement_type   0.598
    equality_framing     0.681    promise_trustee  0.559
    promise_investor     0.315

The speaker labels turned out to matter to the human and barely at all to the
judge, whose marginals moved by under one percentage point and which agreed with
its own earlier reading between 0.78 and 0.96. That comparison is now the
machine reliability figure in the paper, on 7,916 transcripts.

Open, in order of how much they would add.

1. **A second, independent judge.** The reliability figure is a reread by the
   same model, which measures reproducibility. A model from outside the study
   would measure agreement between instruments. `./run_kappa.sh screen` looks
   for one that honours the schema, and `JUDGE_B=<key> ./run_kappa.sh reread`
   runs it. Earlier candidates failed on output format, so screen before paying.

2. **Four transcripts returned an empty reply** and sit in the file as unclean.
   They are 0.04\% of the set and affect nothing, but a rerun of
   `./run_kappa.sh judge` would pick them up.

3. **The recognition probe** on the first clause of H1 is still unrun, about $1
   and twenty minutes, `python3 run_probe.py`.

4. **Arina still owes two figures**, `fig_PS_confirmatory_forest.pdf` and
   `fig_PS_voluntary_itt.pdf`. The build draws a labelled placeholder for each
   until they land.

