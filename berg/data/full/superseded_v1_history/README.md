# Superseded: social-history cells run under history generator v1

These dyads used a SINGLE fixed synthetic report per valence (identical for every
dyad in a cell), so the idiosyncrasy of that one report was perfectly confounded
with the valence. At the stimulus level the design was effectively n=1 per valence.

Generator v2 (run_full_cycle.py, N_HISTORY_VARIANTS=10) draws 10 independent
reports per valence, each with within-report variance and one row running against
its own valence, assigned to dyads by index. Valence separation stays clean:
mean return-to-transfer ratio 1.43-1.79 for positive against 0.26-0.45 for negative.

These records are retained for provenance and for the comparison of the two
stimulus regimes. They must NOT be pooled with v2 data.
