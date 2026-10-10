# Final Promise Switch statistical analysis — article_15

All inferential estimates are model-configuration-specific. No pooled confirmatory test was calculated.

## Included configurations

- gpt-oss-120b (`gpt_oss_120b_nebius_confirmatory_qcv3_20260823`)
- Qwen3-32B (`qwen3_32b_off_confirmatory_clean_20260822`)
- DeepSeek-V4-Flash (`deepseek_v4_flash_off_confirmatory_qcv3_20260823`)
- DeepSeek-V4-Pro (`deepseek_v4_pro_off_confirmatory_qcv3_v2_20260823`)
- DeepSeek-V4-Pro (r) (`deepseek_v4_pro_on_confirmatory_qcv3_v2_20260823`)
- Kimi-K3 (`kimi_k3_confirmatory_qcv3_20260823`)
- MiniMax-M3 (`minimax_m3_off_confirmatory_qcv3_20260823`)
- MiniMax-M3 (r) (`minimax_m3_on_confirmatory_qcv3_20260823`)
- GLM-5.2 (`glm_5_2_off_confirmatory_qcv3_20260823`)
- GLM-5.2 (r) (`glm_5_2_on_confirmatory_qcv3_20260822`)
- Llama-3.3-70B (`llama_3_3_70b_confirmatory_qcv3_20260823`)
- GPT-5.6 (`runs/promise_switch/gpt_5_6_luna_off_confirmatory_20260918`)
- GPT-5.6 (r) (`runs/promise_switch/gpt_5_6_luna_on_medium_confirmatory_20260918`)
- Sonnet-5 (`runs/promise_switch/claude_sonnet_5_off_confirmatory_20260917`)
- Sonnet-5 (r) (`runs/promise_switch/claude_sonnet_5_on_xhigh_confirmatory_20260918`)

## H7

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| Qwen3-32B | 99 | 100 | 0.990 | 0 | 100 | 0.000 | 0.990 | 0.932 | 0.998 | Pearson chi-square (two-sided, no Yates correction) | 1.53e-44 | 4.58e-44 |
| DeepSeek-V4-Flash | 100 | 100 | 1.000 | 22 | 100 | 0.220 | 0.780 | 0.682 | 0.850 | Pearson chi-square (two-sided, no Yates correction) | 1.20e-29 | 3.60e-29 |
| DeepSeek-V4-Pro | 100 | 100 | 1.000 | 53 | 100 | 0.530 | 0.470 | 0.368 | 0.567 | Pearson chi-square (two-sided, no Yates correction) | 4.57e-15 | 9.14e-15 |
| DeepSeek-V4-Pro (r) | 77 | 100 | 0.770 | 3 | 100 | 0.030 | 0.740 | 0.633 | 0.814 | Pearson chi-square (two-sided, no Yates correction) | 1.25e-26 | 2.50e-26 |
| Kimi-K3 | 100 | 100 | 1.000 | 66 | 100 | 0.660 | 0.340 | 0.247 | 0.437 | Pearson chi-square (two-sided, no Yates correction) | 1.55e-10 | 3.10e-10 |
| MiniMax-M3 | 100 | 100 | 1.000 | 27 | 100 | 0.270 | 0.730 | 0.629 | 0.807 | Pearson chi-square (two-sided, no Yates correction) | 8.03e-27 | 2.41e-26 |
| MiniMax-M3 (r) | 95 | 100 | 0.950 | 3 | 100 | 0.030 | 0.920 | 0.838 | 0.955 | Pearson chi-square (two-sided, no Yates correction) | 1.03e-38 | 3.08e-38 |
| GLM-5.2 | 100 | 100 | 1.000 | 23 | 100 | 0.230 | 0.770 | 0.671 | 0.842 | Pearson chi-square (two-sided, no Yates correction) | 4.59e-29 | 9.19e-29 |
| GLM-5.2 (r) | 100 | 100 | 1.000 | 70 | 100 | 0.700 | 0.300 | 0.211 | 0.396 | Pearson chi-square (two-sided, no Yates correction) | 2.83e-09 | 5.67e-09 |
| Llama-3.3-70B | 100 | 100 | 1.000 | 90 | 100 | 0.900 | 0.100 | 0.042 | 0.174 | Pearson chi-square (two-sided, no Yates correction) | 0.001 | 0.004 |
| GPT-5.6 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 (r) | 100 | 100 | 1.000 | 1 | 100 | 0.010 | 0.990 | 0.932 | 0.998 | Pearson chi-square (two-sided, no Yates correction) | 1.53e-44 | 4.58e-44 |
| Sonnet-5 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Sonnet-5 (r) | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## H8

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| Qwen3-32B | 99 | 100 | 0.990 | 73 | 100 | 0.730 | 0.260 | 0.171 | 0.355 | Pearson chi-square (two-sided, no Yates correction) | 1.17e-07 | 1.17e-07 |
| DeepSeek-V4-Flash | 100 | 100 | 1.000 | 89 | 100 | 0.890 | 0.110 | 0.050 | 0.186 | Pearson chi-square (two-sided, no Yates correction) | 6.45e-04 | 6.45e-04 |
| DeepSeek-V4-Pro | 100 | 100 | 1.000 | 86 | 100 | 0.860 | 0.140 | 0.074 | 0.221 | Pearson chi-square (two-sided, no Yates correction) | 1.04e-04 | 1.04e-04 |
| DeepSeek-V4-Pro (r) | 77 | 100 | 0.770 | 16 | 100 | 0.160 | 0.610 | 0.486 | 0.703 | Pearson chi-square (two-sided, no Yates correction) | 5.24e-18 | 5.24e-18 |
| Kimi-K3 | 100 | 100 | 1.000 | 96 | 100 | 0.960 | 0.040 | -0.004 | 0.098 | Fisher exact (two-sided) | 0.121 | 0.121 |
| MiniMax-M3 | 100 | 100 | 1.000 | 95 | 100 | 0.950 | 0.050 | 0.003 | 0.112 | Fisher exact (two-sided) | 0.059 | 0.059 |
| MiniMax-M3 (r) | 95 | 100 | 0.950 | 65 | 100 | 0.650 | 0.300 | 0.194 | 0.402 | Pearson chi-square (two-sided, no Yates correction) | 1.14e-07 | 1.14e-07 |
| GLM-5.2 | 100 | 100 | 1.000 | 60 | 100 | 0.600 | 0.400 | 0.302 | 0.498 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-12 | 1.54e-12 |
| GLM-5.2 (r) | 100 | 100 | 1.000 | 92 | 100 | 0.920 | 0.080 | 0.026 | 0.150 | Fisher exact (two-sided) | 0.007 | 0.007 |
| Llama-3.3-70B | 100 | 100 | 1.000 | 90 | 100 | 0.900 | 0.100 | 0.042 | 0.174 | Pearson chi-square (two-sided, no Yates correction) | 0.001 | 0.004 |
| GPT-5.6 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 (r) | 100 | 100 | 1.000 | 69 | 100 | 0.690 | 0.310 | 0.220 | 0.406 | Pearson chi-square (two-sided, no Yates correction) | 1.39e-09 | 1.39e-09 |
| Sonnet-5 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Sonnet-5 (r) | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## H9

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 100 | 100 | 1.000 | 8 | 100 | 0.080 | 0.920 | 0.841 | 0.959 | Pearson chi-square (two-sided, no Yates correction) | 6.14e-39 | 1.84e-38 |
| Qwen3-32B | 99 | 100 | 0.990 | 18 | 100 | 0.180 | 0.810 | 0.713 | 0.874 | Pearson chi-square (two-sided, no Yates correction) | 3.10e-31 | 6.20e-31 |
| DeepSeek-V4-Flash | 100 | 100 | 1.000 | 78 | 100 | 0.780 | 0.220 | 0.141 | 0.311 | Pearson chi-square (two-sided, no Yates correction) | 6.63e-07 | 1.33e-06 |
| DeepSeek-V4-Pro | 100 | 100 | 1.000 | 48 | 100 | 0.480 | 0.520 | 0.416 | 0.615 | Pearson chi-square (two-sided, no Yates correction) | 5.17e-17 | 1.55e-16 |
| DeepSeek-V4-Pro (r) | 77 | 100 | 0.770 | 0 | 100 | 0.000 | 0.770 | 0.671 | 0.842 | Pearson chi-square (two-sided, no Yates correction) | 4.59e-29 | 1.38e-28 |
| Kimi-K3 | 100 | 100 | 1.000 | 65 | 100 | 0.650 | 0.350 | 0.256 | 0.447 | Pearson chi-square (two-sided, no Yates correction) | 7.35e-11 | 2.20e-10 |
| MiniMax-M3 | 100 | 100 | 1.000 | 91 | 100 | 0.910 | 0.090 | 0.034 | 0.162 | Fisher exact (two-sided) | 0.003 | 0.006 |
| MiniMax-M3 (r) | 95 | 100 | 0.950 | 5 | 100 | 0.050 | 0.900 | 0.813 | 0.940 | Pearson chi-square (two-sided, no Yates correction) | 4.14e-37 | 8.27e-37 |
| GLM-5.2 | 100 | 100 | 1.000 | 20 | 100 | 0.200 | 0.800 | 0.704 | 0.867 | Pearson chi-square (two-sided, no Yates correction) | 7.64e-31 | 2.29e-30 |
| GLM-5.2 (r) | 100 | 100 | 1.000 | 60 | 100 | 0.600 | 0.400 | 0.302 | 0.498 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-12 | 4.61e-12 |
| Llama-3.3-70B | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 (r) | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| Sonnet-5 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Sonnet-5 (r) | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## Secondary randomized Voluntary Message ITT

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 21 | 100 | 0.210 | 28 | 100 | 0.280 | -0.070 | -0.187 | 0.049 | Pearson chi-square (two-sided, no Yates correction) | 0.250 | 1.000 |
| Qwen3-32B | 95 | 100 | 0.950 | 25 | 112 | 0.223 | 0.727 | 0.621 | 0.800 | Pearson chi-square (two-sided, no Yates correction) | 1.59e-26 | 2.07e-25 |
| DeepSeek-V4-Flash | 92 | 101 | 0.911 | 87 | 100 | 0.870 | 0.041 | -0.048 | 0.131 | Pearson chi-square (two-sided, no Yates correction) | 0.353 | 1.000 |
| DeepSeek-V4-Pro | 94 | 100 | 0.940 | 73 | 100 | 0.730 | 0.210 | 0.109 | 0.310 | Pearson chi-square (two-sided, no Yates correction) | 6.32e-05 | 3.79e-04 |
| DeepSeek-V4-Pro (r) | 16 | 100 | 0.160 | 0 | 100 | 0.000 | 0.160 | 0.090 | 0.244 | Pearson chi-square (two-sided, no Yates correction) | 3.04e-05 | 2.13e-04 |
| Kimi-K3 | 98 | 100 | 0.980 | 51 | 101 | 0.505 | 0.475 | 0.367 | 0.572 | Pearson chi-square (two-sided, no Yates correction) | 1.48e-14 | 1.77e-13 |
| MiniMax-M3 | 88 | 100 | 0.880 | 55 | 101 | 0.545 | 0.335 | 0.213 | 0.445 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-07 | 1.23e-06 |
| MiniMax-M3 (r) | 28 | 100 | 0.280 | 0 | 101 | 0.000 | 0.280 | 0.193 | 0.375 | Pearson chi-square (two-sided, no Yates correction) | 9.92e-09 | 8.93e-08 |
| GLM-5.2 | 98 | 100 | 0.980 | 53 | 103 | 0.515 | 0.465 | 0.359 | 0.562 | Pearson chi-square (two-sided, no Yates correction) | 3.07e-14 | 3.38e-13 |
| GLM-5.2 (r) | 97 | 100 | 0.970 | 56 | 103 | 0.544 | 0.426 | 0.319 | 0.524 | Pearson chi-square (two-sided, no Yates correction) | 1.82e-12 | 1.82e-11 |
| Llama-3.3-70B | 100 | 100 | 1.000 | 3 | 100 | 0.030 | 0.970 | 0.904 | 0.990 | Pearson chi-square (two-sided, no Yates correction) | 7.29e-43 | 1.09e-41 |
| GPT-5.6 | 100 | 100 | 1.000 | 99 | 100 | 0.990 | 0.010 | -0.028 | 0.054 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 (r) | 97 | 100 | 0.970 | 0 | 100 | 0.000 | 0.970 | 0.904 | 0.990 | Pearson chi-square (two-sided, no Yates correction) | 7.29e-43 | 1.09e-41 |
| Sonnet-5 | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Sonnet-5 (r) | 100 | 100 | 1.000 | 98 | 100 | 0.980 | 0.020 | -0.020 | 0.070 | Fisher exact (two-sided) | 0.497 | 1.000 |

## Hypothesis-support summary

- H7: 12/15 positive; 12/15 Holm-significant in the predicted direction.
- H8: 12/15 positive; 10/15 Holm-significant in the predicted direction.
- H9: 11/15 positive; 11/15 Holm-significant in the predicted direction.

## Attrition and extreme-case bounds

Across assigned condition-level trials, 562/9584 final outcomes were missing. Detailed model-by-condition and contrast-level diagnostics are in the CSV outputs.
Separately, 6 voluntary-message trials failed before receiving a Switch/No-Switch assignment.

- H7: worst-case RD remained positive for 12/15 configurations.
- H8: worst-case RD remained positive for 12/15 configurations.
- H9: worst-case RD remained positive for 11/15 configurations.
- Voluntary ITT: worst-case RD remained positive for 12/15 configurations.

## Semantic analyses

The profile-specific semantic analysis contains 3,135 annotations. Promise-conditioned subset n=1.

The pre-specified promise-conditioned behavioral comparison is not meaningfully estimable because only one qualifying voluntary explicit promise occurred.

The secondary undertaking-to-cooperate subset contained 159 observations; No-Switch minus Switch RD=0.411, 95% Newcombe CI [0.253, 0.542], raw p=5.69e-07. This is a selected post-message subgroup, not ITT.

Compliance and violation numerators and denominators are reported in `compliance_descriptives.csv`.

## Article-profile validation

Configuration count: 15; conditions: 6; configuration-condition cells: 90.
Assigned trials: 9,584; valid final decisions: 9,022; missing outcomes: 562.
Duplicate trial IDs: 0; duplicate trial-level rows: 0.

## Variance decomposition

LPM total-variance shares: configuration=0.289288, condition=0.173081, interaction=0.162914, residual=0.373889; R²=0.626111.
Logit explained-deviance shares: configuration=0.518161, condition=0.339283, interaction=0.142556; McFadden pseudo-R²=0.590909.

## Excluded configuration

DeepSeek V4 Flash reasoning ON is EXCLUDED FROM CONFIRMATORY INFERENCE. The run stopped with status `PAUSED_BEHAVIORAL_PARSER_REVIEW` after 361 terminal trials; 316 had valid binary outcomes, 22 were behavioral failures, and 23 were infrastructure failures.

The diagnostic describes only this specific model/provider/reasoning configuration.

## Statistical definitions

- Confidence intervals: 95% Newcombe-Wilson intervals for independent-proportion differences.
- Tests: two-sided Fisher exact if any expected cell count was below 5; otherwise two-sided Pearson chi-square without Yates correction.
- Confirmatory multiplicity: Holm correction within each configuration across H7, H8, and H9.
- Secondary multiplicity: Holm correction across the 15 Voluntary Message ITT comparisons.
- Extreme-case missing-outcome bounds are deterministic bounds, not confidence intervals.
