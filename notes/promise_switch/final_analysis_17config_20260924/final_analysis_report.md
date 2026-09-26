# Final Promise Switch statistical analysis

All inferential estimates are model-configuration-specific. No pooled confirmatory test was calculated.

## Included configurations

- Qwen3-32B — reasoning ON (`qwen3_32b_on_confirmatory_clean_20260821`)
- Qwen3-32B — reasoning OFF (`qwen3_32b_off_confirmatory_clean_20260822`)
- GLM-5.2 — reasoning ON (`glm_5_2_on_confirmatory_qcv3_20260822`)
- GLM-5.2 — reasoning OFF (`glm_5_2_off_confirmatory_qcv3_20260823`)
- MiniMax M3 — reasoning ON (`minimax_m3_on_confirmatory_qcv3_20260823`)
- MiniMax M3 — reasoning OFF (`minimax_m3_off_confirmatory_qcv3_20260823`)
- DeepSeek V4 Pro — reasoning ON (`deepseek_v4_pro_on_confirmatory_qcv3_v2_20260823`)
- DeepSeek V4 Pro — reasoning OFF (`deepseek_v4_pro_off_confirmatory_qcv3_v2_20260823`)
- DeepSeek V4 Flash — reasoning OFF (`deepseek_v4_flash_off_confirmatory_qcv3_20260823`)
- Kimi K3 (`kimi_k3_confirmatory_qcv3_20260823`)
- Llama 3.3 70B (`llama_3_3_70b_confirmatory_qcv3_20260823`)
- GPT-OSS-120B — Nebius (`gpt_oss_120b_nebius_confirmatory_qcv3_20260823`)
- GPT-OSS-120B — Groq (`gpt_oss_120b_groq_confirmatory_qcv3_10rpm_20260823`)
- GPT-5.6 Luna — no-think (`runs/promise_switch/gpt_5_6_luna_off_confirmatory_20260918`)
- GPT-5.6 Luna — think, medium (`runs/promise_switch/gpt_5_6_luna_on_medium_confirmatory_20260918`)
- Claude Sonnet 5 — no-think (`runs/promise_switch/claude_sonnet_5_off_confirmatory_20260917`)
- Claude Sonnet 5 — think, adaptive xHigh (`runs/promise_switch/claude_sonnet_5_on_xhigh_confirmatory_20260918`)

## H8

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B — reasoning ON | 82 | 100 | 0.820 | 0 | 100 | 0.000 | 0.820 | 0.726 | 0.883 | Pearson chi-square (two-sided, no Yates correction) | 4.44e-32 | 1.33e-31 |
| Qwen3-32B — reasoning OFF | 99 | 100 | 0.990 | 0 | 100 | 0.000 | 0.990 | 0.932 | 0.998 | Pearson chi-square (two-sided, no Yates correction) | 1.53e-44 | 4.58e-44 |
| GLM-5.2 — reasoning ON | 100 | 100 | 1.000 | 70 | 100 | 0.700 | 0.300 | 0.211 | 0.396 | Pearson chi-square (two-sided, no Yates correction) | 2.83e-09 | 5.67e-09 |
| GLM-5.2 — reasoning OFF | 100 | 100 | 1.000 | 23 | 100 | 0.230 | 0.770 | 0.671 | 0.842 | Pearson chi-square (two-sided, no Yates correction) | 4.59e-29 | 9.19e-29 |
| MiniMax M3 — reasoning ON | 95 | 100 | 0.950 | 3 | 100 | 0.030 | 0.920 | 0.838 | 0.955 | Pearson chi-square (two-sided, no Yates correction) | 1.03e-38 | 3.08e-38 |
| MiniMax M3 — reasoning OFF | 100 | 100 | 1.000 | 27 | 100 | 0.270 | 0.730 | 0.629 | 0.807 | Pearson chi-square (two-sided, no Yates correction) | 8.03e-27 | 2.41e-26 |
| DeepSeek V4 Pro — reasoning ON | 77 | 100 | 0.770 | 3 | 100 | 0.030 | 0.740 | 0.633 | 0.814 | Pearson chi-square (two-sided, no Yates correction) | 1.25e-26 | 2.50e-26 |
| DeepSeek V4 Pro — reasoning OFF | 100 | 100 | 1.000 | 53 | 100 | 0.530 | 0.470 | 0.368 | 0.567 | Pearson chi-square (two-sided, no Yates correction) | 4.57e-15 | 9.14e-15 |
| DeepSeek V4 Flash — reasoning OFF | 100 | 100 | 1.000 | 22 | 100 | 0.220 | 0.780 | 0.682 | 0.850 | Pearson chi-square (two-sided, no Yates correction) | 1.20e-29 | 3.60e-29 |
| Kimi K3 | 100 | 100 | 1.000 | 66 | 100 | 0.660 | 0.340 | 0.247 | 0.437 | Pearson chi-square (two-sided, no Yates correction) | 1.55e-10 | 3.10e-10 |
| Llama 3.3 70B | 100 | 100 | 1.000 | 90 | 100 | 0.900 | 0.100 | 0.042 | 0.174 | Pearson chi-square (two-sided, no Yates correction) | 0.001 | 0.004 |
| GPT-OSS-120B — Nebius | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| GPT-OSS-120B — Groq | 100 | 100 | 1.000 | 3 | 100 | 0.030 | 0.970 | 0.904 | 0.990 | Pearson chi-square (two-sided, no Yates correction) | 7.29e-43 | 2.19e-42 |
| GPT-5.6 Luna — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 Luna — think, medium | 100 | 100 | 1.000 | 1 | 100 | 0.010 | 0.990 | 0.932 | 0.998 | Pearson chi-square (two-sided, no Yates correction) | 1.53e-44 | 4.58e-44 |
| Claude Sonnet 5 — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Claude Sonnet 5 — think, adaptive xHigh | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## H9

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B — reasoning ON | 82 | 100 | 0.820 | 48 | 100 | 0.480 | 0.340 | 0.210 | 0.454 | Pearson chi-square (two-sided, no Yates correction) | 4.64e-07 | 4.64e-07 |
| Qwen3-32B — reasoning OFF | 99 | 100 | 0.990 | 73 | 100 | 0.730 | 0.260 | 0.171 | 0.355 | Pearson chi-square (two-sided, no Yates correction) | 1.17e-07 | 1.17e-07 |
| GLM-5.2 — reasoning ON | 100 | 100 | 1.000 | 92 | 100 | 0.920 | 0.080 | 0.026 | 0.150 | Fisher exact (two-sided) | 0.007 | 0.007 |
| GLM-5.2 — reasoning OFF | 100 | 100 | 1.000 | 60 | 100 | 0.600 | 0.400 | 0.302 | 0.498 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-12 | 1.54e-12 |
| MiniMax M3 — reasoning ON | 95 | 100 | 0.950 | 65 | 100 | 0.650 | 0.300 | 0.194 | 0.402 | Pearson chi-square (two-sided, no Yates correction) | 1.14e-07 | 1.14e-07 |
| MiniMax M3 — reasoning OFF | 100 | 100 | 1.000 | 95 | 100 | 0.950 | 0.050 | 0.003 | 0.112 | Fisher exact (two-sided) | 0.059 | 0.059 |
| DeepSeek V4 Pro — reasoning ON | 77 | 100 | 0.770 | 16 | 100 | 0.160 | 0.610 | 0.486 | 0.703 | Pearson chi-square (two-sided, no Yates correction) | 5.24e-18 | 5.24e-18 |
| DeepSeek V4 Pro — reasoning OFF | 100 | 100 | 1.000 | 86 | 100 | 0.860 | 0.140 | 0.074 | 0.221 | Pearson chi-square (two-sided, no Yates correction) | 1.04e-04 | 1.04e-04 |
| DeepSeek V4 Flash — reasoning OFF | 100 | 100 | 1.000 | 89 | 100 | 0.890 | 0.110 | 0.050 | 0.186 | Pearson chi-square (two-sided, no Yates correction) | 6.45e-04 | 6.45e-04 |
| Kimi K3 | 100 | 100 | 1.000 | 96 | 100 | 0.960 | 0.040 | -0.004 | 0.098 | Fisher exact (two-sided) | 0.121 | 0.121 |
| Llama 3.3 70B | 100 | 100 | 1.000 | 90 | 100 | 0.900 | 0.100 | 0.042 | 0.174 | Pearson chi-square (two-sided, no Yates correction) | 0.001 | 0.004 |
| GPT-OSS-120B — Nebius | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| GPT-OSS-120B — Groq | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| GPT-5.6 Luna — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 Luna — think, medium | 100 | 100 | 1.000 | 69 | 100 | 0.690 | 0.310 | 0.220 | 0.406 | Pearson chi-square (two-sided, no Yates correction) | 1.39e-09 | 1.39e-09 |
| Claude Sonnet 5 — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Claude Sonnet 5 — think, adaptive xHigh | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## H10

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B — reasoning ON | 82 | 100 | 0.820 | 2 | 100 | 0.020 | 0.800 | 0.700 | 0.865 | Pearson chi-square (two-sided, no Yates correction) | 2.06e-30 | 4.12e-30 |
| Qwen3-32B — reasoning OFF | 99 | 100 | 0.990 | 18 | 100 | 0.180 | 0.810 | 0.713 | 0.874 | Pearson chi-square (two-sided, no Yates correction) | 3.10e-31 | 6.20e-31 |
| GLM-5.2 — reasoning ON | 100 | 100 | 1.000 | 60 | 100 | 0.600 | 0.400 | 0.302 | 0.498 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-12 | 4.61e-12 |
| GLM-5.2 — reasoning OFF | 100 | 100 | 1.000 | 20 | 100 | 0.200 | 0.800 | 0.704 | 0.867 | Pearson chi-square (two-sided, no Yates correction) | 7.64e-31 | 2.29e-30 |
| MiniMax M3 — reasoning ON | 95 | 100 | 0.950 | 5 | 100 | 0.050 | 0.900 | 0.813 | 0.940 | Pearson chi-square (two-sided, no Yates correction) | 4.14e-37 | 8.27e-37 |
| MiniMax M3 — reasoning OFF | 100 | 100 | 1.000 | 91 | 100 | 0.910 | 0.090 | 0.034 | 0.162 | Fisher exact (two-sided) | 0.003 | 0.006 |
| DeepSeek V4 Pro — reasoning ON | 77 | 100 | 0.770 | 0 | 100 | 0.000 | 0.770 | 0.671 | 0.842 | Pearson chi-square (two-sided, no Yates correction) | 4.59e-29 | 1.38e-28 |
| DeepSeek V4 Pro — reasoning OFF | 100 | 100 | 1.000 | 48 | 100 | 0.480 | 0.520 | 0.416 | 0.615 | Pearson chi-square (two-sided, no Yates correction) | 5.17e-17 | 1.55e-16 |
| DeepSeek V4 Flash — reasoning OFF | 100 | 100 | 1.000 | 78 | 100 | 0.780 | 0.220 | 0.141 | 0.311 | Pearson chi-square (two-sided, no Yates correction) | 6.63e-07 | 1.33e-06 |
| Kimi K3 | 100 | 100 | 1.000 | 65 | 100 | 0.650 | 0.350 | 0.256 | 0.447 | Pearson chi-square (two-sided, no Yates correction) | 7.35e-11 | 2.20e-10 |
| Llama 3.3 70B | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-OSS-120B — Nebius | 100 | 100 | 1.000 | 8 | 100 | 0.080 | 0.920 | 0.841 | 0.959 | Pearson chi-square (two-sided, no Yates correction) | 6.14e-39 | 1.84e-38 |
| GPT-OSS-120B — Groq | 100 | 100 | 1.000 | 16 | 100 | 0.160 | 0.840 | 0.748 | 0.899 | Pearson chi-square (two-sided, no Yates correction) | 2.34e-33 | 2.34e-33 |
| GPT-5.6 Luna — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 Luna — think, medium | 100 | 100 | 1.000 | 11 | 100 | 0.110 | 0.890 | 0.805 | 0.937 | Pearson chi-square (two-sided, no Yates correction) | 9.44e-37 | 1.89e-36 |
| Claude Sonnet 5 — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Claude Sonnet 5 — think, adaptive xHigh | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |

## Secondary randomized Voluntary Message ITT

| Configuration | T coop | T N | T rate | C coop | C N | C rate | RD | CI low | CI high | Test | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-32B — reasoning ON | 14 | 100 | 0.140 | 2 | 102 | 0.020 | 0.120 | 0.047 | 0.203 | Pearson chi-square (two-sided, no Yates correction) | 0.002 | 0.011 |
| Qwen3-32B — reasoning OFF | 95 | 100 | 0.950 | 25 | 112 | 0.223 | 0.727 | 0.621 | 0.800 | Pearson chi-square (two-sided, no Yates correction) | 1.59e-26 | 2.39e-25 |
| GLM-5.2 — reasoning ON | 97 | 100 | 0.970 | 56 | 103 | 0.544 | 0.426 | 0.319 | 0.524 | Pearson chi-square (two-sided, no Yates correction) | 1.82e-12 | 2.18e-11 |
| GLM-5.2 — reasoning OFF | 98 | 100 | 0.980 | 53 | 103 | 0.515 | 0.465 | 0.359 | 0.562 | Pearson chi-square (two-sided, no Yates correction) | 3.07e-14 | 4.00e-13 |
| MiniMax M3 — reasoning ON | 28 | 100 | 0.280 | 0 | 101 | 0.000 | 0.280 | 0.193 | 0.375 | Pearson chi-square (two-sided, no Yates correction) | 9.92e-09 | 1.09e-07 |
| MiniMax M3 — reasoning OFF | 88 | 100 | 0.880 | 55 | 101 | 0.545 | 0.335 | 0.213 | 0.445 | Pearson chi-square (two-sided, no Yates correction) | 1.54e-07 | 1.54e-06 |
| DeepSeek V4 Pro — reasoning ON | 16 | 100 | 0.160 | 0 | 100 | 0.000 | 0.160 | 0.090 | 0.244 | Pearson chi-square (two-sided, no Yates correction) | 3.04e-05 | 2.74e-04 |
| DeepSeek V4 Pro — reasoning OFF | 94 | 100 | 0.940 | 73 | 100 | 0.730 | 0.210 | 0.109 | 0.310 | Pearson chi-square (two-sided, no Yates correction) | 6.32e-05 | 5.06e-04 |
| DeepSeek V4 Flash — reasoning OFF | 92 | 101 | 0.911 | 87 | 100 | 0.870 | 0.041 | -0.048 | 0.131 | Pearson chi-square (two-sided, no Yates correction) | 0.353 | 1.000 |
| Kimi K3 | 98 | 100 | 0.980 | 51 | 101 | 0.505 | 0.475 | 0.367 | 0.572 | Pearson chi-square (two-sided, no Yates correction) | 1.48e-14 | 2.07e-13 |
| Llama 3.3 70B | 100 | 100 | 1.000 | 3 | 100 | 0.030 | 0.970 | 0.904 | 0.990 | Pearson chi-square (two-sided, no Yates correction) | 7.29e-43 | 1.24e-41 |
| GPT-OSS-120B — Nebius | 21 | 100 | 0.210 | 28 | 100 | 0.280 | -0.070 | -0.187 | 0.049 | Pearson chi-square (two-sided, no Yates correction) | 0.250 | 1.000 |
| GPT-OSS-120B — Groq | 36 | 100 | 0.360 | 21 | 100 | 0.210 | 0.150 | 0.025 | 0.269 | Pearson chi-square (two-sided, no Yates correction) | 0.019 | 0.113 |
| GPT-5.6 Luna — no-think | 100 | 100 | 1.000 | 99 | 100 | 0.990 | 0.010 | -0.028 | 0.054 | Fisher exact (two-sided) | 1.000 | 1.000 |
| GPT-5.6 Luna — think, medium | 97 | 100 | 0.970 | 0 | 100 | 0.000 | 0.970 | 0.904 | 0.990 | Pearson chi-square (two-sided, no Yates correction) | 7.29e-43 | 1.24e-41 |
| Claude Sonnet 5 — no-think | 100 | 100 | 1.000 | 100 | 100 | 1.000 | 0.000 | -0.037 | 0.037 | Fisher exact (two-sided) | 1.000 | 1.000 |
| Claude Sonnet 5 — think, adaptive xHigh | 100 | 100 | 1.000 | 98 | 100 | 0.980 | 0.020 | -0.020 | 0.070 | Fisher exact (two-sided) | 0.497 | 1.000 |

## Hypothesis-support summary

- H8: 14/17 positive; 14/17 Holm-significant in the predicted direction.
- H9: 14/17 positive; 12/17 Holm-significant in the predicted direction.
- H10: 13/17 positive; 13/17 Holm-significant in the predicted direction.

## Attrition and extreme-case bounds

Across assigned condition-level trials, 572/10796 final outcomes were missing. Detailed model-by-condition and contrast-level diagnostics are in the CSV outputs.
Separately, 7 voluntary-message trials failed before receiving a Switch/No-Switch assignment.

- H8: worst-case RD remained positive for 14/17 configurations.
- H9: worst-case RD remained positive for 14/17 configurations.
- H10: worst-case RD remained positive for 13/17 configurations.
- Voluntary ITT: worst-case RD remained positive for 14/17 configurations.

## Semantic analyses

The frozen semantic prevalence counts matched all expected counts across 2,738 annotations. Promise-conditioned subset n=1.

Explicit semantic-v2 activation contains 1 qualifying voluntary explicit promise records; this remains a separately identified descriptive subset.

The secondary undertaking-to-cooperate subset contained 175 observations; No-Switch minus Switch RD=0.424, 95% Newcombe CI [0.273, 0.549], raw p=8.12e-08. This is a selected post-message subgroup, not ITT.

Compliance and violation numerators and denominators are reported in `compliance_descriptives.csv`.

## Excluded configuration

DeepSeek V4 Flash reasoning ON is EXCLUDED FROM CONFIRMATORY INFERENCE. The run stopped with status `PAUSED_BEHAVIORAL_PARSER_REVIEW` after 361 terminal trials; 316 had valid binary outcomes, 22 were behavioral failures, and 23 were infrastructure failures.

The diagnostic describes only this specific model/provider/reasoning configuration.

## Statistical definitions

- Confidence intervals: 95% Newcombe-Wilson intervals for independent-proportion differences.
- Tests: two-sided Fisher exact if any expected cell count was below 5; otherwise two-sided Pearson chi-square without Yates correction.
- Confirmatory multiplicity: Holm correction within each configuration across H8, H9, and H10.
- Secondary multiplicity: Holm correction across the 17 Voluntary Message ITT comparisons.
- Extreme-case missing-outcome bounds are deterministic bounds, not confidence intervals.
