# Factor analysis — functional

traits from `trait-tests/out/model_trait_matrix.csv`; n = 78 models; p = 31 traits -> 31 parcels (parcel off, |r| >= 0.8, complete); rotation varimax; k = 2 by parallel

## Adequacy

Bartlett chi2 = 5712.9, df = 465, p = 0; KMO = 0.889

## Number of factors

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| observed eigenvalue | 21.83 | 2.71 | 1.91 | 1.26 | 0.80 | 0.57 | 0.49 | 0.29 |
| parallel-analysis threshold | 2.62 | 2.35 | 2.16 | 2.00 | 1.87 | 1.75 | 1.64 | 1.54 |

parallel analysis k = 2; Kaiser k = 4; MAP k = 10; **used k = 2 (parallel)**

## Variance

| factor | SS pattern | share of p | SS structure |
|---|---|---|---|
| F1 | 13.28 | 42.9% | 13.28 |
| F2 | 10.93 | 35.3% | 10.93 |

## Pattern loadings (|loading| >= 0.4)

- **F1**: F:C_trans_Evaluate>Planning +0.90, F:C_trans_Reflection>Planning +0.88, F:C_trans_Planning>Planning +0.85, F:C_trans_entropy +0.85, F:A_entropy +0.84, F:C_trans_Deciding>Planning +0.81, F:C_trans_Reflection>Executing +0.80, F:C_trans_Reasoning>Planning +0.80, F:B_units_per_step +0.79, F:A_prop_Planning +0.78, F:D_mid_Planning +0.74, F:D_late_Planning +0.74, F:D_mid_Retrieval -0.74, F:D_early_Planning +0.74, F:C_trans_Reflection>Reasoning +0.71, F:A_prop_Retrieval -0.67, F:C_trans_Executing>Planning +0.66, F:A_density +0.62, F:D_late_Executing -0.60, F:A_prop_Reflection +0.55, F:C_trans_Evaluate>Reasoning +0.55, F:C_trans_Reasoning>Reasoning +0.51, F:D_mid_Executing -0.47, F:C_trans_Planning>Reasoning +0.47
- **F2**: F:A_prop_Reasoning +0.95, F:D_late_Reasoning +0.94, F:D_mid_Reasoning +0.92, F:C_trans_Executing>Reasoning +0.92, F:D_early_Reasoning +0.88, F:C_trans_Planning>Reasoning +0.84, F:C_trans_Reasoning>Reasoning +0.81, F:C_trans_Evaluate>Reasoning +0.73, F:A_prop_Grounding -0.72, F:D_late_Executing -0.61, F:C_trans_Executing>Planning +0.59, F:D_mid_Executing -0.58, F:B_units_per_step +0.55, F:C_trans_Reasoning>Planning +0.54, F:C_trans_Reflection>Reasoning +0.54, F:A_prop_Planning +0.53, F:D_mid_Planning +0.53, F:D_early_Planning +0.53, F:D_late_Planning +0.50, F:C_trans_Planning>Planning +0.46, F:A_density +0.45

## Factor correlations

| | F1 | F2 |
|---|---|---|
| F1 | 1.00 | -0.00 |
| F2 | -0.00 | 1.00 |

communalities < .5: F:A_prop_Reflection 0.35, F:C_trans_Planning>Executing 0.17

## Models

- **F1** top: claude-sonnet-4-6 +1.98, kimi-k3 +1.72, claude-opus-4-8 +1.68, qwen3.5-35b-a3b +1.57, deepseek-v4-pro +1.29
  bottom: gpt-5.2 -1.62, o3-2025-04-16 -1.65, gpt-oss-20b -1.71, o4-mini-2025-04-16 -1.75, gpt-5.1 -2.65
- **F2** top: kimi-k2.7-code +3.22, llama-3.1-70b-instruct +3.13, minimax-m2.7 +2.57, gpt-4o-2024-05-13 +2.20, llama-3.1-405b-instruct +1.96
  bottom: deepseek-v3.2-exp -1.26, gemini-2.5-pro -1.41, gpt-5-mini-2025-08-07 -1.44, gpt-5-nano-2025-08-07 -1.51, gemini-2.5-flash -1.53
