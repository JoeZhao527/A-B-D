# Factor analysis — linguistic

traits from `trait-tests/out/model_trait_matrix.csv`; n = 78 models; p = 48 traits -> 48 parcels (parcel off, |r| >= 0.8, complete); rotation varimax; k = 4 by parallel

## Adequacy

Bartlett chi2 = 8922.6, df = 1128, p = 0; KMO = 0.704

## Number of factors

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| observed eigenvalue | 14.23 | 13.77 | 4.39 | 2.88 | 2.33 | 1.66 | 1.46 | 1.35 |
| parallel-analysis threshold | 3.18 | 2.87 | 2.66 | 2.49 | 2.36 | 2.23 | 2.11 | 2.01 |

parallel analysis k = 4; Kaiser k = 9; MAP k = 16; **used k = 4 (parallel)**

## Variance

| factor | SS pattern | share of p | SS structure |
|---|---|---|---|
| F1 | 13.23 | 27.6% | 13.23 |
| F2 | 9.22 | 19.2% | 9.22 |
| F3 | 8.35 | 17.4% | 8.35 |
| F4 | 3.43 | 7.1% | 3.43 |

## Pattern loadings (|loading| >= 0.4)

- **F1**: L:function +0.95, L:Linguistic +0.93, L:ipron +0.91, L:focuspresent +0.89, L:det +0.88, L:auxverb +0.88, L:article +0.87, L:power -0.85, L:number -0.83, L:BigWords -0.73, L:Lifestyle -0.72, L:pronoun +0.72, L:adverb +0.71, L:work -0.69, L:prep +0.65, L:Drives -0.52, L:achieve +0.52, L:discrep +0.48, L:ppron +0.48, L:we +0.46, L:differ +0.46, L:adj +0.46, L:verb +0.41, L:affiliation +0.41
- **F2**: L:Clout +0.92, L:socrefs +0.87, L:Social +0.83, L:polite +0.83, L:affiliation +0.83, L:prosocial +0.80, L:we +0.79, L:you +0.71, L:i -0.62, L:certitude -0.61, L:Drives +0.61, L:adverb -0.48, L:quantity -0.45, L:BigWords +0.42, L:attention +0.42, L:focusfuture +0.41, L:adj -0.40
- **F3**: L:curiosity +0.86, L:verb +0.77, L:ppron +0.75, L:attention +0.66, L:i +0.62, L:pronoun +0.61, L:you +0.60, L:emo_pos +0.60, L:Tone +0.59, L:conj -0.58, L:insight +0.57, L:quantity -0.55, L:allure +0.55, L:work +0.52, L:differ -0.52, L:adj -0.51, L:Lifestyle +0.48, L:need +0.48, L:tone_pos +0.45, L:Drives +0.45, L:focusfuture +0.44
- **F4**: L:negate +0.89, L:allnone +0.74, L:allure +0.56, L:Analytic -0.48, L:adj -0.43, L:tone_pos -0.40

## Factor correlations

| | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| F1 | 1.00 | 0.00 | 0.00 | 0.00 |
| F2 | 0.00 | 1.00 | 0.00 | 0.00 |
| F3 | 0.00 | 0.00 | 1.00 | 0.00 |
| F4 | 0.00 | 0.00 | 0.00 | 1.00 |

communalities < .5: L:focusfuture 0.43, L:conj 0.45, L:insight 0.47, L:Analytic 0.26, L:need 0.45, L:discrep 0.37, L:assent 0.17, L:emo_pos 0.41, L:tone_pos 0.50, L:Authentic 0.32

## Models

- **F1** top: gpt-4o-2024-05-13 +3.17, llama-3.1-405b-instruct +2.87, llama-3.1-70b-instruct +2.64, llama-3.3-70b-instruct +2.57, deepseek-v2.5 +2.25
  bottom: gpt-5-mini-2025-08-07 -1.22, gpt-5.6-sol -1.39, minimax-m2.1 -1.42, gpt-5.1-codex -1.50, gpt-5-codex -1.86
- **F2** top: claude-3-5-sonnet-20240620 +4.33, gpt-4o-2024-05-13 +3.23, claude-3-5-sonnet-20241022 +1.67, gpt-oss-20b +1.57, gpt-oss-120b +1.34
  bottom: deepseek-v4-pro -1.23, minimax-m2.7 -1.28, claude-sonnet-4-6 -1.30, qwen3.7-plus -1.48, qwen3.6-27b -1.54
- **F3** top: deepseek-v3.2-exp +1.92, minimax-m2.1 +1.66, glm-4.6 +1.58, minimax-m2 +1.55, claude-sonnet-4-20250514 +1.50
  bottom: claude-opus-4-8 -1.46, kimi-k2.6 -1.58, gpt-5.5 -1.85, kimi-k3 -1.97, gpt-5.6-sol -2.22
- **F4** top: gemini-3-pro-preview +4.18, deepseek-v3.2 +3.49, gpt-oss-20b +2.82, claude-opus-4-5 +2.38, kimi-k2.5 +1.63
  bottom: qwen3.5-35b-a3b -1.21, claude-3-5-sonnet-20240620 -1.34, claude-opus-4-7 -1.34, minimax-m2.1 -1.50, gpt-5.6-sol -1.85
