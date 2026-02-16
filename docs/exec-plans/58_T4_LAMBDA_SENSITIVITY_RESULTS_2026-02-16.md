# T4 Result: Mini Hyper-Sensitivity (\lambda_contr = 0 / 3 / 5)

Date: 2026-02-16  
Scope: tuned V-only setup, seed 1234, same recipe; added an isolated `lambda_contr=5.0` run.

## Run paths

- `lambda=0`:
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite`
- `lambda=3`:
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
- `lambda=5`:
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_lambda5_tuned_single/main_contr_vonly_tuned_loto_overwrite_lambda5`

## Key metrics

| lambda_contr | clean score | robustness AUC | sensitivity d=1 | sensitivity d=2 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.2400 | 0.0388 | 52.4250 | 53.3308 |
| 3.0 | 0.2466 | 0.0401 | 48.7083 | 49.4163 |
| 5.0 | 0.2413 | 0.0390 | 54.5851 | 55.3688 |

## Interpretation

- `lambda=3.0` is best among tested values.
- Increasing to `lambda=5.0` hurts both robustness and stability in this setup.
- This is included as an explicit negative result in the main text.
