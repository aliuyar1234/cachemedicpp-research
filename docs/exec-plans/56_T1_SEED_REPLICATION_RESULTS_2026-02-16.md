# T1 Result: 3-Seed Replication (contr vs no_contr)

Date: 2026-02-16  
Scope: paired tuned V-only setup (`lambda_contr=3.0` vs `lambda_contr=0.0`) with identical config and seeds.

## Run set

- Seed 1234 (existing canonical pair):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite`
- Seed 2026 (new):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_seedrep_t1/main_contr_vonly_tuned_loto_overwrite_seed2026`
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_seedrep_t1/ablation_no_contr_vonly_tuned_loto_overwrite_seed2026`
- Seed 2027 (new):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_seedrep_t1/main_contr_vonly_tuned_loto_overwrite_seed2027`
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_seedrep_t1/ablation_no_contr_vonly_tuned_loto_overwrite_seed2027`

## Per-seed key metrics

| Seed | AUC contr | AUC no_contr | AUC gain | Ratio sens d=1 | Ratio sens d=2 |
|---|---:|---:|---:|---:|---:|
| 1234 | 0.040096 | 0.038817 | +0.001279 | 0.9291 | 0.9266 |
| 2026 | 0.039455 | 0.039240 | +0.000215 | 1.0049 | 1.0279 |
| 2027 | 0.039083 | 0.040149 | -0.001067 | 1.2412 | 1.3132 |

## 3-seed aggregate (mean +- std)

- Robustness AUC (contr): `0.0395 +- 0.0005`
- Robustness AUC (no_contr): `0.0394 +- 0.0007`
- AUC gain (contr - no_contr): `+0.0001 +- 0.0012`
- Sensitivity ratio contr/no_contr at `delta=1.0`: `1.0584 +- 0.1628`
- Sensitivity ratio contr/no_contr at `delta=2.0`: `1.0892 +- 0.2004`

## Interpretation

- The canonical single-seed paired result remains positive.
- Across 3 seeds, effect direction is not consistent and variance is high.
- This is now explicitly reported in the main paper table (`tab:seedrep`) as a limitation.
