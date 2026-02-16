# Canonical Runs and Evidence Split (2026-02-16)

This page removes ambiguity about which runs are paper-claim evidence.

## 1. Rule

- `canonical`: post-fix runs used for tables/figures and claims.
- `exploratory`: diagnosis/tuning runs kept for audit trail only.

## 2. Canonical (primary model, gpt2-medium)

- `main_contr_vonly_tuned_loto_overwrite`
  - Path: `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
  - Purpose: primary CacheMedic++ tuned run (`A`, `V-only`, `lambda_contr=3.0`).
- `ablation_no_contr_vonly_tuned_loto_overwrite`
  - Path: `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite`
  - Purpose: exact no-contraction ablation pair (`lambda_contr=0.0`) for Gate-2 comparison.

Use only this pair for final `contr vs no_contr` claims.

## 3. Canonical (second model, gpt2-large)

- `second_model_gpt2_large_vonly_tuned`
  - Path: `empty_dirs_for_codex/outputs/runs_recovery/recovery_second_model_gpt2_large_v1/second_model_gpt2_large_vonly_tuned`
  - Purpose: second-model qualitative confirmation (Gate 3).

## 4. Exploratory (do not use for final claims)

- `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245`
- `empty_dirs_for_codex/outputs/runs_fast_paper_core/fast_paper_sweep_core_v1/main_contr_loto_holdout_overwrite`
- `empty_dirs_for_codex/outputs/runs_fast_paper_core/fast_paper_sweep_core_v1/ablation_no_contr_loto_holdout_overwrite`
- `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_aggressive_loto_overwrite`
- `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_aggressive_loto_overwrite`
- `empty_dirs_for_codex/outputs/runs_recovery/recovery_optionb_sweep_v1/main_contr_optionb_loto_overwrite`
- `empty_dirs_for_codex/outputs/runs_recovery/recovery_optionb_sweep_v1/ablation_no_contr_optionb_loto_overwrite`

Reason: these runs were pre-fix for key metric logic or served as parameter search, not final claim runs.

## 5. Final asset source mapping

- Paper main figures/tables should come from:
  - primary model: `main_contr_vonly_tuned_loto_overwrite`
  - second model result table: `second_model_gpt2_large_vonly_tuned`

Locked filenames expected by LaTeX:

- `paper/figures/figA_score_vs_eps.png`
- `paper/figures/figB_clean_vs_overhead_pareto.png`
- `paper/figures/figC_logit_sensitivity.png`
- `paper/figures/figD_amplification_map.png`
- `paper/tables/table_main_results.tex`
- `paper/tables/table_ablation_summary.tex`

## 6. Remaining blocker

- The only remaining run to finish before freeze is:
  - none

Freeze-ready note:

- `main_contr_vonly_tuned_loto_overwrite` stability was recomputed after the tuned no-contraction pair completed.
- Current paired stability ratios:
  - `delta=1.0`: `cachemedic / no_contr = 0.9291`
  - `delta=2.0`: `cachemedic / no_contr = 0.9266`
