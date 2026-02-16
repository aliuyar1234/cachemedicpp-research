# Progress Log - 2026-02-16

This file records post-fix recovery progress and paper-canonical run selection.

## Session snapshot

- Date: 2026-02-16
- Repo root: `E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`

## Major fixes landed

1. `stability` no-contraction pairing fix:
   - `cachemedic_no_contr` now loads an actual paired ablation run when available.
   - no silent fallback to identical `cachemedic` curve for `lambda_contr > 0`.
2. `plots` robustness:
   - gracefully handles missing no-contraction curves.
3. `train/losses` contraction alignment:
   - contraction ratio now respects `repair.apply_to` (`K`, `V`, or `KV`).
   - `V-only` runs no longer dilute contraction with `K` terms.
4. `stability` instrumentation:
   - writes delta injection diagnostics (`delta_pair_norm`, relative norms).
   - writes `delta_injection_shared_across_variants` in stability settings.

## Key completed runs (post-fix)

- `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
  - completed (`train/eval/stability/plots`)
  - robust OOD and stability-vs-no_contr signal
- `empty_dirs_for_codex/outputs/runs_recovery/recovery_second_model_gpt2_large_v1/second_model_gpt2_large_vonly_tuned`
  - completed (`train/eval/stability/plots`)
  - second-model qualitative confirmation

## In-flight at handoff

- `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite`
  - completed (`train/eval/stability`)

## Finalized paper-facing numbers (canonical)

- Primary tuned contr run:
  - Path: `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
  - Robustness AUC (`cachemedic`): `0.04009623040160504`
  - Robustness AUC diff vs tuned no-contraction: `+0.00127879`
  - WT2 clean relative regression: `+0.2827%`
  - Stability ratios (`cachemedic / no_contr`):
    - `delta=1.0`: `0.9291`
    - `delta=2.0`: `0.9266`
- Second model tuned run:
  - Path: `empty_dirs_for_codex/outputs/runs_recovery/recovery_second_model_gpt2_large_v1/second_model_gpt2_large_vonly_tuned`
  - Robustness AUC (`cachemedic`): `0.04748586656120675`
  - Best heuristic AUC: `0.04720688551888036`
  - No-defense AUC: `0.04717004680487946`

## Evidence classification

- Canonical/evidence split is now defined in:
  - `docs/exec-plans/45_CANONICAL_RUNS_2026-02-16.md`

## Next actions

1. Finish `ablation_no_contr_vonly_tuned_loto_overwrite`.
2. Re-run stability for `main_contr_vonly_tuned_loto_overwrite` with `--no-resume`.
3. Re-run plots for `main_contr_vonly_tuned_loto_overwrite`.
4. Sync canonical figures/tables into `paper/figures` and `paper/tables`.
5. Run final hygiene and paper build steps.

## Session completion update

- `ablation_no_contr_vonly_tuned_loto_overwrite` completed (`train/eval/stability/plots`).
- `main_contr_vonly_tuned_loto_overwrite` stability was recomputed after pair completion.
- `main_contr_vonly_tuned_loto_overwrite` now references:
  - `cachemedic_no_contr_source = .../ablation_no_contr_vonly_tuned_loto_overwrite`
- Paper assets were synchronized from canonical tuned primary run into:
  - `paper/figures/figA_score_vs_eps.png`
  - `paper/figures/figB_clean_vs_overhead_pareto.png`
  - `paper/figures/figC_logit_sensitivity.png`
  - `paper/figures/figD_amplification_map.png`
  - `paper/tables/table_main_results.tex`
  - `paper/tables/table_ablation_summary.tex`
- Added second-model table:
  - `paper/tables/table_second_model_results.tex`
