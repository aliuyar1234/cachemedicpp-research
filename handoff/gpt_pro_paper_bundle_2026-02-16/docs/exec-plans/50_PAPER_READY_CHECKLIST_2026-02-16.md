# Paper-Ready Checklist (2026-02-16)

## 1. Canonical evidence set locked

- Primary (gpt2-medium, tuned V-only contr):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite`
- Paired ablation (gpt2-medium, tuned V-only no_contr):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite`
- Second model (gpt2-large):
  - `empty_dirs_for_codex/outputs/runs_recovery/recovery_second_model_gpt2_large_v1/second_model_gpt2_large_vonly_tuned`

Status: PASS

## 2. Gate-style summary

- Gate 1 (main robustness + clean constraints):
  - `table_main_results.tex`: CacheMedic++ clean score `0.2466` vs no defense `0.2373`.
  - `table_main_results.tex`: CacheMedic++ robustness AUC `0.0401` vs best heuristic `0.0390`.
  - Clean WT2 regression from `metrics/task_metrics.json` (main tuned run):
    - no defense PPL `35.7299985`
    - cachemedic PPL `35.8310228`
    - relative increase `+0.28%`
  - Status: PASS

- Gate 2 (contraction helps vs no_contr):
  - Robustness AUC:
    - contr `0.0400962`
    - no_contr `0.0388174`
    - gain `+0.0012788`
  - Stability ratios (contr / no_contr), from `metrics/stability_metrics.json`:
    - delta `1.0`: `0.9291`
    - delta `2.0`: `0.9266`
  - Status: PASS

- Gate 3 (second-model qualitative confirmation):
  - `table_second_model_results.tex`:
    - clean score `0.3001` vs `0.2974`
    - robustness AUC `0.0475` vs `0.0472`
  - Status: PASS

## 3. Paper asset contract

- Figures present:
  - `paper/figures/figA_score_vs_eps.png`
  - `paper/figures/figB_clean_vs_overhead_pareto.png`
  - `paper/figures/figC_logit_sensitivity.png`
  - `paper/figures/figD_amplification_map.png`
- Tables present:
  - `paper/tables/table_main_results.tex`
  - `paper/tables/table_ablation_summary.tex`
  - `paper/tables/table_second_model_results.tex`

Status: PASS

## 4. Repo hygiene

- `bash scripts/smoke_repo.sh`: PASS (junk/placeholders/links/schema/manifest all pass)
- Manifest regenerated after final canonical updates:
  - `MANIFEST.sha256`

Status: PASS

## 5. Remaining caveat to report explicitly in paper text

- Absolute stability vs baseline remains higher than baseline in current metric scale; paper claims should focus on:
  - improvement vs no_contr (paired ablation),
  - robustness and clean tradeoff wins,
  - second-model qualitative transfer.

Status: ACCEPTED (document as limitation, not a blocker for the Sweetspot paper framing)
