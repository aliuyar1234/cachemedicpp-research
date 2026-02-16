# Sweet-Spot Status Board

Last updated: 2026-02-16

This is the top-level operations page. Canonical vs exploratory evidence is now defined in:

- `docs/exec-plans/45_CANONICAL_RUNS_2026-02-16.md`

## 1. Current pipeline status

| Run / group | Path | Train | Eval | Stability | Plots | Use for final paper |
|---|---|---|---|---|---|---|
| Primary tuned (contr) | `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/main_contr_vonly_tuned_loto_overwrite` | done | done | done | done | yes |
| Primary tuned (no_contr pair) | `empty_dirs_for_codex/outputs/runs_recovery/recovery_vonly_sweep_v1/ablation_no_contr_vonly_tuned_loto_overwrite` | done | done | done | done | yes |
| Second model tuned | `empty_dirs_for_codex/outputs/runs_recovery/recovery_second_model_gpt2_large_v1/second_model_gpt2_large_vonly_tuned` | done | done | done | done | yes |

## 2. Immediate next step

1. Freeze canonical run set (see `45_CANONICAL_RUNS_2026-02-16.md`).
2. Final paper checks:
   - run `scripts/smoke_repo.sh`
   - compile LaTeX paper
3. Prepare final result text from canonical metrics.

## 3. Exact commands (from current state)

Canonical recompute already completed for tuned contr with tuned no-contraction source.

## 4. Monitor

```powershell
python -m cachemedicpp.watch --run_dir <RUN_DIR> --follow --stop_on_terminal
```

## 5. Pause and resume

Pause:

```powershell
New-Item -ItemType File -Force <RUN_DIR>\control\pause.request
```

Resume:

- rerun the same command with `--resume`

## 6. Artifact checklist (final)

- `metrics/task_metrics.json`
- `metrics/stability_metrics.json`
- `paper/figures/figA_score_vs_eps.png`
- `paper/figures/figB_clean_vs_overhead_pareto.png`
- `paper/figures/figC_logit_sensitivity.png`
- `paper/figures/figD_amplification_map.png`
- `paper/tables/table_main_results.tex`
- `paper/tables/table_ablation_summary.tex`
- `run_record.json`
- `config_resolved.yaml`
- `env.json`
- `git.json`
