# Paper Quality Push Checklist (Tasks 1-8)

Purpose: track the remaining quality upgrades for the CacheMedic++ sweet-spot paper in a single, checkable file.

Status legend: `todo` | `in_progress` | `done` | `blocked`

## Master checklist

- [x] `T1` Add 3-seed replication for `contr` vs `no_contr` (same config) and report `mean +- std`.
- [x] `T2` Add bootstrap confidence intervals for robustness AUC and sensitivity ratio (seed-paired bootstrap, `B=20,000`).
- [ ] `T3` Reserved (not specified in current request; keep slot for continuity).
- [x] `T4` Add a mini hyper-sensitivity table for `lambda_contr` (target values: `0 / 3 / 5`).
- [x] `T5` Add a compact `Claim -> Evidence` table in the main text (not appendix only).
- [x] `T6` Add a short `Threats to validity` subsection with internal, external, and statistical threats.
- [x] `T7` Smooth appendix layout (reduce whitespace, compact tables, compact code/pseudocode blocks).
- [x] `T8` Optional: add one explicit negative result ("where it does not help") for reviewer trust.

## Execution tracker

| Task | Status | Owner | Last update | Deliverable path(s) |
|---|---|---|---|---|
| T1 | done | codex | 2026-02-16 | `paper/sections/05_experiments.tex`, `paper/tables/table_seed_replication_3seed.tex`, `configs/recovery_vonly_seedrep_t1.yaml`, `docs/exec-plans/56_T1_SEED_REPLICATION_RESULTS_2026-02-16.md` |
| T2 | done | codex | 2026-02-16 | `paper/sections/04_metrics.tex`, `paper/sections/05_experiments.tex`, `paper/tables/table_seed_bootstrap_ci.tex`, `docs/exec-plans/57_T2_BOOTSTRAP_CI_RESULTS_2026-02-16.md` |
| T3 | todo | codex | 2026-02-16 | reserved |
| T4 | done | codex | 2026-02-16 | `configs/recovery_vonly_lambda5_tuned_single.yaml`, `paper/tables/table_lambda_sensitivity.tex`, `paper/sections/05_experiments.tex`, `docs/exec-plans/58_T4_LAMBDA_SENSITIVITY_RESULTS_2026-02-16.md` |
| T5 | done | codex | 2026-02-16 | `paper/sections/05_experiments.tex`, `paper/tables/table_claim_evidence_compact.tex` |
| T6 | done | codex | 2026-02-16 | `paper/sections/07_limitations_ethics.tex` |
| T7 | done | codex | 2026-02-16 | `paper/appendix.tex`, `paper/tables/table_per_task_breakdown_medium.tex` |
| T8 | done | codex | 2026-02-16 | `paper/sections/05_experiments.tex`, `paper/tables/table_lambda_sensitivity.tex` |

## Done criteria (quick gate)

- [x] Main text contains all new claims with direct evidence references.
- [x] New/updated tables compile cleanly in `paper/main.tex`.
- [x] All numeric values in new sections are traceable to run artifacts or explicitly marked as pending runs.
- [x] No placeholder wording remains.
