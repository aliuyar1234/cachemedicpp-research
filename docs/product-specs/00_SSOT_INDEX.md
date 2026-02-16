# SSOT Index — CacheMedic++ (v1.1)

This directory is the **single source of truth (SSOT)** for CacheMedic++ definitions. If any other file disagrees, this directory wins.

## Canonical specs
1. `10_THREAT_MODEL.md` — corruption family C and deterministic axes
2. `20_REPAIR_OPERATORS.md` — insertion point and operator families A/B/C
3. `30_TRAINING_OBJECTIVE.md` — losses and default hyperparameters
4. `40_STABILITY_METRICS.md` — stability metrics and required plots
5. `50_EVALUATION_PROTOCOL.md` — tasks, subsets, offline fallbacks, and output contracts
6. `60_BASELINES_ABLATIONS.md` — required baselines and ablation matrix
7. `70_ACCEPTANCE_GATES.md` — pass criteria and pivot triggers

## Canonical names (must match code, configs, docs, paper)
- Project: **CacheMedic++**
- Package: `cachemedicpp`
- Corruption family: `C`
- Corruption mixture: `Pi`
- Repair operator: `R_phi`
- Default operator: **Option A** (query-conditioned low-rank additive correction; layer-shared across heads)

## Locked artifacts (paper expects these exact filenames)
In each run directory, the plotter must write:
- `paper/figures/figA_score_vs_eps.png`
- `paper/figures/figB_clean_vs_overhead_pareto.png`
- `paper/figures/figC_logit_sensitivity.png`
- `paper/figures/figD_amplification_map.png`

and tables:
- `paper/tables/table_main_results.tex`
- `paper/tables/table_ablation_summary.tex`

## Schemas
- Config schema (JSON form of YAML): `schemas/config.schema.json`
- Sweep schema: `schemas/sweep.schema.json`
- Run record schema: `schemas/run_record.schema.json`
- Log record schema: `schemas/log_record.schema.json`
- Metric record schema: `schemas/metric_record.schema.json`

## Operations references
- Sweet-spot run status board (first stop before resuming):
  - `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`
- Sweet-spot paper operations and run status tracking:
  - `docs/exec-plans/40_SWEETSPOT_PAPER_OPERATIONS.md`
- Session-level handoff snapshot:
  - `docs/exec-plans/30_PROGRESS_LOG_2026-02-15.md`
