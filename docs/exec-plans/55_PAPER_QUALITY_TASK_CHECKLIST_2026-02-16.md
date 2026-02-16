# Paper Quality Push Checklist (Tasks 1-8)

Purpose: track the remaining quality upgrades for the CacheMedic++ sweet-spot paper in a single, checkable file.

Status legend: `todo` | `in_progress` | `done` | `blocked`

## Master checklist

- [ ] `T1` Add 3-seed replication for `contr` vs `no_contr` (same config) and report `mean +- std`.
- [ ] `T2` Add bootstrap confidence intervals for robustness AUC and sensitivity ratio (can use existing run outputs if per-example/per-prompt data is available).
- [ ] `T3` Reserved (not specified in current request; keep slot for continuity).
- [ ] `T4` Add a mini hyper-sensitivity table for `lambda_contr` (target values: `0 / 3 / 5`).
- [ ] `T5` Add a compact `Claim -> Evidence` table in the main text (not appendix only).
- [ ] `T6` Add a short `Threats to validity` subsection with internal, external, and statistical threats.
- [ ] `T7` Smooth appendix layout (reduce whitespace, compact tables, compact code/pseudocode blocks).
- [ ] `T8` Optional: add one explicit negative result ("where it does not help") for reviewer trust.

## Execution tracker

| Task | Status | Owner | Last update | Deliverable path(s) |
|---|---|---|---|---|
| T1 | todo | codex | 2026-02-16 | `paper/sections/05_experiments.tex`, `paper/tables/*`, `outputs/runs/*` |
| T2 | todo | codex | 2026-02-16 | `paper/sections/04_metrics.tex`, `paper/sections/05_experiments.tex`, `paper/tables/*` |
| T3 | todo | codex | 2026-02-16 | reserved |
| T4 | todo | codex | 2026-02-16 | `paper/tables/*`, `paper/sections/05_experiments.tex` |
| T5 | todo | codex | 2026-02-16 | `paper/sections/01_introduction.tex`, `paper/sections/05_experiments.tex`, `paper/tables/*` |
| T6 | todo | codex | 2026-02-16 | `paper/sections/07_limitations_ethics.tex` |
| T7 | todo | codex | 2026-02-16 | `paper/appendix.tex`, `paper/tables/*` |
| T8 | todo | codex | 2026-02-16 | `paper/sections/05_experiments.tex` |

## Done criteria (quick gate)

- [ ] Main text contains all new claims with direct evidence references.
- [ ] New/updated tables compile cleanly in `paper/main.tex`.
- [ ] All numeric values in new sections are traceable to run artifacts or explicitly marked as pending runs.
- [ ] No placeholder wording remains.
