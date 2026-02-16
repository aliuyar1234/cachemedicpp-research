# CacheMedic++ Harness (v1.1) — Agent Map

This repository is the **system-of-record harness** for implementing and validating **CacheMedic++** end-to-end (code → experiments → plots → paper) on a **single RTX Pro 6000**. It is intentionally **spec-first**: the canonical truth is in `docs/product-specs/` and `schemas/`.

## Where truth lives (read in this order)
1. **Canonical spec index (SSOT):** `docs/product-specs/00_SSOT_INDEX.md`
2. **Threat/fault model (deterministic corruption family):** `docs/product-specs/10_THREAT_MODEL.md`
3. **Operator families + insertion point:** `docs/product-specs/20_REPAIR_OPERATORS.md`
4. **Training objective (KD + identity + contraction):** `docs/product-specs/30_TRAINING_OBJECTIVE.md`
5. **Stability metrics (logit sensitivity + amplification maps):** `docs/product-specs/40_STABILITY_METRICS.md`
6. **Evaluation protocol + subsets + offline fallback:** `docs/product-specs/50_EVALUATION_PROTOCOL.md`
7. **Baselines + ablations:** `docs/product-specs/60_BASELINES_ABLATIONS.md`
8. **Acceptance gates (pass criteria + pivot triggers):** `docs/product-specs/70_ACCEPTANCE_GATES.md`

## Repo layout (what to implement vs what already works)
- `empty_dirs_for_codex/src/` and `empty_dirs_for_codex/tests/` are **contracts** for the future implementation.
- `configs/` contains **ready-to-run YAMLs** for MVE, full matrix sweep, and second-model smoke.
- `paper/` is a LaTeX skeleton that **expects specific figure/table filenames**.
- `scripts/` contains **repo hygiene** checks that run today (no ML code needed).

## Expected CLI (Codex must implement)
All commands and output contracts are defined in `ARCHITECTURE.md` and repeated in `docs/exec-plans/10_CLI_CONTRACTS.md`.

Minimum CLI surface:
- `python -m cachemedicpp.train --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.eval  --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.stability --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.plots --run_dir <DIR>`
- `python -m cachemedicpp.sweep --sweep_config <YAML> --out_root <DIR>`

## What “done” means (high-level)
- **Money plots produced** with locked filenames (see `paper/` and `docs/product-specs/40_STABILITY_METRICS.md`).
- **OOD generalization protocol** executed (leave-one-type-out).
- **Second-model confirmation** executed with the same operator family.
- **Reproducibility artifacts**: run record JSON, config snapshot, git commit hash, environment capture.

## Hygiene checks (run now)
Run from repo root:
```bash
bash scripts/smoke_repo.sh
```
This enforces:
- no junk artifacts (no `__pycache__`, no `.pyc`)
- no placeholder markers
- internal link integrity for markdown
- YAML configs validate against JSON schemas
- manifest completeness + hash matching

## First implementation target (fastest path)
1. Implement eager-attention hooks for GPT-2 (`attn_implementation=eager`, bsz=1 fail-closed).
2. Implement corruption family operators exactly as specified in `docs/product-specs/10_THREAT_MODEL.md`.
3. Implement Option A repair operator (default) and training loss (KD + identity + contraction).
4. Produce the MVE figures (A/B/C) and stability stats with `configs/mve.yaml`.
