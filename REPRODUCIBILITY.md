# Reproducibility Policy - CacheMedic++ Harness v1.1

This project targets fast iteration with reproducible artifacts and resumable execution.

Operational status and next commands are tracked in:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`

## Required run artifacts (per run directory)

A run directory must contain:

- `run_record.json` - canonical metadata (schema: `schemas/run_record.schema.json`)
- `config_resolved.yaml` - fully resolved config (includes defaults)
- `env.json` - Python, CUDA, GPU, library versions
- `git.json` - commit hash and dirty state
- `logs/train.jsonl` - step-wise log records (schema: `schemas/log_record.schema.json`)
- `logs/eval.jsonl` - eval summaries
- `logs/train_heartbeat.json` and `logs/train_heartbeat.jsonl`
- `logs/eval_heartbeat.json` and `logs/eval_heartbeat.jsonl`
- `logs/stability_heartbeat.json` and `logs/stability_heartbeat.jsonl`
- `metrics/task_metrics.json` - task robustness metrics
- `metrics/stability_metrics.json` - stability metrics
- `metrics/eval_progress.json` - resumable eval state
- `metrics/stability_progress.json` - resumable stability state

## Resume-state requirements

The harness must resume from saved state without recomputing completed units:

- Train resumes from `checkpoints/train_state_latest.pt` (or latest step snapshot).
- Eval resumes from `metrics/eval_progress.json`.
- Stability resumes from `metrics/stability_progress.json`.

Pause requests are signaled by `<RUN_DIR>/control/pause.request`.
Phase-specific pause acknowledgements are written to:

- `control/pause_ack.json` (train)
- `control/pause_ack_eval.json` (eval)
- `control/pause_ack_stability.json` (stability)

## Determinism requirements

- All stochasticity uses dedicated seeded generators.
- Orthogonal rotation matrices are generated deterministically on CPU then moved to GPU.
- v1 uses eager attention only; fused attention paths are disallowed.

## Seed and flag checklist

- Set: `PYTHONHASHSEED`, `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
- Torch flags:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- Log effective environment details in `env.json`.

## Manifest and integrity

- The repository ships `MANIFEST.sha256` for file-hash integrity.
- Manifest scope is source-of-record files only; runtime artifacts are excluded:
  - `empty_dirs_for_codex/outputs/`
  - `outputs/`
  - `.pytest_cache/`
- Use `scripts/verify_manifest.py` to verify hashes.

## What is allowed to vary

- Throughput and wall-clock timings may vary by machine load and driver/runtime details.
- Metric variance may exist due to nondeterministic GPU behavior outside strict harness controls.
- For final reporting, run multiple seeds when confidence intervals are required.
