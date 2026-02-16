# Changelog - CacheMedic++ Harness

## v1.1.1 (in progress)
- Add: `cachemedicpp.watch` dashboard CLI for heartbeat monitoring (`status`, `progress`, `eta`, `throughput`, checkpoints).
- Add: `cachemedicpp.prefetch` CLI to pre-download model artifacts (default fast pair: `gpt2-medium` + `gpt2-large`).
- Add: pause/resume train control via `control/pause.request` and resumable train state checkpoints (`train_state_latest.pt` and step snapshots).
- Add: periodic heartbeat artifacts in training (`logs/train_heartbeat.json` + `logs/train_heartbeat.jsonl`).
- Add: finite-difference stability implementation with logit sensitivity and amplification outputs in `cachemedicpp.stability`.
- Add: GPU device fail-closed checks in `train`, `eval`, and `stability` (`device: cuda` now errors if CUDA is unavailable).
- Change: `configs/second_model_smoke.yaml` now tracks the fast GPT-2-family variant (`gpt2-large`).
- Add: regression tests for device fail-closed behavior in `empty_dirs_for_codex/tests/test_device_selection.py`.
- Run: completed MVE training run on RTX Pro 6000 at `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245`.
- Add: eval phase pause/resume with persisted progress state (`metrics/eval_progress.json`) and heartbeats (`logs/eval_heartbeat.*`).
- Add: stability phase pause/resume with persisted progress state (`metrics/stability_progress.json`) and heartbeats (`logs/stability_heartbeat.*`).
- Change: `cachemedicpp.watch` now auto-selects the latest phase heartbeat (`train/eval/stability/sweep`) for live tracking.
- Change: `cachemedicpp.sweep` now supports resume forwarding (`--resume`/`--no-resume`) and sweep-level pause requests.
- Add: sweet-spot paper configs:
  - `configs/fast_paper_base.yaml`
  - `configs/fast_paper_sweep.yaml`
  - `configs/fast_paper_sweep_core.yaml`
  - `configs/fast_second_model_gpt2_large.yaml`
- Add: central ops runbook for paper execution, status, tracking, and pause/resume:
  - `docs/exec-plans/40_SWEETSPOT_PAPER_OPERATIONS.md`
- Add: single-page sweet-spot status board with current run state, next commands, and pause/resume tracking:
  - `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`
- Change: manifest policy now excludes runtime artifacts (`empty_dirs_for_codex/outputs`, `outputs`, `.pytest_cache`) so checkpoints/logs do not fail smoke checks.

## v1.1.0
- Fix: removed accidental bytecode (`__pycache__` / `.pyc`) from shipped artifact.
- Fix: `MANIFEST.sha256` generation/verification now enforces completeness (no extra files).
- Fix: configs now validate against JSON schemas; added `scripts/validate_configs.py`.
- Fix: schema/config alignment for `overwrite_window`, `question_template`, and evaluation keys.
- Add: sweep runner is now a first-class CLI contract (`python -m cachemedicpp.sweep`).

## v1.0.0
- Initial harness release (specs + configs + paper skeleton).
