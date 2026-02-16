# CLI Contracts (v1.1)

For current run status and next command, check:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`

Codex must implement the following commands exactly.

## Train
`python -m cachemedicpp.train --config <YAML> --run_dir <DIR>`

Required behaviors:
- Resolve config defaults and write `<DIR>/config_resolved.yaml`.
- Write `<DIR>/run_record.json` and environment captures.
- Write `<DIR>/logs/train.jsonl` with schema `schemas/log_record.schema.json`.
- Save repair-only checkpoints to `<DIR>/checkpoints/repair_step_<N>.pt`.
- Save resumable state checkpoints to `<DIR>/checkpoints/train_state_latest.pt` and periodic `train_state_step_<N>.pt`.
- Emit heartbeat snapshots to `<DIR>/logs/train_heartbeat.json` and timeline records to `<DIR>/logs/train_heartbeat.jsonl`.
- Respect pause requests from `<DIR>/control/pause.request` by checkpointing and exiting with a paused status.

## Eval
`python -m cachemedicpp.eval --config <YAML> --run_dir <DIR>`

Required behaviors:
- Evaluate tasks and corruption grid.
- Write `<DIR>/metrics/task_metrics.json`.
- Append eval summaries to `<DIR>/logs/eval.jsonl`.
- Emit heartbeat snapshots to `<DIR>/logs/eval_heartbeat.json` and timeline records to `<DIR>/logs/eval_heartbeat.jsonl`.
- Save resumable eval progress to `<DIR>/metrics/eval_progress.json`.
- Respect pause requests from `<DIR>/control/pause.request` by writing `pause_ack_eval.json` and exiting with paused status.
- Support resume control flags:
  - `--resume` (default): continue from `eval_progress.json` when available.
  - `--no-resume`: discard progress state and recompute from scratch.

## Stability
`python -m cachemedicpp.stability --config <YAML> --run_dir <DIR>`

Required behaviors:
- Compute logit sensitivity curves and amplification maps.
- Write `<DIR>/metrics/stability_metrics.json`.
- Emit heartbeat snapshots to `<DIR>/logs/stability_heartbeat.json` and timeline records to `<DIR>/logs/stability_heartbeat.jsonl`.
- Save resumable stability progress to `<DIR>/metrics/stability_progress.json`.
- Respect pause requests from `<DIR>/control/pause.request` by writing `pause_ack_stability.json` and exiting with paused status.
- Support resume control flags:
  - `--resume` (default): continue from `stability_progress.json` when available.
  - `--no-resume`: discard progress state and recompute from scratch.

## Plots
`python -m cachemedicpp.plots --run_dir <DIR>`

Required behaviors:
- Read metrics JSON and write locked filenames under `<DIR>/paper/figures/` and `<DIR>/paper/tables/`.

## Sweep
`python -m cachemedicpp.sweep --sweep_config <YAML> --out_root <DIR>`

Required behaviors:
- Validate sweep config against `schemas/sweep.schema.json`.
- Apply run overrides (dot-keys) over the base config.
- Support `--resume` / `--no-resume` for phase-level resume forwarding.
- Respect sweep-level pause requests from `<DIR>/<sweep_name>/control/pause.request`.
- Implement OOD leave-one-type-out behavior when `eval.ood_protocol` is set:
  - load `configs/corruption/ood_splits.yaml` (or `eval.ood_splits_path`)
  - remove held-out type from training mixture for that run
  - evaluate on held-out type.

## Watch (monitoring helper)
`python -m cachemedicpp.watch --run_dir <DIR> [--follow] [--stop_on_terminal]`

Required behaviors:
- Read latest heartbeat state across phase-specific heartbeat files:
  - `train_heartbeat.*`
  - `eval_heartbeat.*`
  - `stability_heartbeat.*`
  - `sweep_heartbeat.*` (if present)
- Print step progress, status, ETA, throughput, and latest checkpoint/state paths.
- In follow mode, refresh on heartbeat changes until stopped (or until terminal state when `--stop_on_terminal` is set).

## Prefetch (model artifacts helper)
`python -m cachemedicpp.prefetch [--config <YAML>]... [--model <REPO[@REV]>]...`

Required behaviors:
- Resolve model targets from provided config(s) and/or explicit model spec(s).
- If no targets are passed, use the fast preset (`configs/mve.yaml` + `configs/second_model_smoke_gpt2_large.yaml`).
- Download model artifacts to local Hugging Face cache so run startup does not depend on live network fetches.
