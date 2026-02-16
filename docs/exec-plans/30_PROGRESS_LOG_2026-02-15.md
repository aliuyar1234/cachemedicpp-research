# Progress Log - 2026-02-15

This file is the session handoff note. For full operations and commands, use:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`
- `docs/exec-plans/40_SWEETSPOT_PAPER_OPERATIONS.md`

## Session snapshot

- Date: 2026-02-15
- Repo root: `E:\Research\cachemedic_pp_harness_v1_1\cachemedic_pp_harness`
- Primary completed train run: `mve_gpu_20260215_121245`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Active queue at end of session: none (manually stopped)

## What was completed in this session

1. Added sweet-spot configs:
   - `configs/fast_paper_base.yaml`
   - `configs/fast_paper_sweep_core.yaml`
   - `configs/fast_paper_sweep.yaml`
   - `configs/fast_second_model_gpt2_large.yaml`
2. Added OOD all-holdout config (optional/non-sweet-spot):
   - `configs/ood_loto_all.yaml`
3. Added automatic local HF cache preference in runtime loader:
   - `empty_dirs_for_codex/src/cachemedicpp/runtime.py`
4. Added tests for runtime cache behavior:
   - `empty_dirs_for_codex/tests/test_runtime_cache.py`
5. Added phase-level pause/resume and heartbeat for eval:
   - progress state: `metrics/eval_progress.json`
   - heartbeat: `logs/eval_heartbeat.json` and `.jsonl`
   - pause ack: `control/pause_ack_eval.json`
6. Added phase-level pause/resume and heartbeat for stability:
   - progress state: `metrics/stability_progress.json`
   - heartbeat: `logs/stability_heartbeat.json` and `.jsonl`
   - pause ack: `control/pause_ack_stability.json`
7. Extended watch CLI to show latest phase heartbeat (`train/eval/stability/sweep`) instead of train-only.
8. Extended sweep CLI with resume forwarding and sweep-level pause request handling.
9. Updated docs and contracts to reflect sweet-spot operations and full pause/resume behavior.

## Paper-relevant run status

### Run: `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245`

- Train: completed
- Eval: pending
- Stability: pending
- Plots: pending
- Reusable for paper: yes (do not restart from zero)

### Sweet-spot core sweep (`configs/fast_paper_sweep_core.yaml`)

- Target runs:
  - `main_contr_loto_holdout_overwrite`
  - `ablation_no_contr_loto_holdout_overwrite`
- Status: not completed yet

### Fast second-model confirmation (`configs/fast_second_model_gpt2_large.yaml`)

- Status: not completed yet

## Important paths

- Existing completed train checkpoints:
  - `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245/checkpoints/repair_step_12000.pt`
  - `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245/checkpoints/train_state_step_12000.pt`
- Queue logs:
  - `empty_dirs_for_codex/outputs/queue_logs/`

## Next session exact tasks (sweet-spot only)

1. Resume and finish existing MVE eval/stability/plots:
```powershell
$env:HF_HOME='E:\Model\huggingface'
$env:HF_HUB_CACHE='E:\Model\huggingface\hub'
$env:HUGGINGFACE_HUB_CACHE='E:\Model\huggingface\hub'
python -m cachemedicpp.eval --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245 --resume
python -m cachemedicpp.stability --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245 --resume
python -m cachemedicpp.plots --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245
```

2. Run sweet-spot core sweep:
```powershell
python -m cachemedicpp.sweep --sweep_config configs/fast_paper_sweep_core.yaml --out_root empty_dirs_for_codex/outputs/runs_fast_paper_core --resume
```

3. Run fast second-model confirmation:
```powershell
python -m cachemedicpp.train --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.eval --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.stability --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.plots --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large
```

4. Track with watch:
```powershell
python -m cachemedicpp.watch --run_dir <RUN_DIR> --follow --stop_on_terminal
```

5. Pause any active phase:
```powershell
New-Item -ItemType File -Force <RUN_DIR>\control\pause.request
```

## Notes

- Triton CUDA utility warning on Windows may appear and is non-fatal in this setup.
- `run_record.json` can contain `git.commit: unknown` when workspace is not a git checkout.
