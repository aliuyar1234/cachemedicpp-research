# Sweet-Spot Paper Operations (v1.1)

This file is the operational source for running the **paper-relevant sweet-spot pipeline** with
reliable **monitoring**, **pause/resume**, and **resume-from-progress** behavior.

For the current at-a-glance run status, open:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`
- `docs/exec-plans/45_CANONICAL_RUNS_2026-02-16.md`
- `docs/exec-plans/30_PROGRESS_LOG_2026-02-16.md`

## 1. Scope and objective

The sweet-spot target minimizes runtime while preserving reviewer-relevant evidence:

1. Robustness curves (no defense vs best heuristic vs CacheMedic++).
2. Stability curves (with contraction vs no-contraction ablation signal).
3. OOD check (single LOTO holdout: `contiguous_overwrite`).
4. Second-model confirmation (`gpt2-large`).

The sweet-spot run set uses:

- `configs/fast_paper_sweep_core.yaml`
- `configs/fast_second_model_gpt2_large.yaml`

## 2. Current state snapshot (as of 2026-02-15)

### Already achieved

- `empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245`:
  - training completed (`12000/12000`)
  - final checkpoints exist (`repair_step_12000.pt`, `train_state_step_12000.pt`)

### Pending for paper

- On the existing MVE run:
  - `eval`
  - `stability`
  - `plots`
- Sweet-spot core sweep:
  - `main_contr_loto_holdout_overwrite`
  - `ablation_no_contr_loto_holdout_overwrite`
- Fast second-model run (`gpt2-large`):
  - `train`, `eval`, `stability`, `plots`

## 3. Single-run operational model

Each run directory is phase-driven:

1. `train`
2. `eval`
3. `stability`
4. `plots`

`train`, `eval`, and `stability` are now resumable and pauseable.

## 4. Monitoring (authoritative)

Use the watch CLI for live status:

```powershell
python -m cachemedicpp.watch --run_dir <RUN_DIR> --follow --stop_on_terminal
```

What `watch` reads (latest timestamp wins):

- `logs/train_heartbeat.json` or `.jsonl`
- `logs/eval_heartbeat.json` or `.jsonl`
- `logs/stability_heartbeat.json` or `.jsonl`
- `logs/sweep_heartbeat.json` or `.jsonl` (reserved for future expansion)

Heartbeats expose:

- `phase`, `status`, `step/total_steps`, `progress`, `eta_sec`, `reason`, `resumed_from`
- optional `current_unit` for eval/stability

## 5. Pause and resume behavior

### 5.1 Pause signal (same for all run phases)

Create pause request:

```powershell
New-Item -ItemType File -Force <RUN_DIR>\control\pause.request
```

### 5.2 Train pause/resume

- Pause ack: `<RUN_DIR>/control/pause_ack.json`
- Resume state:
  - `<RUN_DIR>/checkpoints/train_state_latest.pt`
  - `<RUN_DIR>/checkpoints/train_state_step_<N>.pt`
- Resume command:

```powershell
python -m cachemedicpp.train --config <YAML> --run_dir <RUN_DIR> --resume
```

### 5.3 Eval pause/resume

- Pause ack: `<RUN_DIR>/control/pause_ack_eval.json`
- Resume state: `<RUN_DIR>/metrics/eval_progress.json`
- Resume command:

```powershell
python -m cachemedicpp.eval --config <YAML> --run_dir <RUN_DIR> --resume
```

If `--no-resume` is passed, eval starts from scratch.

### 5.4 Stability pause/resume

- Pause ack: `<RUN_DIR>/control/pause_ack_stability.json`
- Resume state: `<RUN_DIR>/metrics/stability_progress.json`
- Resume command:

```powershell
python -m cachemedicpp.stability --config <YAML> --run_dir <RUN_DIR> --resume
```

If `--no-resume` is passed, stability starts from scratch.

### 5.5 Sweep pause/resume

Pause a sweep by creating:

```powershell
New-Item -ItemType File -Force <OUT_ROOT>\<SWEEP_NAME>\control\pause.request
```

Resume with the same sweep command:

```powershell
python -m cachemedicpp.sweep --sweep_config <SWEEP_YAML> --out_root <OUT_ROOT> --resume
```

Sweep forwards resume behavior to train/eval/stability.

## 6. Sweet-spot run commands

Set local model cache once:

```powershell
$env:HF_HOME='E:\Model\huggingface'
$env:HF_HUB_CACHE='E:\Model\huggingface\hub'
$env:HUGGINGFACE_HUB_CACHE='E:\Model\huggingface\hub'
```

### 6.1 Finish paper artifacts from existing MVE run

```powershell
python -m cachemedicpp.eval --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245 --resume
python -m cachemedicpp.stability --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245 --resume
python -m cachemedicpp.plots --run_dir empty_dirs_for_codex/outputs/runs/mve_gpu_20260215_121245
```

### 6.2 Run sweet-spot core sweep

```powershell
python -m cachemedicpp.sweep --sweep_config configs/fast_paper_sweep_core.yaml --out_root empty_dirs_for_codex/outputs/runs_fast_paper_core --resume
```

### 6.3 Run fast second-model confirmation

```powershell
python -m cachemedicpp.train --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.eval --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.stability --config configs/fast_second_model_gpt2_large.yaml --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large --resume
python -m cachemedicpp.plots --run_dir empty_dirs_for_codex/outputs/runs_fast_paper_core/second_model_gpt2_large
```

## 7. Paper-relevant artifacts checklist

For each paper run directory:

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

## 8. Quick troubleshooting

If process stops unexpectedly:

1. Check heartbeat files in `<RUN_DIR>/logs`.
2. Check resume state:
   - eval: `metrics/eval_progress.json`
   - stability: `metrics/stability_progress.json`
   - train: `checkpoints/train_state_latest.pt`
3. Rerun same command with `--resume`.

If CUDA warning appears but run progresses:

- This can be non-fatal on Windows for Triton utility checks.
- Trust heartbeat progress and output artifact growth.

## 9. Session handoff rule

At end of each session, update these files in order:

1. `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md` (authoritative status and next command)
2. `docs/exec-plans/30_PROGRESS_LOG_2026-02-15.md` (what changed in this session)
