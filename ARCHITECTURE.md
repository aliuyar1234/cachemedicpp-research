# Architecture — CacheMedic++ Harness (v1.1)

This harness defines **interfaces, contracts, and artifacts** for implementing CacheMedic++.

Operational run status and pause/resume entrypoint:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`

## Design goals
- **Spec-first SSOT:** canonical definitions live in `docs/product-specs/`.
- **Fail-closed:** v1 assumes **bsz=1** and **eager attention**; any other mode must raise.
- **Determinism:** corruption masks and rotations must be reproducible from seeds.
- **Fast cycles:** most experiments run in **1–6 GPU-hours**; main matrix fits **≤72 GPU-hours**.

## System components
1. **HF Attention Patch Layer**
   - Intercepts `past_key_values` **after concat** and **before** attention logits.
   - Applies corruption operator `C` (when enabled) then repair operator `R_phi`.

2. **Corruption Engine**
   - Implements the family `C` with deterministic masking axes and mixture distribution `Π`.
   - Must support: gaussian noise, dropout/zeroing, orthogonal rotation, sparse bitflip-ish sign/jump, quantization noise, contiguous overwrite, targeted layer/head wrapper.

3. **Repair Operators (R_phi)**
   - Options A/B/C as specified in `docs/product-specs/20_REPAIR_OPERATORS.md`.
   - Trainable modules only; base model weights frozen.

4. **Trainer**
   - Runs self-distillation:
     - teacher logits under clean cache
     - student logits under corrupted+repaired cache
   - Loss: KD + identity + contraction (exact formulas in `30_TRAINING_OBJECTIVE.md`).

5. **Evaluators**
   - Task robustness: Wikitext-2 perplexity; SST-2 prompted classification; long-context needle check.
   - Baselines: inference-time heuristics; and ablation `lambda_contr = 0`.

6. **Stability Analyzer**
   - Produces logit sensitivity curves and amplification maps.

7. **Plotter**
   - Produces locked output figures with exact filenames expected by `paper/`.

8. **Sweep Runner**
   - Expands a sweep config into multiple runs (overrides applied on top of a base config).
   - Enforces total budget hints and writes a sweep summary.
   - Implements OOD protocol wiring for leave-one-type-out.

## Required CLI (contract)
Codex must implement these entrypoints in package `cachemedicpp`.

### Train
```bash
python -m cachemedicpp.train --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/<RUN_ID>
```
- Writes `run_record.json`, `config_resolved.yaml`, `env.json`, `git.json`.
- Writes checkpoints: `checkpoints/repair_step_<N>.pt` (repair params only).
- Writes resumable trainer state: `checkpoints/train_state_latest.pt` and periodic `checkpoints/train_state_step_<N>.pt`.
- Writes heartbeat artifacts: `logs/train_heartbeat.json` (latest snapshot) and `logs/train_heartbeat.jsonl` (timeline).
- Supports pause requests via `<RUN_DIR>/control/pause.request` and resume via rerunning train (default `--resume` behavior).

### Eval (task robustness)
```bash
python -m cachemedicpp.eval --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/<RUN_ID> [--resume|--no-resume]
```
- Writes `metrics/task_metrics.json` and per-eps curves.
- Appends eval summaries to `logs/eval.jsonl`.
- Writes phase heartbeat artifacts: `logs/eval_heartbeat.json` and `logs/eval_heartbeat.jsonl`.
- Writes resumable eval progress state: `metrics/eval_progress.json`.
- Supports pause requests via `<RUN_DIR>/control/pause.request` and resume via rerunning eval (default `--resume` behavior).

### Stability
```bash
python -m cachemedicpp.stability --config configs/mve.yaml --run_dir empty_dirs_for_codex/outputs/runs/<RUN_ID> [--resume|--no-resume]
```
- Writes `metrics/stability_metrics.json`.
- Writes phase heartbeat artifacts: `logs/stability_heartbeat.json` and `logs/stability_heartbeat.jsonl`.
- Writes resumable stability progress state: `metrics/stability_progress.json`.
- Supports pause requests via `<RUN_DIR>/control/pause.request` and resume via rerunning stability (default `--resume` behavior).

### Plots
```bash
python -m cachemedicpp.plots --run_dir empty_dirs_for_codex/outputs/runs/<RUN_ID>
```
- Writes figures under `paper/figures/` **inside the run directory** with locked names.

### Sweep
```bash
python -m cachemedicpp.sweep --sweep_config configs/full_gpt2_matrix.yaml --out_root empty_dirs_for_codex/outputs/runs [--resume|--no-resume]
```
- Validates the sweep config against `schemas/sweep.schema.json`.
- Loads the base config and applies dot-key overrides per run.
- For each run:
  - invokes train/eval/stability/plots in sequence
  - writes a per-run `run_record.json`
- Writes a sweep summary to `<out_root>/<sweep_name>_summary.json`.
- Supports sweep-level pause requests via `<out_root>/<sweep_name>/control/pause.request`.
- Resume-aware orchestration forwards resume behavior to train/eval/stability phases.

### Watch (monitoring)
```bash
python -m cachemedicpp.watch --run_dir empty_dirs_for_codex/outputs/runs/<RUN_ID> --follow --stop_on_terminal
```
- Reads latest available phase heartbeat (`train`, `eval`, `stability`, `sweep`) and prints a live dashboard.

### Prefetch (model artifacts)
```bash
python -m cachemedicpp.prefetch
```
- Pre-downloads model artifacts into the Hugging Face cache before runs.
- Default fast preset targets:
  - `configs/mve.yaml` (`gpt2-medium`)
  - `configs/second_model_smoke_gpt2_large.yaml` (`gpt2-large`)

## Output contracts (locked)
Within a run directory `<RUN_DIR>`:

- `paper/figures/figA_score_vs_eps.png`
- `paper/figures/figB_clean_vs_overhead_pareto.png`
- `paper/figures/figC_logit_sensitivity.png`
- `paper/figures/figD_amplification_map.png`

- `paper/tables/table_main_results.tex`
- `paper/tables/table_ablation_summary.tex`

- `logs/train.jsonl` (stepwise)
- `logs/eval.jsonl` (eval records)
- `logs/train_heartbeat.json` and `logs/train_heartbeat.jsonl`
- `logs/eval_heartbeat.json` and `logs/eval_heartbeat.jsonl`
- `logs/stability_heartbeat.json` and `logs/stability_heartbeat.jsonl`
- `metrics/task_metrics.json`
- `metrics/stability_metrics.json`
- `metrics/eval_progress.json`
- `metrics/stability_progress.json`
- `run_record.json`

## Batch size and attention implementation constraints (v1)
- **bsz=1 only**. Raise if `input_ids.shape[0] != 1`.
- **Eager attention only**. If model uses flash/fused attention, raise with a clear message.
- KV shapes are assumed to be `[1, heads, time, head_dim]` per layer.

## Naming conventions
- Package name: `cachemedicpp`
- Method name: **CacheMedic++**
- Repair operator: `R_phi`
- Corruption family: `C`
- Corruption mixture: `Π`
- Contraction target: `alpha_contr` and hinge penalty with `eps0`.
