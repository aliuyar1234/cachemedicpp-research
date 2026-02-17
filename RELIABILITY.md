# Reliability and Fail-Closed Behavior - CacheMedic++ Research Repository

This repository treats invalid assumptions as failures and prioritizes resumable execution.

## Fail-closed invariants (must raise)

- `batch_size != 1`
- flash/fused attention active (must be eager)
- KV cache shape mismatch against `[1, heads, time, head_dim]`
- unknown corruption type outside canonical list
- config missing required keys or violating schema constraints
- `device: cuda` when CUDA is unavailable

## Operational reliability requirements

- Every long phase must expose heartbeat telemetry:
  - train: `train_heartbeat.*`
  - eval: `eval_heartbeat.*`
  - stability: `stability_heartbeat.*`
- Every long phase must support pause/resume:
  - pause via `<RUN_DIR>/control/pause.request`
  - resume from persisted state (train checkpoints, eval/stability progress JSON)
- `watch` must display the latest phase heartbeat without requiring manual phase selection.

## Numeric stability

- Repairs and corruptions run in model dtype while sensitive reductions can use fp32.
- Contraction ratio uses epsilon guards to avoid division-by-zero.
- Training uses gradient clipping.

## Cost controls

- Stability supports top-k logit projection and layer/head subsampling controls.
- Fast configs are provided to reduce end-to-end wall time while preserving key paper evidence:
  - `configs/fast_paper_base.yaml`
  - `configs/fast_paper_sweep_core.yaml`
  - `configs/fast_second_model_gpt2_large.yaml`

## Acceptance gate alignment

Acceptance gates are defined in `docs/product-specs/70_ACCEPTANCE_GATES.md`.
