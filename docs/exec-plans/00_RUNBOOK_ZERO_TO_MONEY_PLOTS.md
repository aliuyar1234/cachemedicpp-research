# Exec Plan: Zero to Money Plots (v1.0)

This plan is written for implementation bootstrapping.

Operational note: for current paper execution, pause/resume workflows, and status tracking, use:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`
- `docs/exec-plans/40_SWEETSPOT_PAPER_OPERATIONS.md`

## Phase 1: Hooking and corruption engine

1. Implement HF eager-attention interception for GPT-2-family models.
2. Enforce invariants from `../design-docs/10_INVARIANTS.md`.
3. Insert after KV concat and before attention logits (`../product-specs/20_REPAIR_OPERATORS.md`).
4. Implement corruption engine as specified in `../product-specs/10_THREAT_MODEL.md`.
5. Implement mixture sampling from `configs/corruption/default_mix.yaml`.
6. Log sampled corruption events to `logs/train.jsonl`.
7. Validate determinism with repeated same-seed prompt runs.

## Phase 2: Repair operator and training objective

1. Implement default Option A repair operator (shared across heads).
2. Implement objective from `../product-specs/30_TRAINING_OBJECTIVE.md`.
3. Train with `configs/mve.yaml`.
4. Verify `loss_kd` decreases and `rho_mean` trends down.

## Phase 3: Evaluate and plot

1. Implement evaluation protocol from `../product-specs/50_EVALUATION_PROTOCOL.md`.
2. Implement stability metrics from `../product-specs/40_STABILITY_METRICS.md`.
3. Implement plotter writing locked filenames under `<RUN_DIR>/paper/figures/`.

## Exit criteria

- Gate 1 in `../product-specs/70_ACCEPTANCE_GATES.md` passes.
