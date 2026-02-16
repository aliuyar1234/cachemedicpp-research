# Invariants (v1.0)

These invariants must hold for v1 results.

## I1: Batch size and attention path
- Batch size is **1** for all core experiments.
- Attention implementation is **eager**; any fused/flash attention path is disallowed in v1.

## I2: Insertion point
Repair and corruption are applied **after** past KV concatenation and **before** attention logits. See `../product-specs/20_REPAIR_OPERATORS.md`.

## I3: Determinism
- All stochastic masks and random perturbations draw from a **dedicated torch.Generator** seeded from config.
- Orthogonal rotation matrices are generated on CPU with fixed seed and then reused.

## I4: Locked artifacts
Run directories must contain the locked figure filenames and metrics schemas as defined in `../product-specs/50_EVALUATION_PROTOCOL.md`.
