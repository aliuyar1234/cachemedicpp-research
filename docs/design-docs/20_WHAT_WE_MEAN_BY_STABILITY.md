# What we mean by “stability”

CacheMedic++ treats transformer decoding as a dynamical system over persistent state (the KV cache). In this framing, **stability** means:

1. **State contraction toward the clean reference:** For a clean state `S` and a corrupted state `S_corr = C(S)`, the repaired state `S_rep = R_phi(S_corr)` should be **closer to S** than `S_corr` is.
2. **Reduced state-to-output gain:** Small state perturbations should cause **smaller drift in next-token logits** after repair.
3. **Preserved clean behavior:** When no corruption is applied, `R_phi` should be near-identity (low clean regression).

The harness enforces stability in two complementary ways:

- **Training-time stability:** a contraction penalty computed directly in KV space (see `../product-specs/30_TRAINING_OBJECTIVE.md`).
- **Evaluation-time stability:** finite-difference measurements of logit sensitivity and per-layer/head amplification (see `../product-specs/40_STABILITY_METRICS.md`).

Stability is not defined as “smaller cache norm” or “less memory.” This project explicitly avoids compression/eviction/clustering and does not optimize cache size.
