# Baselines and Ablations — v1.0

This spec defines the required baselines and the minimum ablations reviewers will expect.

## 1. Baselines (required)
### 1.1 No defense
- Run with corruption active and no repair operator.

### 1.2 Heuristic inference-time defenses
Heuristic defenses modify KV at inference without training. Implement all four.

1) **Reset/Clear**
- Policy: for each protected layer, set `(K,V)=0` for corrupted timesteps under the time mask.
- When used with `old_only`, this clears only old cache.

2) **Smoothing**
- Policy: along time dimension, apply exponential smoothing on the corrupted region:
  - `X[:, :, t, :] <- (1 - beta) * X[:, :, t, :] + beta * X[:, :, t-1, :]`
- Use `beta` from config (`baseline.smoothing_beta`), default `0.3`.

3) **Masking/Dropout**
- Policy: apply dropout-style zeroing with probability `baseline.dropout_p` on masked entries.

4) **Clipping/Renorm**
- Policy: clamp value range and renormalize RMS:
  - clamp: `X <- clamp(X, -clip_val, clip_val)`
  - renorm: `X <- X * (target_rms / (rms(X)+eps0))`

### 1.3 Ablation baseline: repair without contraction
- Train the same repair operator but set `lambda_contr = 0.0`.
- This ablation is required to justify the stability framing.

## 2. Minimum ablations (required)
1. `lambda_contr` sweep including 0.0.
2. K-only vs V-only vs KV.
3. Protected layer subset sweep (k in {2,4,8}, early/mid/late).
4. Operator family comparison (A vs B vs C) at matched parameter budget.
5. OOD corruption protocol: leave-one-type-out.

## 3. Reporting requirements
- Report best heuristic baseline chosen by a fixed rule: select the heuristic with best mean robustness curve area-under-curve on the MVE epsilon grid.
- Report clean-regression for each defense at epsilon=0.
