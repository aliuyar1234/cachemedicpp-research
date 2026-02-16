# Training Objective (KD + Identity + Contraction) — v1.0

CacheMedic++ trains only `phi` (repair params). Base model weights `theta` are frozen.

## 1. Teacher/student definitions
- Teacher: frozen model run with **clean** cache state.
- Student: same frozen model run with corruption `C` applied to the cache, followed by repair `R_phi`.

We distill next-token logits.

## 2. KD loss (canonical)
Let `z_clean` and `z_rep` be next-token logits (shape `[1, vocab]`). With temperature `T`:

`p = softmax(z_clean / T)`
`q = softmax(z_rep / T)`

Loss:
`L_KD = KL(p || q)`

Default: `T=2.0`.

## 3. Identity regularizer (canonical)
When corruption is disabled (`epsilon = 0` or `type = none`), penalize deviation from identity:

For each protected layer `l`:
- clean KV: `K_clean[l], V_clean[l]`
- repaired KV: `K_rep[l], V_rep[l]` (repair applied even in clean mode)

Define normalized squared error:

`L_id(l) = ( ||K_rep[l] - K_clean[l]||_F^2 + ||V_rep[l] - V_clean[l]||_F^2 ) / ( ||K_clean[l]||_F^2 + ||V_clean[l]||_F^2 + eps0 )`

`L_id = sum_{l in protect_layers} L_id(l)`

Default: `eps0 = 1e-8`, `lambda_id = 1.0`.

## 4. Contraction regularizer (canonical)
The contraction regularizer enforces that repair contracts corrupted cache toward the clean cache manifold.

For each protected layer `l`:
- corrupted KV: `K_corr[l], V_corr[l]`
- repaired KV: `K_rep[l], V_rep[l]`

Define layer error tensors:
- `DeltaK(l) = K_corr[l] - K_clean[l]`
- `DeltaV(l) = V_corr[l] - V_clean[l]`
- `DeltaK_rep(l) = K_rep[l] - K_clean[l]`
- `DeltaV_rep(l) = V_rep[l] - V_clean[l]`

Define contraction ratio:

`rho(l) = ( ||DeltaK_rep(l)||_F + ||DeltaV_rep(l)||_F ) / ( ||DeltaK(l)||_F + ||DeltaV(l)||_F + eps0 )`

Hinge penalty toward target `alpha_contr`:

`L_contr(l) = max(0, rho(l) - alpha_contr)^2`

`L_contr = sum_{l in protect_layers} L_contr(l)`

Defaults:
- `alpha_contr = 0.8`
- `lambda_contr = 0.5`

## 5. Total loss
`L = L_KD + lambda_id * L_id + lambda_contr * L_contr + lambda_w * ||phi||_2^2`

Default: `lambda_w = 0.0` in v1.

## 6. Minimal sweeps (reviewer-proof)
Required sweeps:
- `lambda_contr in {0.0, 0.1, 0.5, 1.0}` (demonstrate stability term matters)
- `rank in {4, 8, 16}`
- `protect_k_layers in {2, 4, 8}` and locations {early, mid, late}
- corruption OOD: leave-one-type-out

## 7. Logging requirements
During training, log:
- `loss_total`, `loss_kd`, `loss_id`, `loss_contr`
- `rho_mean`, `rho_p95` across protected layers
- sampled corruption type and severity

See schemas: `schemas/log_record.schema.json`.
