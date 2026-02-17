# Repair / Stability Operators R_phi

This spec defines the insertion point and the three operator families for `R_phi`.

## 1. Insertion point (canonical)
For each attention layer:
1. Compute current-step `key_states`, `value_states`.
2. Concatenate into full cache `K_full, V_full` using `past_key_values`.
3. **Apply corruption** (optional): `(K_corr, V_corr) = C(K_full, V_full)`.
4. **Apply repair**: `(K_hat, V_hat) = R_phi(q, K_corr, V_corr)`.
5. Compute attention logits with `q` and `K_hat`.

### Current constraints
- bsz=1 only.
- eager attention path only.

## 2. Option A (default): query-conditioned low-rank additive correction
### Definition
For each protected layer, share parameters across heads by default.
Let `d` be head dim and `r` be rank.
Learn matrices:
- `Wk, Uk in R^{d x r}`
- `Wv, Uv in R^{d x r}`
and a gating MLP `g(q) -> alpha in (0,1)^r` per head.

Vectorized form (per head):
- `Ck = K @ Wk` giving shape `[1,H,T,r]`
- `dK = (Ck * alpha[:, :, None, :]) @ Uk^T`
- `K_hat = K + dK`
Similarly for `V`.

### Parameter count
Per protected layer (shared across heads):
- matrices: `4*d*r`
- gating MLP (two-layer): approximately `2*d*r + 2*r*r`
Total: `~ (6*d*r + 2*r^2)`.

### Compute complexity
For each protected layer, per token step:
- projections and backprojections cost `O(H*T*d*r)`.
Protecting `k` layers yields `O(k*H*T*d*r)`.

### Generalization notes
- Pros: expressive, query-conditional.
- Cons: not contractive by construction; relies on contraction regularizer.

## 3. Option B: learned-basis shrinkage (bounded scaling)
### Definition
Learn bases `Bk, Bv in R^{d x r}` and scalars `s(q) in [0, alpha_max]^r` where `alpha_max < 1` is fixed in config.
Project and shrink in the learned basis:
- `ck = K @ Bk`
- `ck_shrunk = ck * s(q)`
- `K_hat = K - (ck - ck_shrunk) @ Bk^T`
Likewise for `V`.

### Parameter count
Per protected layer:
- bases: `2*d*r`
- gating MLP(s): `~ 2*d*r + 2*r^2` (for K) and same for V if separate
Total: typically `~ (4*d*r + 4*r^2)` with shared gating.

### Compute complexity
Same order as Option A, often slightly cheaper due to shrinkage.

### Generalization notes
- Pros: stability framing is natural because `s(q)` is bounded below 1.
- Cons: may underfit complex corruptions.

## 4. Option C: shared global operator with small conditioning
### Definition
Share `W/U` (Option A) or `B` (Option B) globally across protected layers.
Provide conditioning vectors `e_layer[l]` and optional `e_head[h]` to the gating network.

### Parameter count
- global matrices: `O(d*r)`
- embeddings: `O(L*r)` (and optionally `O(H*r)`)
- small gating MLP.

### Compute complexity
Same functional form as A/B with lower parameter memory and better reuse.

### Generalization notes
- Pros: lower risk of overfitting; lower overhead.
- Cons: may miss head-specific vulnerabilities.

## 5. Default choice
Default is **Option A** with:
- `share_across_heads: true`
- `rank: 8`
- `apply_to: KV`
- mid-layer protection block chosen per model (see configs).
