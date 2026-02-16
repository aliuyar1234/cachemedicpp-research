# Symbol table

All symbols in the harness use the following meanings.

- `theta` (θ): frozen base model parameters.
- `phi` (φ): trainable repair operator parameters.
- `S`: clean KV state across protected layers.
- `S_corr`: corrupted KV state after applying a corruption operator.
- `S_rep`: repaired KV state after applying `R_phi` to `S_corr`.
- `C`: corruption operator sampled from a corruption family.
- `Pi`: mixture distribution over corruption types and parameters.
- `K^(l), V^(l)`: cached keys and values at layer `l`.
- `q`: query tensor for current decode step.
- `z(S)`: next-token logits produced when the model runs with state `S`.
- `Delta(l)`: state error at layer `l` (corrupted minus clean).
- `rho(l)`: contraction ratio at layer `l`.
- `alpha_contr`: contraction target threshold.
- `gamma(l,h)`: amplification factor for layer `l` and head `h`.
