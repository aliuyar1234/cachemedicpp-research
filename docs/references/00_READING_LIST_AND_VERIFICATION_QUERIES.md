# Reading List (Non-exhaustive) + Verification Queries

This list supports novelty auditing and framing. It does not claim any results; it identifies prior art to differentiate from.

## Cluster A: KV cache corruption / faults / attacks
- Hossain et al. "Can Transformer Memory Be Corrupted?" (MTI framework). 
- Nahian et al. "CacheTrap" (KV cache bitflip-style Trojan triggers).
- Ganesh et al. "Whose Narrative is it Anyway? A KV Cache Manipulation Attack" (history swapping).
- Kelle (ACM, 2025) “Simulation of KV-cache data corruption” (hardware-oriented fault modeling; see DOI 10.1145/3725843.3756071). 
- Wu et al. (NDSS) prompt leakage via KV cache sharing in multi-tenant serving.

## Cluster B: KV cache interventions / state manipulation
- Belitsky et al. "KV Cache Steering" (one-shot KV interventions for control).
- RobustKV (ICLR 2025) KV cache eviction for jailbreak defense.
- DTR (arXiv 2025) dynamic token reweighting with KV-based defense for VLMs.
- Output-perturbation-based KV criticality (OpenReview) (useful measurement ideas).

## Cluster C: Stability/contractive operators and internal editing
- Activation Boundary Defense (ABD) (ACL) inference-time activation constraints.
- Spectral Editing of Activations (SEA) (NeurIPS 2024) projection-style editing.
- Conceptor steering (arXiv 2024) projection operators for steering.
- Pseudo-contractive denoiser analysis (OpenReview) (motivates contraction regularization).
- Lipschitz continuity analysis (ICLR) (general robustness framing).

## Verification queries (use these exact queries when refreshing prior art)
- "KV cache corruption transformer memory corrupted" 
- "KV cache bit-flip fault attack" 
- "history swapping KV cache manipulation" 
- "KV cache steering intervention" 
- "activation boundary defense LLM" 
- "conceptor steering language model" 
- "contractive regularizer denoiser pseudo-contractive" 
- "logit sensitivity finite difference stability metric transformer" 

## How this project differentiates (framing reminder)
CacheMedic++ is not KV compression/eviction/clustering. It is a learned **stability operator** with explicit contraction regularization and stability metrics as primary outputs.
