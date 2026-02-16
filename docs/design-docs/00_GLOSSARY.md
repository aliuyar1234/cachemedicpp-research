# Glossary

- **KV cache / cache state**: The persistent attention memory stored as past keys and values during autoregressive decoding. In this harness, a layer state is 
  \(K^{(l)},V^{(l)} \in \mathbb{R}^{1\times H\times T\times d}\) for batch size 1.
- **State \(S\)**: The collection of KV tensors over protected layers: \(S=\{(K^{(l)},V^{(l)})\}_{l\in\mathcal{L}_{prot}}\).
- **Corruption family \(\mathcal{C}\)**: A deterministic family of functions that maps a clean state \(S\) to a corrupted state \(\tilde S\) under parameters (type, severity, axes masks). Canonical in `../product-specs/10_THREAT_MODEL.md`.
- **Mixture \(\Pi\)**: A discrete distribution over corruption types and parameter sampling rules.
- **Repair / stability operator \(R_\phi\)**: A learned operator inserted inside attention that maps corrupted KV cache tensors to repaired tensors before attention logits are computed.
- **Teacher vs student**: Teacher is the frozen base model on clean caches; student is the same base model with corruption + repair applied.
- **Identity regularization**: Loss term that penalizes deviation from identity when corruption is disabled.
- **Contraction regularization**: Loss term that penalizes a repair mapping that fails to reduce the distance to the clean state.
- **Logit sensitivity**: Expected \(\ell_2\) drift of next-token logits under cache-state perturbations.
- **Amplification map**: Layer/head-local sensitivity proxy \(\gamma_{l,h}\) mapping perturbation magnitude to logit drift.
