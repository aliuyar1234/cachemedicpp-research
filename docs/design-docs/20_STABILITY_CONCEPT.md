# What we mean by “stability”

CacheMedic++ treats autoregressive inference as a dynamical system whose persistent state is the KV cache. A system is **more stable** if **small state perturbations cause smaller output changes**.

In this harness, stability is operationalized with **measurable, reproducible proxies**:

1. **Logit sensitivity curve**
   \[
   \mathcal{S}(\|\delta\|)=\mathbb{E}_{x,\delta}\left[\|z(S+\delta)-z(S)\|_2\right]
   \]
   where \(z(\cdot)\) are next-token logits and \(\delta\) perturbs the cache state.

2. **Amplification map**
   \[
   \gamma_{l,h}=\mathrm{median}_{x,\delta}\left(\frac{\|z(S+\delta_{l,h})-z(S)\|_2}{\|\delta_{l,h}\|_F}\right)
   \]
   where \(\delta_{l,h}\) perturbs only layer \(l\), head \(h\).

CacheMedic++ aims to reduce these quantities while preserving clean-task performance.
