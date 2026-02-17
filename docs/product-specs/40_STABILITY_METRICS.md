# Stability Metrics (Primary Results)

CacheMedic++ treats stability metrics as first-class outputs.

## 1. Logit sensitivity curves
Goal: measure expected next-token logit drift vs perturbation magnitude.

### 1.1 Definition
Let `S` be a clean KV state (protected layers only). Let `delta` be a sampled perturbation tensor with the same structure as `S`.
For a prompt `x`, define:
- `z_clean = z(S)`
- `z_corr = z(S + delta)` (no repair)
- `z_rep = z( R_phi(S + delta) )` (repair active)

Sensitivity at a magnitude level is:
`Sens(norm) = E_{x,delta}[ || z(S + delta) - z(S) ||_2 ]`
And repaired sensitivity uses `z_rep`.

### 1.2 Sampling protocol (finite differences)
- Choose `num_prompts` prompts.
- For each prompt and each `delta_norm` in `stability.delta_norms`, sample `num_directions` random perturbations.
- Perturbations are RMS-scaled:
  - sample `u ~ N(0,1)` for each tensor in `S`
  - normalize so that `||u||_F` equals 1 in each layer
  - set `delta = delta_norm * u * rms_scale`, where `rms_scale` is the per-layer RMS of the clean KV tensor.

### 1.3 Cost controls
- Optionally compute `||Â·||_2` over logits restricted to top-k indices from `z_clean` with `k = stability.topk_logits`.
- Default: `topk_logits = 1000`.

### 1.4 Required output
- Write a plot to `<RUN_DIR>/paper/figures/figC_logit_sensitivity.png`.
- Include three curves:
  1. baseline (no defense)
  2. heuristics (best heuristic baseline selected per protocol)
  3. CacheMedic++
- If `lambda_contr` sweep is run, include an additional curve for `lambda_contr = 0`.

## 2. Amplification maps (layer/head)
Goal: estimate which layers/heads amplify KV perturbations into output drift.

### 2.1 Definition
For a fixed layer `l` and head `h`, construct a perturbation `delta_{l,h}` that is nonzero only in that layer/head and only over a chosen time segment.
Define amplification:
`gamma(l,h) = median_{x,delta}[ || z(S + delta_{l,h}) - z(S) ||_2 / (||delta_{l,h}||_F + eps0) ]`
Report maps before and after repair.

### 2.2 Protocol
- Use `stability.layers` and optionally a head subset if `stability.head_subset` is configured.
- Use `time_mode = old_only` for stability by default (exposes long-range dependence).
- Use `num_directions` perturbations per (l,h) and compute median.

### 2.3 Required output
- Write a heatmap plot to `<RUN_DIR>/paper/figures/figD_amplification_map.png`.

## 3. Metric records
Stability metrics must be written to `<RUN_DIR>/metrics/stability_metrics.json` as JSON matching `schemas/metric_record.schema.json`.
Required keys:
- `stability.logit_sensitivity` (array of points with `delta_norm` and `value` for baseline/heuristics/cachemedic)
- `stability.amplification_map` (2D array or sparse list with indices)
- `stability.settings` (sampling counts, topk, time_mode)
