# Threat / Fault Model - Corruption Family C

This spec defines the canonical corruption family `C` used to train and evaluate CacheMedic++.

## 1. State representation
- The current setup assumes batch size `B=1`.
- For each transformer layer `l`, the KV cache tensors are:
  - `K[l]` and `V[l]` with shape `[1, H, T, d]` (dtype = model dtype).
- A full protected state is `S = { (K[l],V[l]) for l in protect_layers }`.

## 2. Axes of corruption (deterministic masking)
Each corruption op is applied under three masks:
- `layer_mask[l] in {0,1}`
- `head_mask[h] in {0,1}`
- `time_mask[t] in {0,1}`

Mask sampling uses a dedicated `torch.Generator` seeded from `config.seed`.

### 2.1 Layer modes
- `layer_mode: "all"` -> apply to all layers in `protect_layers`.
- `layer_mode: "bernoulli"` -> sample `layer_mask[l] ~ Bernoulli(p_layer)` for each protected layer.
- `layer_mode: "targeted"` -> apply only to `layers: [...]`.

### 2.2 Head modes
- `head_mode: "all"`
- `head_mode: "bernoulli"` with `p_head`
- `head_mode: "targeted"` with `heads: [...]`

### 2.3 Time/segment modes
- `time_mode: "all_past"` -> corrupt all cached timesteps.
- `time_mode: "old_only"` -> corrupt only timesteps `t < T - N_recent`.
- `time_mode: "window"` with `window: [a,b]` -> corrupt timesteps `a <= t < b`.

## 3. Severity schedule and mixture Pi
Severity is controlled by:
- `epsilon` for continuous perturbations (noise magnitude relative to per-vector RMS)
- `dropout_p` for zeroing
- `bitflip_p` and `bitflip_eps_jump` for sparse bitflip-ish jumps
- `quant_bits` for quantization noise

A corruption run uses a mixture `Pi` that samples:
- a corruption type from a categorical distribution with weights
- per-type parameters (including an `epsilon` sampled from `eps_grid`)
- axis masks (layer/head/time)

The canonical default mixture is encoded in `configs/corruption/default_mix.yaml`.

## 4. Corruption operators (torch-only pseudocode)
All operators below are specified for a single layer's tensors `K,V` of shape `[1,H,T,d]`.

### Shared helpers
```python
def rms(x):
    # x: [1,H,T,d]
    return torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-8).to(x.dtype)

def mask_HT(head_mask, time_mask, H, T, device):
    mH = head_mask.view(1, H, 1, 1).to(device)
    mT = time_mask.view(1, 1, T, 1).to(device)
    return (mH & mT)  # [1,H,T,1] boolean
```

### 4.1 Gaussian noise (`type: gaussian`)
```python
def corrupt_gaussian(K, V, head_mask, time_mask, epsilon, gen):
    M = mask_HT(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    scaleK = epsilon * rms(K)
    scaleV = epsilon * rms(V)
    noiseK = torch.randn_like(K, generator=gen) * scaleK
    noiseV = torch.randn_like(V, generator=gen) * scaleV
    return torch.where(M, K + noiseK, K), torch.where(M, V + noiseV, V)
```

### 4.2 Dropout / zeroing (`type: dropout_zero`)
Elementwise zeroing over masked heads/timesteps.
```python
def corrupt_dropout_zero(K, V, head_mask, time_mask, dropout_p, gen):
    M = mask_HT(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    # dropout mask is per (head,time,dim)
    drop = torch.rand_like(K, generator=gen) < dropout_p
    Mfull = M.expand_as(K) & drop
    return torch.where(Mfull, torch.zeros_like(K), K), torch.where(Mfull, torch.zeros_like(V), V)
```

### 4.3 Orthogonal rotation (`type: orthogonal_rotation`)
Rotation is applied on the last dimension and must be deterministic.

Deterministic generation policy:
- generate `R` on CPU from a fixed seed using QR decomposition
- reuse `R` for the entire run unless config requests per-head rotations

```python
def make_orthogonal_matrix(d, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q  # [d,d]

def corrupt_rotation(K, V, head_mask, time_mask, R):
    M = mask_HT(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    Krot = K @ R.to(K.device)
    Vrot = V @ R.to(V.device)
    return torch.where(M, Krot, K), torch.where(M, Vrot, V)
```

### 4.4 Sparse bitflip-ish perturbation (`type: bitflipish_sparse`)
Simulates two float fault modes:
- sign flip: `x -> -x`
- jump: `x -> x + sign(randn) * bitflip_eps_jump * max(|x|, 1e-3)`

```python
def corrupt_bitflipish(X, head_mask, time_mask, bitflip_p, bitflip_eps_jump, gen):
    M = mask_HT(head_mask, time_mask, X.shape[1], X.shape[2], X.device).expand_as(X)
    sel = (torch.rand_like(X, generator=gen) < bitflip_p) & M
    flip = (torch.rand_like(X, generator=gen) < 0.5) & sel
    X2 = torch.where(flip, -X, X)
    jump_sel = sel & (~flip)
    jump = bitflip_eps_jump * torch.clamp(torch.abs(X), min=1e-3) * torch.sign(torch.randn_like(X, generator=gen))
    X3 = torch.where(jump_sel, X2 + jump, X2)
    return X3
```
Apply to K and V independently.

### 4.5 Quantization noise (`type: quant_noise`)
Simulates de/quant error with symmetric uniform quantization.

```python
def quant_dequant(x, n_bits):
    qmax = (2 ** (n_bits - 1)) - 1
    # per-head scale, max over time and dim
    max_abs = torch.amax(torch.abs(x.float()), dim=(2,3), keepdim=True) + 1e-8  # [1,H,1,1]
    scale = max_abs / qmax
    xq = torch.clamp(torch.round(x.float() / scale), -qmax, qmax)
    return (xq * scale).to(x.dtype)

def corrupt_quant_noise(K, V, head_mask, time_mask, quant_bits):
    M = mask_HT(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    Kq = quant_dequant(K, quant_bits)
    Vq = quant_dequant(V, quant_bits)
    return torch.where(M, Kq, K), torch.where(M, Vq, V)
```

### 4.6 Contiguous overwrite (`type: contiguous_overwrite`)
This models history-swapping-like cache manipulation by overwriting a contiguous block of past KV with a cached donor block.

Deterministic donor policy:
- donor KV is produced by running a fixed donor prompt through the same model (clean, no repair)
- donor prompt must be deterministic and stored in config as `donor_prompt` or generated from `seed`

Overwrite policy:
- canonical config key: `corruption.params.overwrite_window: [a,b]`
- if `corruption.axes.time_mode == "window"` and `corruption.axes.window` is provided, that window takes precedence for this corruption type
- choose a segment `[a,b)` in past timesteps (respecting `time_mask`)
- overwrite `K[:, :, a:b, :]` and `V[:, :, a:b, :]` with donor KV at matching indices (or truncated)

```python
def corrupt_contiguous_overwrite(K, V, head_mask, time_mask, donorK, donorV, window):
    # window = [a,b]
    a, b = window
    M = mask_HT(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    # apply overwrite only where M is true within [a,b)
    K2, V2 = K.clone(), V.clone()
    T = K.shape[2]
    a2, b2 = max(0,a), min(T,b)
    if a2 < b2:
        # donor length may differ
        donor_len = donorK.shape[2]
        seg_len = b2 - a2
        use_len = min(seg_len, donor_len)
        # overwrite head/time region; mask per head
        hm = head_mask.view(1, -1, 1, 1).to(K.device)
        K2[:, :, a2:a2+use_len, :] = torch.where(hm, donorK[:, :, :use_len, :].to(K.device), K2[:, :, a2:a2+use_len, :])
        V2[:, :, a2:a2+use_len, :] = torch.where(hm, donorV[:, :, :use_len, :].to(V.device), V2[:, :, a2:a2+use_len, :])
    return K2, V2
```

### 4.7 Targeted wrapper (`type: targeted`)
The targeted wrapper applies any corruption type but forces `layer_mode=targeted` and/or `head_mode=targeted`.

## 5. Determinism notes
- Use a single `torch.Generator` per run for all corruption sampling.
- Log every sampled corruption event (type, epsilon, masks summary) to `logs/train.jsonl`.

