"""Deterministic corruption family C for CacheMedic++."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import resolve_repo_root


EPS0 = 1e-8


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _randn_like(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, generator=gen, device="cpu", dtype=torch.float32).to(
        device=x.device, dtype=x.dtype
    )


def _rand_like(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.rand(x.shape, generator=gen, device="cpu", dtype=torch.float32).to(
        device=x.device, dtype=x.dtype
    )


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + EPS0).to(x.dtype)


def mask_ht(
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    H: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    m_h = head_mask.view(1, H, 1, 1).to(device=device, dtype=torch.bool)
    m_t = time_mask.view(1, 1, T, 1).to(device=device, dtype=torch.bool)
    return m_h & m_t


def make_orthogonal_matrix(d: int, seed: int) -> torch.Tensor:
    g = make_generator(seed)
    a = torch.randn((d, d), generator=g, device="cpu", dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    return q


def quant_dequant(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    qmax = (2 ** (int(n_bits) - 1)) - 1
    max_abs = torch.amax(torch.abs(x.float()), dim=(2, 3), keepdim=True) + EPS0
    scale = max_abs / float(qmax)
    xq = torch.clamp(torch.round(x.float() / scale), -qmax, qmax)
    return (xq * scale).to(x.dtype)


def corrupt_gaussian(
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    epsilon: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = mask_ht(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    scaleK = float(epsilon) * rms(K)
    scaleV = float(epsilon) * rms(V)
    noiseK = _randn_like(K, gen) * scaleK
    noiseV = _randn_like(V, gen) * scaleV
    return torch.where(M, K + noiseK, K), torch.where(M, V + noiseV, V)


def corrupt_dropout_zero(
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    dropout_p: float,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = mask_ht(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    drop = _rand_like(K, gen) < float(dropout_p)
    M_full = M.expand_as(K) & drop
    zK = torch.zeros_like(K)
    zV = torch.zeros_like(V)
    return torch.where(M_full, zK, K), torch.where(M_full, zV, V)


def corrupt_rotation(
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = mask_ht(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    Rdev = R.to(device=K.device, dtype=K.dtype)
    Krot = K @ Rdev
    Vrot = V @ Rdev
    return torch.where(M, Krot, K), torch.where(M, Vrot, V)


def corrupt_bitflipish(
    X: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    bitflip_p: float,
    bitflip_eps_jump: float,
    gen: torch.Generator,
) -> torch.Tensor:
    M = mask_ht(head_mask, time_mask, X.shape[1], X.shape[2], X.device).expand_as(X)
    sel = (_rand_like(X, gen) < float(bitflip_p)) & M
    flip = (_rand_like(X, gen) < 0.5) & sel
    X2 = torch.where(flip, -X, X)
    jump_sel = sel & (~flip)
    jump = float(bitflip_eps_jump) * torch.clamp(torch.abs(X), min=1e-3) * torch.sign(_randn_like(X, gen))
    return torch.where(jump_sel, X2 + jump, X2)


def corrupt_quant_noise(
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    quant_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = mask_ht(head_mask, time_mask, K.shape[1], K.shape[2], K.device)
    Kq = quant_dequant(K, quant_bits)
    Vq = quant_dequant(V, quant_bits)
    return torch.where(M, Kq, K), torch.where(M, Vq, V)


def corrupt_contiguous_overwrite(
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    donorK: torch.Tensor,
    donorV: torch.Tensor,
    window: list[int] | tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = int(window[0]), int(window[1])
    T = int(K.shape[2])
    a2, b2 = max(0, a), min(T, b)
    K2, V2 = K.clone(), V.clone()
    if a2 >= b2:
        return K2, V2

    donor_len = int(donorK.shape[2])
    seg_len = b2 - a2
    use_len = min(seg_len, donor_len)
    if use_len <= 0:
        return K2, V2

    head_mask_4d = head_mask.view(1, -1, 1, 1).to(device=K.device, dtype=torch.bool)
    local_tmask = time_mask[a2 : a2 + use_len].view(1, 1, use_len, 1).to(device=K.device, dtype=torch.bool)
    M = head_mask_4d & local_tmask

    donorK_slice = donorK[:, :, :use_len, :].to(device=K.device, dtype=K.dtype)
    donorV_slice = donorV[:, :, :use_len, :].to(device=V.device, dtype=V.dtype)
    Kseg = K2[:, :, a2 : a2 + use_len, :]
    Vseg = V2[:, :, a2 : a2 + use_len, :]
    K2[:, :, a2 : a2 + use_len, :] = torch.where(M, donorK_slice, Kseg)
    V2[:, :, a2 : a2 + use_len, :] = torch.where(M, donorV_slice, Vseg)
    return K2, V2


def sample_layer_mask(
    protect_layers: list[int],
    axes: dict[str, Any],
    gen: torch.Generator,
) -> dict[int, bool]:
    mode = axes.get("layer_mode", "all")
    if mode == "all":
        return {layer: True for layer in protect_layers}
    if mode == "targeted":
        targets = {int(x) for x in axes.get("layers", [])}
        return {layer: layer in targets for layer in protect_layers}
    if mode == "bernoulli":
        p = float(axes.get("p_layer", 1.0))
        out: dict[int, bool] = {}
        for layer in protect_layers:
            v = torch.rand((1,), generator=gen, device="cpu").item() < p
            out[layer] = bool(v)
        return out
    raise ValueError(f"Unsupported layer_mode: {mode}")


def sample_head_mask(H: int, axes: dict[str, Any], gen: torch.Generator) -> torch.Tensor:
    mode = axes.get("head_mode", "all")
    if mode == "all":
        return torch.ones((H,), dtype=torch.bool)
    if mode == "targeted":
        out = torch.zeros((H,), dtype=torch.bool)
        for h in axes.get("heads", []):
            hi = int(h)
            if 0 <= hi < H:
                out[hi] = True
        return out
    if mode == "bernoulli":
        p = float(axes.get("p_head", 1.0))
        draws = torch.rand((H,), generator=gen, device="cpu")
        return draws < p
    raise ValueError(f"Unsupported head_mode: {mode}")


def sample_time_mask(T: int, axes: dict[str, Any], gen: torch.Generator) -> torch.Tensor:
    del gen  # reserved for future randomized time policies
    mode = axes.get("time_mode", "all_past")
    if mode == "all_past":
        return torch.ones((T,), dtype=torch.bool)
    if mode == "old_only":
        n_recent = max(0, int(axes.get("N_recent", 0)))
        limit = max(0, T - n_recent)
        out = torch.zeros((T,), dtype=torch.bool)
        out[:limit] = True
        return out
    if mode == "window":
        w = axes.get("window", [0, T])
        a, b = int(w[0]), int(w[1])
        out = torch.zeros((T,), dtype=torch.bool)
        out[max(0, a) : min(T, b)] = True
        return out
    raise ValueError(f"Unsupported time_mode: {mode}")


@dataclass
class CorruptionEvent:
    corruption_type: str
    epsilon: float

    def as_log_dict(self) -> dict[str, Any]:
        return {"type": self.corruption_type, "epsilon": float(self.epsilon)}


class CorruptionEngine:
    """Deterministic corruption sampler and operator dispatcher."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        protect_layers: list[int],
        repo_root: Path | None = None,
    ) -> None:
        self.config = config
        self.protect_layers = [int(x) for x in protect_layers]
        self.repo_root = repo_root or resolve_repo_root()
        self.seed = int(config["seed"])
        self.gen = make_generator(self.seed)
        self.mix = self._load_mixture(config["corruption"]["mixture"])
        self.default_params = dict(self.mix.get("default_params", {}))
        self.weights = dict(self.mix.get("weights", {}))
        if not self.weights:
            raise ValueError("Corruption mixture has no weights.")
        self._rot_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _load_mixture(self, mixture_ref: str) -> dict[str, Any]:
        candidate = Path(mixture_ref)
        if candidate.exists():
            path = candidate
        elif (self.repo_root / mixture_ref).exists():
            path = self.repo_root / mixture_ref
        else:
            path = self.repo_root / "configs" / "corruption" / f"{mixture_ref}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Corruption mixture config not found: {mixture_ref}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid corruption mixture YAML: {path}")
        return data

    def sample_event(self, *, allowed_types: list[str] | None = None) -> CorruptionEvent:
        eps_grid = [float(x) for x in self.config["corruption"]["eps_grid"]]
        eps_idx = int(torch.randint(0, len(eps_grid), (1,), generator=self.gen, device="cpu").item())
        epsilon = eps_grid[eps_idx]

        allowed = set(allowed_types) if allowed_types else set(self.weights.keys())
        keys = [k for k in self.weights.keys() if k in allowed]
        if not keys:
            raise ValueError("No corruption types available after applying filters.")
        probs = torch.tensor([float(self.weights[k]) for k in keys], dtype=torch.float64)
        probs = probs / probs.sum()
        idx = int(torch.multinomial(probs, num_samples=1, generator=self.gen).item())
        return CorruptionEvent(corruption_type=keys[idx], epsilon=epsilon)

    def _rotation(self, d: int) -> torch.Tensor:
        seed = int(self.config["corruption"]["params"].get("rotation_seed", 0))
        key = (d, seed)
        if key not in self._rot_cache:
            self._rot_cache[key] = make_orthogonal_matrix(d, seed)
        return self._rot_cache[key]

    def apply_to_layer(
        self,
        *,
        layer_index: int,
        K: torch.Tensor,
        V: torch.Tensor,
        event: CorruptionEvent,
        donor: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_masks: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        axes = dict(self.config["corruption"]["axes"])
        params = dict(self.default_params)
        params.update(self.config["corruption"].get("params", {}))

        layer_mask = sample_layer_mask(self.protect_layers, axes, self.gen)
        if not layer_mask.get(int(layer_index), False):
            summary = {"layer_active": False, "head_count": 0, "time_count": 0}
            if return_masks:
                summary["head_mask"] = torch.zeros((int(K.shape[1]),), dtype=torch.bool)
                summary["time_mask"] = torch.zeros((int(K.shape[2]),), dtype=torch.bool)
            return K, V, summary

        H, T = int(K.shape[1]), int(K.shape[2])
        head_mask = sample_head_mask(H, axes, self.gen)
        time_mask = sample_time_mask(T, axes, self.gen)

        ctype = event.corruption_type
        if ctype == "targeted":
            base_type = str(params.get("targeted_base_type", "gaussian"))
            ctype = base_type

        if ctype == "gaussian":
            K2, V2 = corrupt_gaussian(
                K,
                V,
                head_mask=head_mask,
                time_mask=time_mask,
                epsilon=event.epsilon,
                gen=self.gen,
            )
        elif ctype == "dropout_zero":
            K2, V2 = corrupt_dropout_zero(
                K,
                V,
                head_mask=head_mask,
                time_mask=time_mask,
                dropout_p=float(params["dropout_p"]),
                gen=self.gen,
            )
        elif ctype == "orthogonal_rotation":
            R = self._rotation(int(K.shape[-1]))
            K2, V2 = corrupt_rotation(K, V, head_mask=head_mask, time_mask=time_mask, R=R)
        elif ctype == "bitflipish_sparse":
            K2 = corrupt_bitflipish(
                K,
                head_mask=head_mask,
                time_mask=time_mask,
                bitflip_p=float(params["bitflip_p"]),
                bitflip_eps_jump=float(params["bitflip_eps_jump"]),
                gen=self.gen,
            )
            V2 = corrupt_bitflipish(
                V,
                head_mask=head_mask,
                time_mask=time_mask,
                bitflip_p=float(params["bitflip_p"]),
                bitflip_eps_jump=float(params["bitflip_eps_jump"]),
                gen=self.gen,
            )
        elif ctype == "quant_noise":
            K2, V2 = corrupt_quant_noise(
                K,
                V,
                head_mask=head_mask,
                time_mask=time_mask,
                quant_bits=int(params["quant_bits"]),
            )
        elif ctype == "contiguous_overwrite":
            donorK, donorV = donor if donor is not None else (K, V)
            window = params.get("overwrite_window", [0, int(K.shape[2])])
            if axes.get("time_mode") == "window" and axes.get("window"):
                window = axes["window"]
            K2, V2 = corrupt_contiguous_overwrite(
                K,
                V,
                head_mask=head_mask,
                time_mask=time_mask,
                donorK=donorK,
                donorV=donorV,
                window=window,
            )
        else:
            raise ValueError(f"Unsupported corruption type: {ctype}")

        summary = {
            "layer_active": True,
            "head_count": int(head_mask.sum().item()),
            "time_count": int(time_mask.sum().item()),
            "head_frac": float(head_mask.float().mean().item()) if H > 0 else 0.0,
            "time_frac": float(time_mask.float().mean().item()) if T > 0 else 0.0,
        }
        if return_masks:
            summary["head_mask"] = head_mask
            summary["time_mask"] = time_mask
        return K2, V2, summary


def renormalize_weights(weights: dict[str, float], keep_types: list[str]) -> dict[str, float]:
    kept = {k: float(v) for k, v in weights.items() if k in set(keep_types)}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("Cannot renormalize corruption weights: no positive weights for keep_types.")
    return {k: v / total for k, v in kept.items()}


def auc(values: list[float], xs: list[float]) -> float:
    if len(values) != len(xs) or len(values) < 2:
        return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        dx = float(xs[i]) - float(xs[i - 1])
        area += 0.5 * dx * (float(values[i]) + float(values[i - 1]))
    return area


def deterministic_noise(tag: str, scale: float = 1.0) -> float:
    h = 0
    for ch in tag:
        h = (h * 131 + ord(ch)) % 10_000_019
    return scale * ((h / 10_000_019.0) - 0.5)
