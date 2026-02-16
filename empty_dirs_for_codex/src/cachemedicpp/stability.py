"""CLI for CacheMedic++ stability metrics via finite-difference perturbations."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import nn

from .config import load_and_validate_config, resolve_repo_root, stable_json_hash
from .data import load_wikitext2
from .eval import HEURISTICS, _apply_heuristic
from .hf_patch import enforce_v1_constraints, patch_gpt2_attention
from .io_utils import append_jsonl, ensure_run_layout, load_run_record, utc_now_iso, write_json
from .repair_ops import build_repair_operator
from .runtime import ModelGeometry, load_model_and_tokenizer

EPS0 = 1e-8
STABILITY_PROGRESS_PATH = Path("metrics") / "stability_progress.json"


def _select_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CacheMedic++ fail-closed: config requested CUDA, "
                "but torch.cuda.is_available() is False."
            )
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_name}'. Use 'cpu' or 'cuda[:index]'.")


def _build_repair_modules(
    cfg: dict[str, Any],
    geom: ModelGeometry,
    protect_layers: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.ModuleDict:
    modules = nn.ModuleDict()
    for layer in protect_layers:
        modules[f"layer_{layer}"] = build_repair_operator(
            cfg["repair"],
            head_dim=geom.head_dim,
            num_heads=geom.num_heads,
            layer_index=layer,
            num_layers=geom.num_layers,
        )
    modules.to(device=device, dtype=dtype)
    modules.eval()
    return modules


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    pats = list(ckpt_dir.glob("repair_step_*.pt"))
    if not pats:
        return None

    def step_of(path: Path) -> int:
        m = re.search(r"repair_step_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    pats.sort(key=step_of)
    return pats[-1]


def _load_repair_state(run_dir: Path, modules: nn.ModuleDict) -> bool:
    ckpt = _latest_checkpoint(run_dir)
    if ckpt is None:
        return False
    payload = torch.load(ckpt, map_location="cpu")
    state = payload.get("repair_state")
    if not isinstance(state, dict):
        return False
    modules.load_state_dict(state, strict=False)
    return True


def _candidate_run_config_path(run_dir: Path) -> Path | None:
    for rel in ("sweep_resolved_config.yaml", "config_resolved.yaml"):
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _load_run_config_shallow(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = _candidate_run_config_path(run_dir)
    if cfg_path is None:
        return None
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _run_signature(cfg: dict[str, Any]) -> tuple[Any, ...]:
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    repair = cfg.get("repair", {}) if isinstance(cfg.get("repair"), dict) else {}
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else {}
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
    layers = tuple(int(x) for x in repair.get("protect_layers", []))
    return (
        str(model.get("name", "")),
        str(repair.get("family", "")),
        str(repair.get("apply_to", "")),
        int(repair.get("rank", 0) or 0),
        layers,
        int(train.get("steps", 0) or 0),
        str(eval_cfg.get("ood_protocol", "")),
    )


def _resolve_no_contr_pair_run_dir(run_dir: Path, cfg: dict[str, Any]) -> Path | None:
    parent = run_dir.parent
    if not parent.exists():
        return None

    preferred: list[Path] = []
    if "main_contr" in run_dir.name:
        preferred.append(parent / run_dir.name.replace("main_contr", "ablation_no_contr", 1))

    candidates: list[Path] = []
    seen: set[Path] = set()
    for p in preferred:
        if p not in seen:
            seen.add(p)
            candidates.append(p)
    for sib in parent.iterdir():
        if not sib.is_dir():
            continue
        if sib.name in {run_dir.name, "control"}:
            continue
        if sib not in seen:
            seen.add(sib)
            candidates.append(sib)

    target_sig = _run_signature(cfg)
    for cand in candidates:
        cand_cfg = _load_run_config_shallow(cand)
        if cand_cfg is None:
            continue
        train_cfg = cand_cfg.get("train", {}) if isinstance(cand_cfg.get("train"), dict) else {}
        try:
            lam = float(train_cfg.get("lambda_contr", 1.0))
        except Exception:
            continue
        if lam != 0.0:
            continue
        if _run_signature(cand_cfg) != target_sig:
            continue
        if _latest_checkpoint(cand) is None:
            continue
        return cand
    return None


def _resolve_best_heuristic(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics" / "task_metrics.json"
    if not metrics_path.exists():
        return "smoothing"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return "smoothing"
    metrics = payload.get("metrics", {})
    best = str(metrics.get("best_heuristic_name", "smoothing"))
    if best in HEURISTICS:
        return best
    return "smoothing"


def _stable_seed(base: int, *parts: int) -> int:
    mod = 2_147_483_647
    value = int(base) % mod
    for i, part in enumerate(parts):
        value = (value * 1_103_515_245 + 12_345 + int(part) + 97 * i) % mod
    return value


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _mean_or_none(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(torch.tensor(xs, dtype=torch.float32).median().item())


def _pause_request_path(run_dir: Path) -> Path:
    return run_dir / "control" / "pause.request"


def _pause_requested(run_dir: Path) -> bool:
    return _pause_request_path(run_dir).exists()


def _stability_progress_file(run_dir: Path) -> Path:
    return run_dir / STABILITY_PROGRESS_PATH


def _load_stability_progress(run_dir: Path) -> dict[str, Any] | None:
    path = _stability_progress_file(run_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_stability_progress(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(_stability_progress_file(run_dir), payload)


def _delta_key(delta_norm: float) -> str:
    return f"{float(delta_norm):.12g}"


def _make_stability_heartbeat(
    *,
    run_id: str,
    status: str,
    step: int,
    total_steps: int,
    elapsed_wall_sec: float,
    reason: str,
    resumed_from: str | None,
    current_unit: dict[str, Any] | None,
) -> dict[str, Any]:
    done = max(0, int(step))
    total = max(1, int(total_steps))
    progress = min(1.0, float(done / total))
    units_per_sec = float(done / max(elapsed_wall_sec, 1e-8)) if done > 0 else 0.0
    remaining = max(0, total - done)
    eta = float(remaining / max(units_per_sec, 1e-8)) if done > 0 else None
    payload: dict[str, Any] = {
        "phase": "stability",
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "reason": reason,
        "step": done,
        "total_steps": total,
        "progress": progress,
        "steps_remaining": remaining,
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "eta_sec": eta,
        "steps_per_sec": units_per_sec,
        "resumed_from": resumed_from,
    }
    if current_unit is not None:
        payload["current_unit"] = current_unit
    return payload


def _emit_stability_heartbeat(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "logs" / "stability_heartbeat.json", payload)
    append_jsonl(run_dir / "logs" / "stability_heartbeat.jsonl", payload)


def _make_time_mask(
    T: int,
    *,
    stability_cfg: dict[str, Any],
    corruption_cfg: dict[str, Any],
) -> torch.Tensor:
    mode = str(stability_cfg.get("time_mode", "all_past"))
    if mode == "all_past":
        return torch.ones((T,), dtype=torch.bool)

    if mode == "old_only":
        n_recent = max(0, int(stability_cfg.get("N_recent", 0)))
        out = torch.zeros((T,), dtype=torch.bool)
        out[: max(0, T - n_recent)] = True
        return out

    if mode == "window":
        window = stability_cfg.get("window")
        if window is None:
            window = (corruption_cfg.get("axes", {}) or {}).get("window", [0, T])
        a, b = int(window[0]), int(window[1])
        out = torch.zeros((T,), dtype=torch.bool)
        out[max(0, a) : min(T, b)] = True
        return out

    raise ValueError(f"Unsupported stability.time_mode: {mode}")


def _masked_unit_noise_like(x: torch.Tensor, mask: torch.Tensor, *, seed: int) -> torch.Tensor:
    if not bool(mask.any()):
        return torch.zeros_like(x)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=gen, device="cpu", dtype=torch.float32).to(
        device=x.device, dtype=x.dtype
    )
    masked = torch.where(mask, noise, torch.zeros_like(noise))
    norm = torch.linalg.vector_norm(masked.float())
    if float(norm.item()) <= EPS0:
        return torch.zeros_like(x)
    return masked / norm.to(device=x.device, dtype=x.dtype)


def _build_delta_pair(
    *,
    K_clean: torch.Tensor,
    V_clean: torch.Tensor,
    delta_norm: float,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(delta_norm) <= 0.0:
        return torch.zeros_like(K_clean), torch.zeros_like(V_clean)

    H, T = int(K_clean.shape[1]), int(K_clean.shape[2])
    m_h = head_mask.view(1, H, 1, 1).to(device=K_clean.device, dtype=torch.bool)
    m_t = time_mask.view(1, 1, T, 1).to(device=K_clean.device, dtype=torch.bool)
    mask = (m_h & m_t).expand_as(K_clean)

    uK = _masked_unit_noise_like(K_clean, mask, seed=seed)
    uV = _masked_unit_noise_like(V_clean, mask, seed=seed + 1)

    rms_k = torch.sqrt(torch.mean(K_clean.float() ** 2) + EPS0).to(
        device=K_clean.device, dtype=K_clean.dtype
    )
    rms_v = torch.sqrt(torch.mean(V_clean.float() ** 2) + EPS0).to(
        device=V_clean.device, dtype=V_clean.dtype
    )
    scale = float(delta_norm)
    return scale * rms_k * uK, scale * rms_v * uV


def _pair_norm(delta_k: torch.Tensor, delta_v: torch.Tensor) -> float:
    sq = torch.sum(delta_k.float() ** 2) + torch.sum(delta_v.float() ** 2)
    return float(torch.sqrt(sq + EPS0).item())


def _logit_drift(
    z_clean: torch.Tensor,
    z_other: torch.Tensor,
    *,
    topk_idx: torch.Tensor | None,
) -> float:
    if topk_idx is None:
        diff = z_other - z_clean
    else:
        diff = z_other.index_select(0, topk_idx) - z_clean.index_select(0, topk_idx)
    return float(torch.linalg.vector_norm(diff.float()).item())


def _forward_next_token_logits(
    model: Any,
    input_ids: torch.Tensor,
    *,
    patch_fn: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    | None,
    protect_layers: list[int],
) -> torch.Tensor:
    if patch_fn is None:
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=True).logits[0, -1, :].float().detach()

    handle = patch_gpt2_attention(model, patch_fn, protect_layers=protect_layers)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=True).logits[0, -1, :].float().detach()
    finally:
        handle.unpatch()


def _prepare_prompt_tensors(
    tokenizer: Any,
    texts: list[str],
    *,
    num_prompts: int,
    max_seq_len: int,
    device: torch.device,
) -> list[torch.Tensor]:
    prompts: list[torch.Tensor] = []
    eos = tokenizer.eos_token_id
    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"]
        if len(ids) < 2 and eos is not None:
            ids = [*ids, int(eos)]
        if len(ids) < 2:
            continue
        prompts.append(torch.tensor([ids], dtype=torch.long, device=device))
        if len(prompts) >= num_prompts:
            break

    if not prompts:
        fallback = [int(eos), int(eos)] if eos is not None else [0, 1]
        prompts = [torch.tensor([fallback], dtype=torch.long, device=device)]

    while len(prompts) < num_prompts:
        prompts.append(prompts[len(prompts) % len(prompts)].clone())
    return prompts[:num_prompts]


def _topk_index(z_clean: torch.Tensor, topk_logits: int) -> torch.Tensor | None:
    if topk_logits <= 0:
        return None
    if topk_logits >= int(z_clean.shape[0]):
        return None
    return torch.topk(z_clean, k=topk_logits, dim=0).indices


def _compute_logit_sensitivity(
    *,
    model: Any,
    prompts: list[torch.Tensor],
    clean_logits: list[torch.Tensor],
    topk_indices: list[torch.Tensor | None],
    layers: list[int],
    delta_norms: list[float],
    num_directions: int,
    seed: int,
    stability_cfg: dict[str, Any],
    corruption_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
    best_heuristic: str,
    repair_modules: nn.ModuleDict,
    repair_no_contr_modules: nn.ModuleDict | None,
) -> list[dict[str, Any]]:
    layer_set = set(layers)
    curves: list[dict[str, float]] = []

    for dn in delta_norms:
        samples = {
            "baseline": [],
            "heuristics": [],
            "cachemedic": [],
            "cachemedic_no_contr": [],
        }
        delta_pair_norms: list[float] = []
        delta_pair_rel_norms: list[float] = []
        delta_k_rel_norms: list[float] = []
        delta_v_rel_norms: list[float] = []
        for p_idx, input_ids in enumerate(prompts):
            z_clean = clean_logits[p_idx]
            topk_idx = topk_indices[p_idx]

            if float(dn) <= 0.0:
                for _ in range(num_directions):
                    samples["baseline"].append(0.0)
                    samples["heuristics"].append(0.0)
                    samples["cachemedic"].append(0.0)
                    if repair_no_contr_modules is not None:
                        samples["cachemedic_no_contr"].append(0.0)
                continue

            for d_idx in range(num_directions):
                delta_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                mask_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

                def patch_baseline(
                    layer_idx: int,
                    q: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    del q
                    if layer_idx not in layer_set:
                        return key, value
                    if layer_idx not in delta_cache:
                        H, T = int(key.shape[1]), int(key.shape[2])
                        head_mask = torch.ones((H,), dtype=torch.bool)
                        time_mask = _make_time_mask(
                            T,
                            stability_cfg=stability_cfg,
                            corruption_cfg=corruption_cfg,
                        )
                        dK, dV = _build_delta_pair(
                            K_clean=key.detach(),
                            V_clean=value.detach(),
                            delta_norm=dn,
                            head_mask=head_mask,
                            time_mask=time_mask,
                            seed=_stable_seed(seed, 1001, p_idx, d_idx, layer_idx),
                        )
                        delta_abs = _pair_norm(dK, dV)
                        clean_abs = _pair_norm(key, value)
                        delta_pair_norms.append(delta_abs)
                        delta_pair_rel_norms.append(delta_abs / (clean_abs + EPS0))
                        delta_k_rel_norms.append(
                            float(
                                torch.linalg.vector_norm(dK.float()).item()
                                / (torch.linalg.vector_norm(key.float()).item() + EPS0)
                            )
                        )
                        delta_v_rel_norms.append(
                            float(
                                torch.linalg.vector_norm(dV.float()).item()
                                / (torch.linalg.vector_norm(value.float()).item() + EPS0)
                            )
                        )
                        delta_cache[layer_idx] = (dK, dV)
                        mask_cache[layer_idx] = (head_mask, time_mask)
                    dK, dV = delta_cache[layer_idx]
                    return key + dK, value + dV

                z_corr = _forward_next_token_logits(
                    model,
                    input_ids,
                    patch_fn=patch_baseline,
                    protect_layers=layers,
                )
                samples["baseline"].append(
                    _logit_drift(z_clean, z_corr, topk_idx=topk_idx)
                )

                def patch_heuristic(
                    layer_idx: int,
                    q: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    del q
                    if layer_idx not in layer_set:
                        return key, value
                    if layer_idx not in delta_cache:
                        _ = patch_baseline(layer_idx, q, key, value)
                    dK, dV = delta_cache[layer_idx]
                    head_mask, time_mask = mask_cache[layer_idx]
                    Kp, Vp = key + dK, value + dV
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(_stable_seed(seed, 1002, p_idx, d_idx, layer_idx))
                    return _apply_heuristic(
                        best_heuristic,
                        K=Kp,
                        V=Vp,
                        head_mask=head_mask,
                        time_mask=time_mask,
                        baseline_cfg=baseline_cfg,
                        gen=gen,
                    )

                z_heur = _forward_next_token_logits(
                    model,
                    input_ids,
                    patch_fn=patch_heuristic,
                    protect_layers=layers,
                )
                samples["heuristics"].append(
                    _logit_drift(z_clean, z_heur, topk_idx=topk_idx)
                )

                def patch_repair(
                    modules: nn.ModuleDict,
                ) -> Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
                    def fn(
                        layer_idx: int,
                        q: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        if layer_idx not in layer_set:
                            return key, value
                        if layer_idx not in delta_cache:
                            _ = patch_baseline(layer_idx, q, key, value)
                        dK, dV = delta_cache[layer_idx]
                        Kp, Vp = key + dK, value + dV
                        module_key = f"layer_{layer_idx}"
                        if module_key not in modules:
                            return Kp, Vp
                        module = modules[module_key]
                        return module(q.detach(), Kp.detach(), Vp.detach())

                    return fn

                z_rep = _forward_next_token_logits(
                    model,
                    input_ids,
                    patch_fn=patch_repair(repair_modules),
                    protect_layers=layers,
                )
                cache_drift = _logit_drift(z_clean, z_rep, topk_idx=topk_idx)
                samples["cachemedic"].append(cache_drift)

                if repair_no_contr_modules is not None:
                    z_rep_no_contr = _forward_next_token_logits(
                        model,
                        input_ids,
                        patch_fn=patch_repair(repair_no_contr_modules),
                        protect_layers=layers,
                    )
                    samples["cachemedic_no_contr"].append(
                        _logit_drift(z_clean, z_rep_no_contr, topk_idx=topk_idx)
                    )

        curves.append(
            {
                "delta_norm": float(dn),
                "baseline": _mean(samples["baseline"]),
                "heuristics": _mean(samples["heuristics"]),
                "cachemedic": _mean(samples["cachemedic"]),
                "cachemedic_no_contr": _mean_or_none(samples["cachemedic_no_contr"]),
                "delta_pair_norm": _mean(delta_pair_norms),
                "delta_pair_rel_norm": _mean(delta_pair_rel_norms),
                "delta_k_rel_norm": _mean(delta_k_rel_norms),
                "delta_v_rel_norm": _mean(delta_v_rel_norms),
            }
        )
    return curves


def _compute_amplification_map(
    *,
    model: Any,
    prompts: list[torch.Tensor],
    clean_logits: list[torch.Tensor],
    topk_indices: list[torch.Tensor | None],
    layers: list[int],
    heads: list[int],
    num_directions: int,
    amp_delta_norm: float,
    seed: int,
    stability_cfg: dict[str, Any],
    corruption_cfg: dict[str, Any],
    repair_modules: nn.ModuleDict,
) -> tuple[list[list[float]], list[list[float]]]:
    amp_baseline: list[list[float]] = []
    amp_cachemedic: list[list[float]] = []

    for layer_idx in layers:
        row_base: list[float] = []
        row_rep: list[float] = []
        for head_idx in heads:
            ratios_base: list[float] = []
            ratios_rep: list[float] = []

            for p_idx, input_ids in enumerate(prompts):
                z_clean = clean_logits[p_idx]
                topk_idx = topk_indices[p_idx]

                for d_idx in range(num_directions):
                    delta_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                    denom_cache: dict[int, float] = {}

                    def patch_target(
                        layer_id: int,
                        q: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        del q
                        if layer_id != layer_idx:
                            return key, value
                        if layer_id not in delta_cache:
                            H, T = int(key.shape[1]), int(key.shape[2])
                            if not (0 <= head_idx < H):
                                delta_cache[layer_id] = (
                                    torch.zeros_like(key),
                                    torch.zeros_like(value),
                                )
                                denom_cache[layer_id] = EPS0
                                return key, value

                            head_mask = torch.zeros((H,), dtype=torch.bool)
                            head_mask[head_idx] = True
                            time_mask = _make_time_mask(
                                T,
                                stability_cfg=stability_cfg,
                                corruption_cfg=corruption_cfg,
                            )
                            dK, dV = _build_delta_pair(
                                K_clean=key.detach(),
                                V_clean=value.detach(),
                                delta_norm=amp_delta_norm,
                                head_mask=head_mask,
                                time_mask=time_mask,
                                seed=_stable_seed(
                                    seed,
                                    2001,
                                    p_idx,
                                    d_idx,
                                    layer_idx,
                                    head_idx,
                                ),
                            )
                            delta_cache[layer_id] = (dK, dV)
                            denom_cache[layer_id] = max(_pair_norm(dK, dV), EPS0)
                        dK, dV = delta_cache[layer_id]
                        return key + dK, value + dV

                    z_corr = _forward_next_token_logits(
                        model,
                        input_ids,
                        patch_fn=patch_target,
                        protect_layers=[layer_idx],
                    )
                    denom = denom_cache.get(layer_idx, EPS0)
                    ratios_base.append(_logit_drift(z_clean, z_corr, topk_idx=topk_idx) / denom)

                    def patch_repair(
                        layer_id: int,
                        q: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        if layer_id != layer_idx:
                            return key, value
                        if layer_id not in delta_cache:
                            _ = patch_target(layer_id, q, key, value)
                        dK, dV = delta_cache[layer_id]
                        Kp, Vp = key + dK, value + dV
                        module_key = f"layer_{layer_id}"
                        if module_key not in repair_modules:
                            return Kp, Vp
                        module = repair_modules[module_key]
                        return module(q.detach(), Kp.detach(), Vp.detach())

                    z_rep = _forward_next_token_logits(
                        model,
                        input_ids,
                        patch_fn=patch_repair,
                        protect_layers=[layer_idx],
                    )
                    ratios_rep.append(_logit_drift(z_clean, z_rep, topk_idx=topk_idx) / denom)

            row_base.append(_median(ratios_base))
            row_rep.append(_median(ratios_rep))

        amp_baseline.append(row_base)
        amp_cachemedic.append(row_rep)

    return amp_baseline, amp_cachemedic


def run_stability(
    config_path: Path,
    run_dir: Path,
    *,
    resume: bool = True,
    heartbeat_every: int = 1,
) -> dict[str, Any]:
    repo_root = resolve_repo_root(config_path.parent)
    cfg = load_and_validate_config(config_path, repo_root=repo_root)
    enforce_v1_constraints(cfg, batch_size=1)
    ensure_run_layout(run_dir)
    if int(heartbeat_every) < 1:
        raise ValueError("heartbeat_every must be >= 1.")

    run_record = load_run_record(run_dir)
    run_id = str(run_record.get("run_id")) if run_record else run_dir.name
    config_hash = stable_json_hash(cfg)
    progress_path = _stability_progress_file(run_dir)
    pause_request_path = _pause_request_path(run_dir)

    delta_norms = [float(x) for x in cfg["stability"]["delta_norms"]]
    num_prompts = int(cfg["stability"]["num_prompts"])
    num_directions = int(cfg["stability"]["num_directions"])
    topk_logits = int(cfg["stability"]["topk_logits"])
    layers = [int(x) for x in cfg["stability"]["layers"]]
    seed = int(cfg["seed"])
    total_units = len(delta_norms) + len(layers)

    if resume and (run_dir / "metrics" / "stability_metrics.json").exists() and not progress_path.exists():
        _emit_stability_heartbeat(
            run_dir,
            _make_stability_heartbeat(
                run_id=run_id,
                status="completed",
                step=total_units,
                total_steps=total_units,
                elapsed_wall_sec=0.0,
                reason="already_completed",
                resumed_from=None,
                current_unit=None,
            ),
        )
        return {"run_id": run_id, "status": "completed", "already_completed": True}

    device = _select_device(str(cfg["device"]))
    model, tokenizer, geom, runtime_dtype = load_model_and_tokenizer(cfg, device=device)

    if not layers:
        raise ValueError("stability.layers must not be empty.")
    if max(layers) >= geom.num_layers:
        raise ValueError(
            f"stability.layers contains layer index >= num_layers ({geom.num_layers})."
        )

    head_subset = cfg["stability"].get("head_subset")
    if head_subset is not None:
        heads = [int(h) for h in head_subset]
    else:
        heads = list(range(int(geom.num_heads)))
    if not heads:
        raise ValueError("stability.head_subset resolved to an empty head list.")
    if max(heads) >= geom.num_heads:
        raise ValueError(
            f"stability.head_subset contains head index >= num_heads ({geom.num_heads})."
        )

    protect_layers = [int(x) for x in cfg["repair"]["protect_layers"]]
    repair_modules = _build_repair_modules(
        cfg,
        geom,
        protect_layers,
        device=device,
        dtype=runtime_dtype,
    )
    checkpoint_loaded = _load_repair_state(run_dir, repair_modules)

    # If this run is already lambda_contr=0, the run itself is the ablation curve.
    if float(cfg["train"]["lambda_contr"]) == 0.0:
        repair_no_contr_modules: nn.ModuleDict | None = repair_modules
        no_contr_source = "self"
    else:
        pair_dir = _resolve_no_contr_pair_run_dir(run_dir, cfg)
        if pair_dir is None:
            repair_no_contr_modules = None
            no_contr_source = "missing_pair"
        else:
            repair_no_contr_modules = _build_repair_modules(
                cfg,
                geom,
                protect_layers,
                device=device,
                dtype=runtime_dtype,
            )
            if _load_repair_state(pair_dir, repair_no_contr_modules):
                no_contr_source = str(pair_dir)
            else:
                repair_no_contr_modules = None
                no_contr_source = "pair_checkpoint_missing"

    best_heuristic = _resolve_best_heuristic(run_dir)
    max_seq_len = int(cfg["eval"]["max_seq_len"])
    texts = load_wikitext2(cfg["data"], num_examples=max(num_prompts, 64), repo_root=repo_root)
    prompts = _prepare_prompt_tensors(
        tokenizer,
        texts,
        num_prompts=num_prompts,
        max_seq_len=max_seq_len,
        device=device,
    )

    clean_logits: list[torch.Tensor] = []
    topk_indices: list[torch.Tensor | None] = []
    for input_ids in prompts:
        z_clean = _forward_next_token_logits(
            model,
            input_ids,
            patch_fn=None,
            protect_layers=layers,
        )
        clean_logits.append(z_clean)
        topk_indices.append(_topk_index(z_clean, topk_logits=topk_logits))
    progress = _load_stability_progress(run_dir) if resume else None
    curve_points: dict[str, dict[str, Any]] = {}
    amp_rows: dict[int, dict[str, Any]] = {}
    resumed_from: str | None = None
    elapsed_before = 0.0
    created_utc = utc_now_iso()
    if progress is not None:
        saved_hash = str(progress.get("config_hash", ""))
        if saved_hash and saved_hash != config_hash:
            raise ValueError(
                f"Config hash mismatch for stability resume. state={saved_hash} current={config_hash}"
            )
        for point in progress.get("curve_points", []):
            if not isinstance(point, dict):
                continue
            if "delta_norm" not in point:
                continue
            curve_points[_delta_key(float(point["delta_norm"]))] = point
        for row in progress.get("amp_rows", []):
            if not isinstance(row, dict):
                continue
            if "layer" not in row:
                continue
            amp_rows[int(row["layer"])] = row
        resumed_from = str(progress_path)
        elapsed_before = float(progress.get("elapsed_wall_sec", 0.0))
        created_utc = str(progress.get("created_utc", created_utc))

        if (
            str(progress.get("status", "")) == "completed"
            and (run_dir / "metrics" / "stability_metrics.json").exists()
        ):
            _emit_stability_heartbeat(
                run_dir,
                _make_stability_heartbeat(
                    run_id=run_id,
                    status="completed",
                    step=total_units,
                    total_steps=total_units,
                    elapsed_wall_sec=elapsed_before,
                    reason="already_completed",
                    resumed_from=resumed_from,
                    current_unit=None,
                ),
            )
            return {
                "run_id": run_id,
                "status": "completed",
                "already_completed": True,
                "resumed": True,
            }

    start_wall = time.perf_counter()

    def persist_progress(
        *,
        status: str,
        reason: str,
        current_unit: dict[str, Any] | None,
    ) -> float:
        elapsed_total = elapsed_before + (time.perf_counter() - start_wall)
        ordered_curve_points = [
            curve_points[_delta_key(dn)] for dn in delta_norms if _delta_key(dn) in curve_points
        ]
        ordered_amp_rows = [amp_rows[layer] for layer in layers if layer in amp_rows]
        payload = {
            "phase": "stability",
            "run_id": run_id,
            "config_hash": config_hash,
            "status": status,
            "reason": reason,
            "created_utc": created_utc,
            "updated_utc": utc_now_iso(),
            "elapsed_wall_sec": float(elapsed_total),
            "total_units": total_units,
            "completed_units": len(curve_points) + len(amp_rows),
            "resumed_from": resumed_from,
            "curve_points": ordered_curve_points,
            "amp_rows": ordered_amp_rows,
            "settings": {
                "delta_norms": delta_norms,
                "layers": layers,
                "heads": heads,
                "num_prompts": num_prompts,
                "num_directions": num_directions,
                "topk_logits": topk_logits,
            },
        }
        _save_stability_progress(run_dir, payload)
        _emit_stability_heartbeat(
            run_dir,
            _make_stability_heartbeat(
                run_id=run_id,
                status=status,
                step=len(curve_points) + len(amp_rows),
                total_steps=total_units,
                elapsed_wall_sec=elapsed_total,
                reason=reason,
                resumed_from=resumed_from,
                current_unit=current_unit,
            ),
        )
        return elapsed_total

    def pause_and_return(reason: str, current_unit: dict[str, Any] | None) -> dict[str, Any]:
        pause_request_path.unlink(missing_ok=True)
        elapsed_total = persist_progress(status="paused", reason=reason, current_unit=current_unit)
        write_json(
            run_dir / "control" / "pause_ack_stability.json",
            {
                "phase": "stability",
                "run_id": run_id,
                "status": "paused",
                "reason": reason,
                "timestamp_utc": utc_now_iso(),
                "resume_state": str(progress_path),
                "completed_units": len(curve_points) + len(amp_rows),
                "total_units": total_units,
                "elapsed_wall_sec": float(elapsed_total),
            },
        )
        return {
            "run_id": run_id,
            "status": "paused",
            "pause_reason": reason,
            "resume_state": str(progress_path),
            "completed_units": len(curve_points) + len(amp_rows),
            "total_units": total_units,
            "resumed": resumed_from is not None,
        }

    startup_reason = "startup_resume" if resumed_from is not None else "startup"
    startup_status = "resumed" if resumed_from is not None else "running"
    _emit_stability_heartbeat(
        run_dir,
        _make_stability_heartbeat(
            run_id=run_id,
            status=startup_status,
            step=len(curve_points) + len(amp_rows),
            total_steps=total_units,
            elapsed_wall_sec=elapsed_before,
            reason=startup_reason,
            resumed_from=resumed_from,
            current_unit=None,
        ),
    )

    if (len(curve_points) + len(amp_rows)) >= total_units and (
        run_dir / "metrics" / "stability_metrics.json"
    ).exists():
        persist_progress(status="completed", reason="already_completed", current_unit=None)
        return {
            "run_id": run_id,
            "status": "completed",
            "already_completed": True,
            "resumed": resumed_from is not None,
        }

    units_since_heartbeat = 0
    for dn in delta_norms:
        key = _delta_key(dn)
        if key in curve_points:
            continue
        if _pause_requested(run_dir):
            return pause_and_return("pause_request", {"kind": "curve", "delta_norm": float(dn)})
        point = _compute_logit_sensitivity(
            model=model,
            prompts=prompts,
            clean_logits=clean_logits,
            topk_indices=topk_indices,
            layers=layers,
            delta_norms=[float(dn)],
            num_directions=num_directions,
            seed=seed,
            stability_cfg=cfg["stability"],
            corruption_cfg=cfg["corruption"],
            baseline_cfg=cfg["baseline"],
            best_heuristic=best_heuristic,
            repair_modules=repair_modules,
            repair_no_contr_modules=repair_no_contr_modules,
        )[0]
        curve_points[key] = point
        units_since_heartbeat += 1
        if units_since_heartbeat >= int(heartbeat_every):
            persist_progress(
                status="running",
                reason="curve_progress",
                current_unit={"kind": "curve", "delta_norm": float(dn)},
            )
            units_since_heartbeat = 0

    positive = [x for x in delta_norms if x > 0]
    amp_delta_norm = positive[0] if positive else 1.0
    for layer in layers:
        if int(layer) in amp_rows:
            continue
        if _pause_requested(run_dir):
            return pause_and_return("pause_request", {"kind": "amplification", "layer": int(layer)})
        amp_base_rows, amp_rep_rows = _compute_amplification_map(
            model=model,
            prompts=prompts,
            clean_logits=clean_logits,
            topk_indices=topk_indices,
            layers=[int(layer)],
            heads=heads,
            num_directions=num_directions,
            amp_delta_norm=amp_delta_norm,
            seed=seed,
            stability_cfg=cfg["stability"],
            corruption_cfg=cfg["corruption"],
            repair_modules=repair_modules,
        )
        amp_rows[int(layer)] = {
            "layer": int(layer),
            "baseline": amp_base_rows[0],
            "cachemedic": amp_rep_rows[0],
        }
        units_since_heartbeat += 1
        if units_since_heartbeat >= int(heartbeat_every):
            persist_progress(
                status="running",
                reason="amplification_progress",
                current_unit={"kind": "amplification", "layer": int(layer)},
            )
            units_since_heartbeat = 0

    persist_progress(status="running", reason="finalizing", current_unit=None)
    curves = [curve_points[_delta_key(dn)] for dn in delta_norms]
    amp_baseline = [amp_rows[int(layer)]["baseline"] for layer in layers]
    amp_cachemedic = [amp_rows[int(layer)]["cachemedic"] for layer in layers]

    out = {
        "run_id": run_id,
        "config_hash": config_hash,
        "metrics": {
            "stability.logit_sensitivity": curves,
            "stability.amplification_map": {
                "layers": layers,
                "heads": heads,
                "baseline": amp_baseline,
                "cachemedic": amp_cachemedic,
            },
            "stability.settings": {
                "num_prompts": num_prompts,
                "num_directions": num_directions,
                "topk_logits": topk_logits,
                "time_mode": str(cfg["stability"]["time_mode"]),
                "N_recent": int(cfg["stability"]["N_recent"]),
                "amplification_delta_norm": float(amp_delta_norm),
                "best_heuristic_name": best_heuristic,
                "checkpoint_loaded": checkpoint_loaded,
                "cachemedic_no_contr_source": no_contr_source,
                "delta_injection_shared_across_variants": True,
            },
        },
        "settings": {
            "delta_norms": delta_norms,
            "layers": layers,
            "heads": heads,
        },
    }
    write_json(run_dir / "metrics" / "stability_metrics.json", out)
    elapsed_total = persist_progress(status="completed", reason="completed", current_unit=None)
    return {
        "run_id": run_id,
        "status": "completed",
        "points": len(curves),
        "layers": len(layers),
        "heads": len(heads),
        "best_heuristic": best_heuristic,
        "checkpoint_loaded": checkpoint_loaded,
        "elapsed_wall_sec": elapsed_total,
        "resumed": resumed_from is not None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute CacheMedic++ stability metrics.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--run_dir", required=True, help="Output run directory.")
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume stability from metrics/stability_progress.json if available (default).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore stability progress state and start from scratch.",
    )
    p.add_argument(
        "--heartbeat_every",
        type=int,
        default=1,
        help="Emit stability heartbeat/progress after this many completed units (default: 1).",
    )
    p.set_defaults(resume=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_stability(
        Path(args.config).resolve(),
        Path(args.run_dir).resolve(),
        resume=bool(args.resume),
        heartbeat_every=int(args.heartbeat_every),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
