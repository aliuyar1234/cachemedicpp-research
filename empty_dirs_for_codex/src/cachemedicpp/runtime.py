"""Runtime helpers for loading models/tokenizers in CacheMedic++."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ModelGeometry:
    num_layers: int
    num_heads: int
    head_dim: int


def resolve_runtime_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping.get(dtype_name, torch.float32)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _maybe_get_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _candidate_hf_homes() -> list[Path]:
    out: list[Path] = []
    preferred = os.environ.get("CACHEMEDICPP_HF_HOME")
    if preferred:
        out.append(Path(preferred).expanduser())
    for drive in ("E", "D", "C"):
        out.append(Path(f"{drive}:/Model/huggingface"))
    return out


def _autoconfigure_local_hf_home() -> None:
    managed_keys = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    )
    if any(os.environ.get(k) for k in managed_keys):
        return

    for candidate in _candidate_hf_homes():
        if not candidate.exists():
            continue
        os.environ["HF_HOME"] = str(candidate)
        hub_cache = candidate / "hub"
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
        return


def _resolve_hf_cache_dir() -> Path | None:
    _autoconfigure_local_hf_home()

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return None


def extract_gpt2_geometry(model: Any) -> ModelGeometry:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise TypeError("CacheMedic++ v1 currently supports GPT-2 style models with transformer.h.")
    blocks = list(model.transformer.h)
    if not blocks:
        raise RuntimeError("Model has no transformer blocks.")
    attn = blocks[0].attn
    num_heads = _maybe_get_attr(attn, ["num_heads", "n_head"])
    head_dim = _maybe_get_attr(attn, ["head_dim"])
    if num_heads is None or head_dim is None:
        # Fallback from config values.
        n_embd = _maybe_get_attr(model.config, ["n_embd", "hidden_size"])
        n_head = _maybe_get_attr(model.config, ["n_head", "num_attention_heads"])
        if n_embd is None or n_head is None:
            raise RuntimeError("Unable to infer attention geometry from model.")
        num_heads = int(n_head)
        head_dim = int(int(n_embd) // int(n_head))
    return ModelGeometry(num_layers=len(blocks), num_heads=int(num_heads), head_dim=int(head_dim))


def load_model_and_tokenizer(
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[Any, Any, ModelGeometry, torch.dtype]:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency issue path
        raise RuntimeError(
            "transformers is required for real GPT-2 hooking/evaluation. Install dependencies first."
        ) from exc

    model_name = str(cfg["model"]["name"])
    revision = cfg["model"].get("revision")
    runtime_dtype = resolve_runtime_dtype(str(cfg["dtype"]), device)
    cache_dir = _resolve_hf_cache_dir()
    cache_dir_str = str(cache_dir) if cache_dir is not None else None

    config_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": True,
    }
    if cache_dir_str is not None:
        config_kwargs["cache_dir"] = cache_dir_str

    hf_cfg = AutoConfig.from_pretrained(model_name, **config_kwargs)
    if hasattr(hf_cfg, "_attn_implementation"):
        setattr(hf_cfg, "_attn_implementation", "eager")
    if hasattr(hf_cfg, "attn_implementation"):
        setattr(hf_cfg, "attn_implementation", "eager")
    if hasattr(hf_cfg, "use_cache"):
        setattr(hf_cfg, "use_cache", bool(cfg["model"]["use_cache"]))

    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "config": hf_cfg,
        "trust_remote_code": True,
    }
    if cache_dir_str is not None:
        model_kwargs["cache_dir"] = cache_dir_str
    if runtime_dtype != torch.float32 or device.type == "cuda":
        model_kwargs["torch_dtype"] = runtime_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    attn_impl = getattr(model.config, "_attn_implementation", None)
    if attn_impl not in (None, "eager"):
        raise ValueError(
            f"CacheMedic++ v1 fail-closed: eager attention required, found '{attn_impl}'."
        )

    tok_kwargs: dict[str, Any] = {
        "revision": revision,
        "use_fast": True,
        "trust_remote_code": True,
    }
    if cache_dir_str is not None:
        tok_kwargs["cache_dir"] = cache_dir_str

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    geom = extract_gpt2_geometry(model)
    return model, tokenizer, geom, runtime_dtype
