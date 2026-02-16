"""CLI to pre-download model artifacts used by CacheMedic++ experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml, resolve_repo_root

DEFAULT_FAST_CONFIGS = [
    "configs/mve.yaml",
    "configs/second_model_smoke_gpt2_large.yaml",
]


@dataclass(frozen=True)
class ModelRef:
    name: str
    revision: str | None = None


def _resolve_config_path(ref: str, repo_root: Path) -> Path:
    p = Path(ref)
    if p.exists():
        return p.resolve()
    cand = (repo_root / ref).resolve()
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Unable to resolve config path: {ref}")


def _model_ref_from_config(cfg: dict[str, Any]) -> ModelRef:
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ValueError("Config is missing a 'model' object.")
    name = str(model.get("name", "")).strip()
    if not name:
        raise ValueError("Config model.name must be non-empty.")
    revision_raw = model.get("revision")
    revision = str(revision_raw).strip() if isinstance(revision_raw, str) else None
    if revision == "":
        revision = None
    return ModelRef(name=name, revision=revision)


def _parse_model_spec(spec: str) -> ModelRef:
    raw = spec.strip()
    if not raw:
        raise ValueError("Model spec must be non-empty.")
    if "@" in raw:
        name, revision = raw.split("@", 1)
        name = name.strip()
        revision = revision.strip() or None
        if not name:
            raise ValueError(f"Invalid model spec: {spec!r}")
        return ModelRef(name=name, revision=revision)
    return ModelRef(name=raw, revision=None)


def _dedupe_model_refs(refs: list[ModelRef]) -> list[ModelRef]:
    out: list[ModelRef] = []
    seen: set[tuple[str, str | None]] = set()
    for ref in refs:
        key = (ref.name, ref.revision)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def collect_model_refs(
    *,
    config_refs: list[str],
    model_specs: list[str],
    repo_root: Path,
) -> list[ModelRef]:
    refs: list[ModelRef] = []
    for cfg_ref in config_refs:
        cfg_path = _resolve_config_path(cfg_ref, repo_root)
        cfg = load_yaml(cfg_path)
        refs.append(_model_ref_from_config(cfg))
    for spec in model_specs:
        refs.append(_parse_model_spec(spec))
    return _dedupe_model_refs(refs)


def resolve_prefetch_targets(
    *,
    config_refs: list[str],
    model_specs: list[str],
    repo_root: Path,
) -> list[ModelRef]:
    selected_configs = list(config_refs)
    if not selected_configs and not model_specs:
        selected_configs = list(DEFAULT_FAST_CONFIGS)
    return collect_model_refs(
        config_refs=selected_configs,
        model_specs=model_specs,
        repo_root=repo_root,
    )


def prefetch_models(
    refs: list[ModelRef],
    *,
    cache_dir: Path | None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency issue path
        raise RuntimeError(
            "huggingface_hub is required for prefetch. Install dependencies first."
        ) from exc

    if not refs:
        raise ValueError("No models selected for prefetch.")

    out: list[dict[str, Any]] = []
    for ref in refs:
        kwargs: dict[str, Any] = {
            "repo_id": ref.name,
            "revision": ref.revision,
            "local_files_only": bool(local_files_only),
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        local_path = snapshot_download(**kwargs)
        rec = {
            "name": ref.name,
            "revision": ref.revision,
            "local_path": str(local_path),
        }
        out.append(rec)
        print(
            f"[prefetch] model={ref.name} revision={ref.revision or 'default'} path={local_path}"
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefetch model artifacts for CacheMedic++ runs.")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Config YAML to read model.name/model.revision from. "
            "May be provided multiple times."
        ),
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model repo id, optionally with '@revision' suffix. "
            "May be provided multiple times."
        ),
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory for downloads.",
    )
    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not use network; only resolve from already-cached local files.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved model targets without downloading.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = resolve_repo_root()
    refs = resolve_prefetch_targets(
        config_refs=[str(x) for x in args.config],
        model_specs=[str(x) for x in args.model],
        repo_root=repo_root,
    )
    if args.dry_run:
        for ref in refs:
            print(f"[prefetch:dry-run] model={ref.name} revision={ref.revision or 'default'}")
        return 0

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    prefetch_models(
        refs,
        cache_dir=cache_dir,
        local_files_only=bool(args.local_files_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
