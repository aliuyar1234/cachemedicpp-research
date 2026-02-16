"""Configuration loading, validation, hashing, and override utilities."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def resolve_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward until ARCHITECTURE.md exists."""
    starts = []
    if start is not None:
        starts.append(start.resolve())
    starts.append(Path.cwd().resolve())
    starts.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for begin in starts:
        if begin in seen:
            continue
        seen.add(begin)
        for path in [begin, *begin.parents]:
            if (path / "ARCHITECTURE.md").exists():
                return path
    raise RuntimeError("Unable to resolve repository root (ARCHITECTURE.md not found).")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must parse to an object: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON file must parse to an object: {path}")
    return obj


def validate_with_schema(instance: dict[str, Any], schema: dict[str, Any], name: str) -> None:
    try:
        from jsonschema import Draft7Validator  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency issue path
        raise RuntimeError(
            "jsonschema is required for config validation. Install dependencies first."
        ) from exc

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    if not errors:
        return

    lines = []
    for err in errors[:20]:
        location = "/".join(str(x) for x in err.path)
        if location:
            lines.append(f"{name}:{location}: {err.message}")
        else:
            lines.append(f"{name}: {err.message}")
    raise ValueError("Config validation failed:\n" + "\n".join(lines))


def stable_json_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def apply_dot_override(config: dict[str, Any], dot_key: str, value: Any) -> None:
    parts = dot_key.split(".")
    if not parts:
        raise ValueError("override key must not be empty")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        next_node = node[part]
        if not isinstance(next_node, dict):
            raise TypeError(
                f"Cannot apply override '{dot_key}': '{part}' is not a mapping in config."
            )
        node = next_node
    node[parts[-1]] = value


def apply_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base_config)
    for key in sorted(overrides.keys()):
        apply_dot_override(merged, key, overrides[key])
    return merged


def load_and_validate_config(config_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or resolve_repo_root(config_path.parent)
    config = load_yaml(config_path)
    schema = load_json(root / "schemas" / "config.schema.json")
    validate_with_schema(config, schema, str(config_path))
    return config
