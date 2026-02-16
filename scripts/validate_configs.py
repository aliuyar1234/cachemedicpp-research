#!/usr/bin/env python3
"""Validate shipped YAML configs against JSON Schemas.

This is a harness integrity check: configs must match schemas, and schemas must
match the SSOT specs.

Checks performed:
- Parse JSON schemas.
- Validate base configs (mve, second_model_smoke) against config.schema.json.
- Validate sweep config (full_gpt2_matrix) against sweep.schema.json.
- Verify referenced paths exist (base_config, corruption mix file).

This script runs without any ML code.
"""

import json
from pathlib import Path

import yaml
from jsonschema import Draft7Validator


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate(schema, instance, name: str) -> list[str]:
    v = Draft7Validator(schema)
    errors = sorted(v.iter_errors(instance), key=lambda e: e.path)
    msgs = []
    for e in errors:
        loc = "/".join(str(x) for x in e.path)
        if loc:
            msgs.append(f"{name}: {loc}: {e.message}")
        else:
            msgs.append(f"{name}: {e.message}")
    return msgs


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    schema_config = load_json(root / "schemas" / "config.schema.json")
    schema_sweep = load_json(root / "schemas" / "sweep.schema.json")

    bad = []

    # Base configs
    for cfg_name in ["mve.yaml", "second_model_smoke.yaml"]:
        cfg_path = root / "configs" / cfg_name
        cfg = load_yaml(cfg_path)
        bad.extend(validate(schema_config, cfg, f"configs/{cfg_name}"))

        # Referenced corruption mixture file should exist.
        mix = cfg.get("corruption", {}).get("mixture")
        mix_path = root / "configs" / "corruption" / f"{mix}.yaml"
        if not mix_path.exists():
            bad.append(f"configs/{cfg_name}: corruption.mixture refers to missing file: {mix_path}")

    # Sweep config
    sweep_path = root / "configs" / "full_gpt2_matrix.yaml"
    sweep = load_yaml(sweep_path)
    bad.extend(validate(schema_sweep, sweep, "configs/full_gpt2_matrix.yaml"))

    base_cfg = sweep.get("base_config")
    if base_cfg:
        base_path = (root / base_cfg).resolve()
        if not base_path.exists():
            bad.append(f"configs/full_gpt2_matrix.yaml: base_config path does not exist: {base_cfg}")

    if bad:
        for m in bad:
            print(f"CONFIG_SCHEMA_FAIL {m}")
        return 2

    print("OK: configs validate against schemas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
