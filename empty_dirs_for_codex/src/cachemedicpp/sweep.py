"""CLI for CacheMedic++ sweep execution."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .config import (
    apply_overrides,
    dump_yaml,
    load_and_validate_config,
    load_json,
    load_yaml,
    resolve_repo_root,
    validate_with_schema,
)
from .corruption import renormalize_weights
from .eval import run_eval
from .plots import generate_plots
from .stability import run_stability
from .train import run_train


def _resolve_config_path(ref: str, repo_root: Path) -> Path:
    p = Path(ref)
    if p.exists():
        return p.resolve()
    cand = (repo_root / ref).resolve()
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Unable to resolve path: {ref}")


def _load_ood_split(split_name: str, cfg: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    splits_ref = cfg["eval"].get("ood_splits_path") or "configs/corruption/ood_splits.yaml"
    splits_path = _resolve_config_path(str(splits_ref), repo_root)
    data = yaml.safe_load(splits_path.read_text(encoding="utf-8"))
    for entry in data.get("splits", []):
        if str(entry.get("name")) == split_name:
            return entry
    raise ValueError(f"OOD protocol split not found: {split_name}")


def _resolve_mixture_yaml(mixture_ref: str, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(mixture_ref)
    if candidate.exists():
        path = candidate.resolve()
    elif (repo_root / mixture_ref).exists():
        path = (repo_root / mixture_ref).resolve()
    else:
        path = (repo_root / "configs" / "corruption" / f"{mixture_ref}.yaml").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mixture YAML not found: {mixture_ref}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return path, payload


def _apply_ood_wiring(run_cfg: dict[str, Any], run_dir: Path, repo_root: Path) -> dict[str, Any]:
    protocol = run_cfg["eval"].get("ood_protocol")
    if protocol in (None, "null"):
        return run_cfg

    split = _load_ood_split(str(protocol), run_cfg, repo_root)
    train_types = [str(x) for x in split["train_types"]]
    test_type = str(split["test_type"])

    _, mix_payload = _resolve_mixture_yaml(str(run_cfg["corruption"]["mixture"]), repo_root)
    weights = {str(k): float(v) for k, v in mix_payload["weights"].items()}
    mix_payload["weights"] = renormalize_weights(weights, train_types)
    mix_payload["name"] = f"ood_train_mix_{protocol}"
    ood_mix_path = run_dir / "ood_train_mix.yaml"
    ood_mix_path.parent.mkdir(parents=True, exist_ok=True)
    ood_mix_path.write_text(yaml.safe_dump(mix_payload, sort_keys=False), encoding="utf-8")

    out = copy.deepcopy(run_cfg)
    out["corruption"]["mixture"] = str(ood_mix_path)
    out["eval"]["corruption_eval"]["types"] = [test_type]
    return out


def run_sweep(sweep_config_path: Path, out_root: Path, *, resume: bool = True) -> dict[str, Any]:
    repo_root = resolve_repo_root(sweep_config_path.parent)
    sweep_schema = load_json(repo_root / "schemas" / "sweep.schema.json")
    sweep_cfg = load_yaml(sweep_config_path)
    validate_with_schema(sweep_cfg, sweep_schema, str(sweep_config_path))

    base_config_path = _resolve_config_path(str(sweep_cfg["base_config"]), repo_root)
    base_cfg = load_and_validate_config(base_config_path, repo_root=repo_root)

    sweep_name = str(sweep_cfg["sweep_name"])
    sweep_dir = out_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_pause_path = sweep_dir / "control" / "pause.request"
    sweep_pause_path.parent.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    sweep_status = "completed"
    paused_run: str | None = None
    for run_item in sweep_cfg["runs"]:
        run_name = str(run_item["name"])
        overrides = dict(run_item.get("overrides", {}))
        merged = apply_overrides(base_cfg, overrides)

        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        merged = _apply_ood_wiring(merged, run_dir, repo_root)

        run_cfg_path = run_dir / "sweep_resolved_config.yaml"
        dump_yaml(run_cfg_path, merged)

        if sweep_pause_path.exists():
            sweep_pause_path.unlink(missing_ok=True)
            run_summaries.append(
                {
                    "name": run_name,
                    "run_dir": str(run_dir),
                    "overrides": overrides,
                    "train": {"status": "skipped", "reason": "sweep_pause_request"},
                    "eval": {"status": "skipped", "reason": "sweep_pause_request"},
                    "stability": {"status": "skipped", "reason": "sweep_pause_request"},
                    "plots": {"status": "skipped", "reason": "sweep_pause_request"},
                }
            )
            sweep_status = "paused"
            paused_run = run_name
            break

        train_out = run_train(run_cfg_path, run_dir, resume=bool(resume))
        train_status = str(train_out.get("status", "completed"))
        if train_status != "completed":
            eval_out = {"status": "skipped", "reason": f"train_{train_status}"}
            stability_out = {"status": "skipped", "reason": f"train_{train_status}"}
            plot_out = {"status": "skipped", "reason": f"train_{train_status}"}
            sweep_status = train_status
            paused_run = run_name
        else:
            eval_out = run_eval(run_cfg_path, run_dir, resume=bool(resume))
            eval_status = str(eval_out.get("status", "completed"))
            if eval_status != "completed":
                stability_out = {"status": "skipped", "reason": f"eval_{eval_status}"}
                plot_out = {"status": "skipped", "reason": f"eval_{eval_status}"}
                sweep_status = eval_status
                paused_run = run_name
            else:
                stability_out = run_stability(run_cfg_path, run_dir, resume=bool(resume))
                stability_status = str(stability_out.get("status", "completed"))
                if stability_status != "completed":
                    plot_out = {"status": "skipped", "reason": f"stability_{stability_status}"}
                    sweep_status = stability_status
                    paused_run = run_name
                else:
                    plot_out = generate_plots(run_dir)

        run_summaries.append(
            {
                "name": run_name,
                "run_dir": str(run_dir),
                "overrides": overrides,
                "train": train_out,
                "eval": eval_out,
                "stability": stability_out,
                "plots": plot_out,
            }
        )
        if str(run_summaries[-1]["train"].get("status", "completed")) != "completed":
            break
        if str(run_summaries[-1]["eval"].get("status", "completed")) != "completed":
            break
        if str(run_summaries[-1]["stability"].get("status", "completed")) != "completed":
            break

    summary = {
        "sweep_name": sweep_name,
        "base_config": str(base_config_path),
        "num_runs": len(run_summaries),
        "runs": run_summaries,
        "budget": sweep_cfg.get("budget"),
        "status": sweep_status,
        "paused_run": paused_run,
    }
    summary_path = out_root / f"{sweep_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CacheMedic++ config sweep.")
    p.add_argument("--sweep_config", required=True, help="Sweep YAML path.")
    p.add_argument("--out_root", required=True, help="Output root directory.")
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume per-run phases from available state/checkpoints when possible (default).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume behavior for run phases.",
    )
    p.set_defaults(resume=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_sweep(
        Path(args.sweep_config).resolve(),
        Path(args.out_root).resolve(),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
