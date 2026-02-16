"""Run directory layout and artifact IO helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import dump_yaml, stable_json_hash


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_run_layout(run_dir: Path) -> None:
    for rel in [
        "checkpoints",
        "control",
        "logs",
        "metrics",
        "paper/figures",
        "paper/tables",
    ]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")


def capture_env() -> dict[str, str]:
    torch_ver = "not-installed"
    transformers_ver = "not-installed"
    cuda_name = "cpu"
    cuda_ver = "none"
    try:
        import torch  # type: ignore

        torch_ver = str(torch.__version__)
        if torch.cuda.is_available():
            cuda_ver = str(torch.version.cuda)
            cuda_name = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    try:
        import transformers  # type: ignore

        transformers_ver = str(transformers.__version__)
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "torch": torch_ver,
        "transformers": transformers_ver,
        "cuda": cuda_ver,
        "gpu_name": cuda_name,
        "platform": platform.platform(),
    }


def capture_git(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"]) or "unknown"
    dirty = bool(run_git(["status", "--porcelain"]))
    return {"commit": commit, "dirty": dirty}


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    run_dir: Path
    config_path: str
    config_hash: str
    command: str


def build_run_identity(run_dir: Path, config_obj: dict[str, Any], config_path: Path) -> RunIdentity:
    run_id = run_dir.name
    return RunIdentity(
        run_id=run_id,
        run_dir=run_dir,
        config_path=str(config_path).replace("\\", "/"),
        config_hash=stable_json_hash(config_obj),
        command=" ".join(sys.argv),
    )


def write_run_metadata(
    *,
    repo_root: Path,
    identity: RunIdentity,
    config_obj: dict[str, Any],
) -> None:
    ensure_run_layout(identity.run_dir)
    dump_yaml(identity.run_dir / "config_resolved.yaml", config_obj)

    env = capture_env()
    git = capture_git(repo_root)
    write_json(identity.run_dir / "env.json", env)
    write_json(identity.run_dir / "git.json", git)

    run_record = {
        "run_id": identity.run_id,
        "created_utc": utc_now_iso(),
        "command": identity.command,
        "config_path": identity.config_path,
        "config_hash": identity.config_hash,
        "git": git,
        "env": {
            "python": env.get("python", "unknown"),
            "torch": env.get("torch", "unknown"),
            "transformers": env.get("transformers", "unknown"),
            "cuda": env.get("cuda", "unknown"),
            "gpu_name": env.get("gpu_name", "unknown"),
        },
    }
    write_json(identity.run_dir / "run_record.json", run_record)


def load_run_record(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_record.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
