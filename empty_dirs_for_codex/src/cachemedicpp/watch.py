"""CLI for live training heartbeat monitoring."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"completed", "paused", "failed"}
HEARTBEAT_PHASES = ["train", "eval", "stability", "sweep"]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    try:
        payload = json.loads(last_line)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_ts(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return datetime.fromtimestamp(0, tz=timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = datetime.fromisoformat(text)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def load_heartbeat(run_dir: Path) -> dict[str, Any] | None:
    logs = run_dir / "logs"
    candidates: list[dict[str, Any]] = []
    for phase in HEARTBEAT_PHASES:
        snap = _read_json(logs / f"{phase}_heartbeat.json")
        if snap is not None:
            candidates.append(snap)
            continue
        tail = _read_last_jsonl(logs / f"{phase}_heartbeat.jsonl")
        if tail is not None:
            candidates.append(tail)
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (
            _parse_ts(p.get("timestamp_utc")),
            int(p.get("step", 0)),
        ),
        reverse=True,
    )
    return candidates[0]


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(max(0, round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _heartbeat_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("phase"),
        payload.get("timestamp_utc"),
        payload.get("status"),
        payload.get("step"),
        payload.get("reason"),
        payload.get("latest_train_state"),
        payload.get("latest_repair_checkpoint"),
    )


def _render_dashboard(run_dir: Path, payload: dict[str, Any]) -> str:
    phase = str(payload.get("phase", "unknown"))
    run_id = str(payload.get("run_id", run_dir.name))
    status = str(payload.get("status", "unknown"))
    updated = str(payload.get("timestamp_utc", "n/a"))
    step = int(payload.get("step", 0))
    total = int(payload.get("total_steps", 0))
    progress = 100.0 * float(payload.get("progress", 0.0))
    remaining = int(payload.get("steps_remaining", max(0, total - step)))
    eta = _fmt_duration(payload.get("eta_sec"))
    elapsed = _fmt_duration(payload.get("elapsed_wall_sec"))
    steps_per_sec = _fmt_float(payload.get("steps_per_sec"), digits=3)
    tok_per_sec = _fmt_float(payload.get("tokens_per_sec"), digits=1)
    gpu_mem = _fmt_float(payload.get("gpu_mem_gb"), digits=2)
    loss = _fmt_float(payload.get("loss_total"), digits=6)
    reason = str(payload.get("reason", "n/a"))
    latest_repair = payload.get("latest_repair_checkpoint")
    latest_state = payload.get("latest_train_state")
    resumed_from = payload.get("resumed_from")
    pause_request = run_dir / "control" / "pause.request"
    current_unit = payload.get("current_unit")

    lines = [
        f"[CacheMedic++ Watch] run={run_id} phase={phase} status={status} updated={updated}",
        f"step={step}/{total} ({progress:.2f}%) remaining={remaining} eta={eta} elapsed={elapsed}",
        f"rate: steps/s={steps_per_sec} tokens/s={tok_per_sec} gpu_mem_gb={gpu_mem} loss={loss}",
        f"reason={reason}",
        f"latest_repair_checkpoint={latest_repair if latest_repair else 'n/a'}",
        f"latest_train_state={latest_state if latest_state else 'n/a'}",
        f"resumed_from={resumed_from if resumed_from else 'n/a'}",
        f"pause_request_path={pause_request}",
    ]
    if current_unit is not None:
        lines.append(f"current_unit={current_unit}")
    return "\n".join(lines)


def watch_run(
    run_dir: Path,
    *,
    follow: bool,
    poll_sec: float,
    stop_on_terminal: bool,
    clear_screen: bool,
) -> int:
    printed_waiting = False
    last_key: tuple[Any, ...] | None = None

    while True:
        payload = load_heartbeat(run_dir)
        if payload is None:
            if not printed_waiting:
                print(f"No heartbeat yet under {run_dir / 'logs'}. Waiting...")
                printed_waiting = True
            if not follow:
                return 1
            time.sleep(max(0.1, poll_sec))
            continue

        printed_waiting = False
        key = _heartbeat_key(payload)
        if key != last_key:
            if clear_screen:
                print("\033[2J\033[H", end="")
            print(_render_dashboard(run_dir, payload), flush=True)
            print("", flush=True)
            last_key = key

        status = str(payload.get("status", "unknown"))
        if not follow:
            return 0
        if stop_on_terminal and status in TERMINAL_STATUSES:
            return 0
        time.sleep(max(0.1, poll_sec))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch CacheMedic++ phase heartbeats (train/eval/stability/sweep)."
    )
    p.add_argument("--run_dir", required=True, help="Run directory to monitor.")
    p.add_argument(
        "--follow",
        action="store_true",
        help="Continue polling and printing updates as heartbeat changes.",
    )
    p.add_argument(
        "--poll_sec",
        type=float,
        default=2.0,
        help="Polling interval in seconds when --follow is enabled.",
    )
    p.add_argument(
        "--stop_on_terminal",
        action="store_true",
        help="When following, exit automatically on status completed/paused/failed.",
    )
    p.add_argument(
        "--clear_screen",
        action="store_true",
        help="Clear terminal before each update to keep a single dashboard view.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return watch_run(
        Path(args.run_dir).resolve(),
        follow=bool(args.follow),
        poll_sec=float(args.poll_sec),
        stop_on_terminal=bool(args.stop_on_terminal),
        clear_screen=bool(args.clear_screen),
    )


if __name__ == "__main__":
    raise SystemExit(main())
