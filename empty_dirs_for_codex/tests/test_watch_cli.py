from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cachemedicpp.watch import _render_dashboard, load_heartbeat


class WatchCliTest(unittest.TestCase):
    def test_load_heartbeat_prefers_snapshot_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logs = run_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)

            jsonl_payload = {"run_id": "r1", "status": "running", "step": 1}
            (logs / "train_heartbeat.jsonl").write_text(
                json.dumps(jsonl_payload) + "\n",
                encoding="utf-8",
            )
            snap_payload = {"run_id": "r1", "status": "paused", "step": 2}
            (logs / "train_heartbeat.json").write_text(
                json.dumps(snap_payload),
                encoding="utf-8",
            )

            out = load_heartbeat(run_dir)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(out["status"], "paused")
            self.assertEqual(out["step"], 2)

    def test_render_dashboard_contains_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = {
                "phase": "train",
                "run_id": "run_abc",
                "status": "running",
                "timestamp_utc": "2026-02-15T12:00:00+00:00",
                "step": 12,
                "total_steps": 100,
                "progress": 0.12,
                "steps_remaining": 88,
                "eta_sec": 400,
                "elapsed_wall_sec": 50,
                "steps_per_sec": 0.24,
                "tokens_per_sec": 100.0,
                "gpu_mem_gb": 3.5,
                "loss_total": 0.25,
                "reason": "heartbeat",
                "latest_repair_checkpoint": "checkpoints/repair_step_10.pt",
                "latest_train_state": "checkpoints/train_state_latest.pt",
            }
            dashboard = _render_dashboard(run_dir, payload)
            self.assertIn("run=run_abc", dashboard)
            self.assertIn("phase=train", dashboard)
            self.assertIn("status=running", dashboard)
            self.assertIn("step=12/100", dashboard)
            self.assertIn("latest_repair_checkpoint=checkpoints/repair_step_10.pt", dashboard)

    def test_load_heartbeat_uses_latest_phase_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logs = run_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            train_payload = {
                "phase": "train",
                "run_id": "r1",
                "status": "running",
                "timestamp_utc": "2026-02-15T12:00:00+00:00",
                "step": 1,
            }
            eval_payload = {
                "phase": "eval",
                "run_id": "r1",
                "status": "running",
                "timestamp_utc": "2026-02-15T12:05:00+00:00",
                "step": 3,
            }
            (logs / "train_heartbeat.json").write_text(
                json.dumps(train_payload),
                encoding="utf-8",
            )
            (logs / "eval_heartbeat.json").write_text(
                json.dumps(eval_payload),
                encoding="utf-8",
            )

            out = load_heartbeat(run_dir)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(out["phase"], "eval")
            self.assertEqual(out["step"], 3)


if __name__ == "__main__":
    unittest.main()
