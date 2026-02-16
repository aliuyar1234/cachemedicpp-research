from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cachemedicpp.train import _latest_train_state_path, _make_heartbeat_payload


class TrainRunControlHelpersTest(unittest.TestCase):
    def test_latest_train_state_prefers_latest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "train_state_step_1.pt").write_bytes(b"x")
            (ckpt_dir / "train_state_step_7.pt").write_bytes(b"x")

            latest = _latest_train_state_path(run_dir)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.name, "train_state_step_7.pt")

            (ckpt_dir / "train_state_latest.pt").write_bytes(b"x")
            latest2 = _latest_train_state_path(run_dir)
            self.assertIsNotNone(latest2)
            self.assertEqual(latest2.name, "train_state_latest.pt")

    def test_heartbeat_payload_contains_progress_and_eta(self) -> None:
        payload = _make_heartbeat_payload(
            run_id="r1",
            status="running",
            step=50,
            total_steps=100,
            elapsed_wall_sec=25.0,
            last_tokens_per_sec=123.0,
            last_gpu_mem_gb=4.5,
            last_loss_total=0.42,
            latest_repair_checkpoint="checkpoints/repair_step_50.pt",
            latest_train_state="checkpoints/train_state_step_50.pt",
            resumed_from=None,
            reason="heartbeat",
        )
        self.assertEqual(payload["run_id"], "r1")
        self.assertEqual(payload["status"], "running")
        self.assertAlmostEqual(float(payload["progress"]), 0.5, places=6)
        self.assertGreater(float(payload["eta_sec"]), 0.0)
        self.assertEqual(payload["steps_remaining"], 50)


if __name__ == "__main__":
    unittest.main()
