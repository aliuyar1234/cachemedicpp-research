from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cachemedicpp.plots import LOCKED_FIGURES, LOCKED_TABLES, generate_plots


class PlotContractTest(unittest.TestCase):
    def test_locked_filenames_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
            task_metrics = {
                "run_id": "r1",
                "config_hash": "h",
                "metrics": {
                    "score_vs_eps": {
                        "eps": [0.0, 0.1],
                        "no_defense": [0.8, 0.4],
                        "best_heuristic": [0.82, 0.5],
                        "cachemedic": [0.84, 0.6],
                    },
                    "clean_regression": {
                        "no_defense": 0.8,
                        "best_heuristic": 0.79,
                        "cachemedic": 0.81,
                    },
                    "robustness_auc": {
                        "no_defense": 0.1,
                        "best_heuristic": 0.2,
                        "cachemedic": 0.3,
                    },
                    "best_heuristic_name": "smoothing",
                    "overhead": {
                        "no_defense": {"tokens_per_sec": 1000},
                        "best_heuristic": {"tokens_per_sec": 900},
                        "cachemedic": {"tokens_per_sec": 800},
                    },
                },
                "settings": {},
            }
            stability_metrics = {
                "run_id": "r1",
                "config_hash": "h",
                "metrics": {
                    "stability.logit_sensitivity": [
                        {
                            "delta_norm": 0.0,
                            "baseline": 0.0,
                            "heuristics": 0.0,
                            "cachemedic": 0.0,
                            "cachemedic_no_contr": 0.0,
                        },
                        {
                            "delta_norm": 1.0,
                            "baseline": 1.0,
                            "heuristics": 0.8,
                            "cachemedic": 0.6,
                            "cachemedic_no_contr": 0.7,
                        },
                    ],
                    "stability.amplification_map": {
                        "layers": [0, 1],
                        "heads": [0, 1],
                        "baseline": [[0.5, 0.4], [0.6, 0.5]],
                        "cachemedic": [[0.3, 0.2], [0.4, 0.3]],
                    },
                    "stability.settings": {},
                },
                "settings": {},
            }
            (run_dir / "metrics" / "task_metrics.json").write_text(
                json.dumps(task_metrics), encoding="utf-8"
            )
            (run_dir / "metrics" / "stability_metrics.json").write_text(
                json.dumps(stability_metrics), encoding="utf-8"
            )
            (run_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(
                    {
                        "repair": {"family": "A", "rank": 8, "apply_to": "KV"},
                        "train": {"lambda_contr": 0.5, "lambda_id": 1.0},
                    }
                ),
                encoding="utf-8",
            )

            generate_plots(run_dir)

            for fname in LOCKED_FIGURES:
                self.assertTrue((run_dir / "paper" / "figures" / fname).exists(), msg=fname)
            for fname in LOCKED_TABLES:
                self.assertTrue((run_dir / "paper" / "tables" / fname).exists(), msg=fname)


if __name__ == "__main__":
    unittest.main()

