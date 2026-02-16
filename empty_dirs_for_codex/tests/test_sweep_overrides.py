from __future__ import annotations

import unittest

from cachemedicpp.config import apply_overrides


class SweepOverridesTest(unittest.TestCase):
    def test_dot_key_overrides_are_deterministic(self) -> None:
        base = {
            "train": {"lambda_contr": 0.5, "steps": 100},
            "repair": {"rank": 8, "protect_layers": [8, 9, 10, 11]},
            "eval": {"ood_protocol": None},
        }
        overrides_a = {
            "repair.rank": 16,
            "train.lambda_contr": 0.0,
            "eval.ood_protocol": "loto_holdout_gaussian",
        }
        overrides_b = {
            "eval.ood_protocol": "loto_holdout_gaussian",
            "train.lambda_contr": 0.0,
            "repair.rank": 16,
        }

        merged_a = apply_overrides(base, overrides_a)
        merged_b = apply_overrides(base, overrides_b)
        self.assertEqual(merged_a, merged_b)
        self.assertEqual(merged_a["repair"]["rank"], 16)
        self.assertEqual(merged_a["train"]["lambda_contr"], 0.0)
        self.assertEqual(merged_a["eval"]["ood_protocol"], "loto_holdout_gaussian")


if __name__ == "__main__":
    unittest.main()

