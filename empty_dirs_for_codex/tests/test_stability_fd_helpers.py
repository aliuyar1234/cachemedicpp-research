from __future__ import annotations

import unittest

import torch

from cachemedicpp.stability import _build_delta_pair, _logit_drift


class StabilityFiniteDifferenceHelpersTest(unittest.TestCase):
    def test_delta_pair_is_deterministic_and_respects_masks(self) -> None:
        torch.manual_seed(0)
        K = torch.randn(1, 4, 6, 8)
        V = torch.randn(1, 4, 6, 8)
        head_mask = torch.tensor([False, True, False, False], dtype=torch.bool)
        time_mask = torch.tensor([True, True, False, False, False, False], dtype=torch.bool)

        dK_a, dV_a = _build_delta_pair(
            K_clean=K,
            V_clean=V,
            delta_norm=1.0,
            head_mask=head_mask,
            time_mask=time_mask,
            seed=12345,
        )
        dK_b, dV_b = _build_delta_pair(
            K_clean=K,
            V_clean=V,
            delta_norm=1.0,
            head_mask=head_mask,
            time_mask=time_mask,
            seed=12345,
        )
        self.assertTrue(torch.allclose(dK_a, dK_b))
        self.assertTrue(torch.allclose(dV_a, dV_b))

        mask = (
            head_mask.view(1, 4, 1, 1).to(dtype=torch.bool)
            & time_mask.view(1, 1, 6, 1).to(dtype=torch.bool)
        ).expand_as(dK_a)
        self.assertTrue(torch.allclose(dK_a[~mask], torch.zeros_like(dK_a[~mask])))
        self.assertTrue(torch.allclose(dV_a[~mask], torch.zeros_like(dV_a[~mask])))

    def test_logit_drift_topk_projection(self) -> None:
        z_clean = torch.tensor([0.0, 1.0, 2.0])
        z_other = torch.tensor([1.0, 1.0, 2.0])

        full = _logit_drift(z_clean, z_other, topk_idx=None)
        top1 = _logit_drift(
            z_clean,
            z_other,
            topk_idx=torch.tensor([2], dtype=torch.long),
        )

        self.assertAlmostEqual(full, 1.0, places=6)
        self.assertAlmostEqual(top1, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
