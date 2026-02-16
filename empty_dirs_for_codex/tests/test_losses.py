from __future__ import annotations

import unittest

import torch

from cachemedicpp.losses import contraction_ratio


class LossesTest(unittest.TestCase):
    def test_contraction_ratio_is_one_for_identity_repair(self) -> None:
        torch.manual_seed(0)
        K_clean = torch.randn(1, 4, 8, 16)
        V_clean = torch.randn(1, 4, 8, 16)
        K_corr = K_clean + 0.1 * torch.randn_like(K_clean)
        V_corr = V_clean + 0.1 * torch.randn_like(V_clean)

        K_rep = K_corr.clone()
        V_rep = V_corr.clone()
        rho = contraction_ratio(K_clean, V_clean, K_corr, V_corr, K_rep, V_rep)
        self.assertAlmostEqual(float(rho.item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

