from __future__ import annotations

import unittest

import torch

from cachemedicpp.corruption import make_orthogonal_matrix


class RotationMatrixTest(unittest.TestCase):
    def test_rotation_matrix_is_deterministic_and_orthogonal(self) -> None:
        R1 = make_orthogonal_matrix(16, seed=999)
        R2 = make_orthogonal_matrix(16, seed=999)
        self.assertTrue(torch.allclose(R1, R2))

        I = torch.eye(16, dtype=R1.dtype)
        self.assertTrue(torch.allclose(R1.T @ R1, I, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

