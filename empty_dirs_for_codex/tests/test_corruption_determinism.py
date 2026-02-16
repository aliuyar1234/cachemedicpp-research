from __future__ import annotations

import unittest

import torch

from cachemedicpp.corruption import (
    corrupt_gaussian,
    make_generator,
    sample_head_mask,
    sample_time_mask,
)


class CorruptionDeterminismTest(unittest.TestCase):
    def test_masks_and_gaussian_are_deterministic(self) -> None:
        axes = {"head_mode": "bernoulli", "p_head": 0.5, "time_mode": "old_only", "N_recent": 3}

        gen_a = make_generator(1234)
        head_a = sample_head_mask(8, axes, gen_a)
        time_a = sample_time_mask(16, axes, gen_a)

        gen_b = make_generator(1234)
        head_b = sample_head_mask(8, axes, gen_b)
        time_b = sample_time_mask(16, axes, gen_b)

        self.assertTrue(torch.equal(head_a, head_b))
        self.assertTrue(torch.equal(time_a, time_b))

        K = torch.randn(1, 8, 16, 32)
        V = torch.randn(1, 8, 16, 32)
        out1 = corrupt_gaussian(
            K,
            V,
            head_mask=head_a,
            time_mask=time_a,
            epsilon=0.08,
            gen=make_generator(987),
        )
        out2 = corrupt_gaussian(
            K,
            V,
            head_mask=head_a,
            time_mask=time_a,
            epsilon=0.08,
            gen=make_generator(987),
        )
        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out1[1], out2[1]))


if __name__ == "__main__":
    unittest.main()

