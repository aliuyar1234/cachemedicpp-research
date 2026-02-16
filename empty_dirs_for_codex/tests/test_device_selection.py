from __future__ import annotations

import unittest
from unittest.mock import patch

from cachemedicpp.eval import _select_device as eval_select_device
from cachemedicpp.stability import _select_device as stability_select_device
from cachemedicpp.train import _select_device as train_select_device


SELECTORS = [train_select_device, eval_select_device, stability_select_device]


class DeviceSelectionTest(unittest.TestCase):
    def test_cpu_is_explicit(self) -> None:
        for selector in SELECTORS:
            self.assertEqual(selector("cpu").type, "cpu")

    def test_cuda_fail_closed_when_unavailable(self) -> None:
        with patch("cachemedicpp.train.torch.cuda.is_available", return_value=False), patch(
            "cachemedicpp.eval.torch.cuda.is_available", return_value=False
        ), patch("cachemedicpp.stability.torch.cuda.is_available", return_value=False):
            for selector in SELECTORS:
                with self.assertRaises(RuntimeError):
                    selector("cuda")

    def test_cuda_selected_when_available(self) -> None:
        with patch("cachemedicpp.train.torch.cuda.is_available", return_value=True), patch(
            "cachemedicpp.eval.torch.cuda.is_available", return_value=True
        ), patch("cachemedicpp.stability.torch.cuda.is_available", return_value=True):
            for selector in SELECTORS:
                self.assertEqual(selector("cuda").type, "cuda")

    def test_invalid_device_raises(self) -> None:
        for selector in SELECTORS:
            with self.assertRaises(ValueError):
                selector("auto")


if __name__ == "__main__":
    unittest.main()
