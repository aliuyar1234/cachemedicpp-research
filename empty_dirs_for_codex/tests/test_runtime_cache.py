from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cachemedicpp.runtime import _resolve_hf_cache_dir


class RuntimeCacheResolutionTest(unittest.TestCase):
    def test_hf_home_resolves_to_hub_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hf_home = Path(td)
            with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=True):
                cache_dir = _resolve_hf_cache_dir()
                self.assertEqual(cache_dir, hf_home / "hub")

    def test_explicit_hub_cache_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            hub_cache = Path(td) / "hub-cache"
            hub_cache.mkdir(parents=True, exist_ok=True)
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub_cache)}, clear=True):
                cache_dir = _resolve_hf_cache_dir()
                self.assertEqual(cache_dir, hub_cache)
                self.assertNotIn("HF_HOME", os.environ)

    def test_autodetect_sets_hf_home_when_candidate_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            candidate = Path(td)
            with patch.dict(os.environ, {}, clear=True), patch(
                "cachemedicpp.runtime._candidate_hf_homes", return_value=[candidate]
            ):
                cache_dir = _resolve_hf_cache_dir()
                self.assertEqual(cache_dir, candidate / "hub")
                self.assertEqual(os.environ.get("HF_HOME"), str(candidate))
                self.assertEqual(os.environ.get("HF_HUB_CACHE"), str(candidate / "hub"))


if __name__ == "__main__":
    unittest.main()
