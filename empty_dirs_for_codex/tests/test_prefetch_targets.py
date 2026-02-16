from __future__ import annotations

import unittest
from pathlib import Path

from cachemedicpp.prefetch import (
    ModelRef,
    _dedupe_model_refs,
    _parse_model_spec,
    resolve_prefetch_targets,
)


class PrefetchTargetsTest(unittest.TestCase):
    def test_parse_model_spec_with_revision(self) -> None:
        ref = _parse_model_spec("gpt2-large@main")
        self.assertEqual(ref.name, "gpt2-large")
        self.assertEqual(ref.revision, "main")

    def test_dedupe_refs(self) -> None:
        refs = [
            ModelRef("gpt2-medium", None),
            ModelRef("gpt2-medium", None),
            ModelRef("gpt2-large", "main"),
            ModelRef("gpt2-large", "main"),
        ]
        out = _dedupe_model_refs(refs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], ModelRef("gpt2-medium", None))
        self.assertEqual(out[1], ModelRef("gpt2-large", "main"))

    def test_default_fast_targets_resolve_medium_and_large(self) -> None:
        repo_root = Path.cwd().resolve()
        refs = resolve_prefetch_targets(config_refs=[], model_specs=[], repo_root=repo_root)
        names = [r.name for r in refs]
        self.assertIn("gpt2-medium", names)
        self.assertIn("gpt2-large", names)


if __name__ == "__main__":
    unittest.main()
