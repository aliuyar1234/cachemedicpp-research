"""Compatibility package shim for implementation under empty_dirs_for_codex/src.

This keeps the CLI contract `python -m cachemedicpp.<cmd>` working from repo root
without requiring an editable install.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_IMPL = _HERE.parent / "empty_dirs_for_codex" / "src" / "cachemedicpp"

if _IMPL.exists():
    __path__.append(str(_IMPL))  # type: ignore[name-defined]

