#!/usr/bin/env python3
"""Verify MANIFEST.sha256 against repo contents.

Verification policy (v1.2):
- Manifest covers source-of-record files only.
- Runtime artifacts (runs/checkpoints/log queues) are excluded by path prefix.
- Every listed file must exist and its sha256 must match.
- No extra source-of-record files may exist that are not listed.

MANIFEST.sha256 format:
<sha256>  <relative/path>
"""

import hashlib
from pathlib import Path

EXCLUDE_FILES = {
    "MANIFEST.sha256",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "DECISIONS.md",
    "RELIABILITY.md",
    "REPRODUCIBILITY.md",
    "SECURITY.md",
}
EXCLUDE_DIRS = {".git"}
EXCLUDE_PATH_PREFIXES = {
    "handoff/",
    "docs/exec-plans/",
    "docs/design-docs/",
    "docs/generated/",
    "docs/references/",
    "empty_dirs_for_codex/outputs/",
    "outputs/",
    ".pytest_cache/",
}
EXCLUDE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_exclude(path: Path, root: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return True
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    rel = str(path.relative_to(root)).replace("\\", "/")
    if any(rel.startswith(prefix) for prefix in EXCLUDE_PATH_PREFIXES):
        return True
    return any(rel.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.exists():
        print("MANIFEST.sha256 is missing")
        return 2

    expected = {}
    for ln in manifest_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            print(f"Invalid manifest line: {ln}")
            return 2
        h, rel = parts[0], parts[1].strip()
        expected[rel] = h

    # 1) Check manifest entries exist and hashes match.
    bad = []
    for rel, h in expected.items():
        path = (root / rel).resolve()
        if not path.exists():
            bad.append((rel, "missing"))
            continue
        actual = sha256_file(path)
        if actual != h:
            bad.append((rel, "mismatch"))

    # 2) Check no extra files exist.
    actual_files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if should_exclude(p, root):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        actual_files.append(rel)
    actual_set = set(actual_files)
    expected_set = set(expected.keys())

    extras = sorted(actual_set - expected_set)
    missing = sorted(expected_set - actual_set)

    if missing:
        for rel in missing:
            bad.append((rel, "listed_but_missing"))
    if extras:
        for rel in extras:
            bad.append((rel, "extra_not_listed"))

    if bad:
        for rel, why in bad:
            print(f"MANIFEST_FAIL {why}: {rel}")
        return 2

    print("OK: MANIFEST.sha256 verified (complete and matching).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
