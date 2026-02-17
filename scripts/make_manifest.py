#!/usr/bin/env python3
"""Generate MANIFEST.sha256 for the repo.

Policy (v1.2):
- Include source-of-record files only.
- Exclude runtime artifacts (runs/checkpoints/log queues) by path prefix.
- Do not silently exclude junk directories like __pycache__. Repo hygiene checks
  should keep those out of the tree.

Output format:
<sha256>  <relative/path>
"""

import hashlib
from pathlib import Path

EXCLUDE_FILES = {"MANIFEST.sha256"}
# .git is not part of the shipped artifact, but developers may run this script in a checkout.
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
    out = root / "MANIFEST.sha256"

    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if should_exclude(p, root):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        entries.append((sha256_file(p), rel))

    out.write_text("\n".join(f"{h}  {rel}" for h, rel in entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
