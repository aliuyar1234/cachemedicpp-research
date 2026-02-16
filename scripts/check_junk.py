#!/usr/bin/env python3
"""Fail if the repo contains junk artifacts.

Policy (v1.1):
- No __pycache__ directories
- No .pyc files
- No OS junk like .DS_Store

Rationale: the harness ZIP should be a clean system-of-record artifact.
"""

from pathlib import Path

JUNK_DIRS = {"__pycache__"}
JUNK_FILES = {".DS_Store"}
JUNK_SUFFIXES = {".pyc"}
EXCLUDE_DIRS = {".git"}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bad = []

    for p in root.rglob("*"):
        if should_skip(p):
            continue
        if p.is_dir() and p.name in JUNK_DIRS:
            bad.append(str(p.relative_to(root)))
        if p.is_file():
            if p.name in JUNK_FILES:
                bad.append(str(p.relative_to(root)))
            if p.suffix in JUNK_SUFFIXES:
                bad.append(str(p.relative_to(root)))

    if bad:
        for rel in sorted(set(bad)):
            print(f"JUNK_FOUND {rel}")
        return 2

    print("OK: no junk artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
