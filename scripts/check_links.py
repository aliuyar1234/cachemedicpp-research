#!/usr/bin/env python3
"""Validate that relative markdown links in this repo resolve to files.

Rules:
- Only checks links of the form [text](path) where path does not start with http(s) or mailto.
- Strips URL fragments (#...) and query strings (?...) for filesystem resolution.
- Allows links to directories if they exist.
"""

import re
from pathlib import Path

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
EXCLUDE_DIRS = {".git", "__pycache__", "outputs"}


def should_exclude(p: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in p.parts)


def normalize_link(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # ignore web links
    if target.startswith("http://") or target.startswith("https://") or target.startswith("mailto:"):
        return ""
    # strip fragments and query strings
    target = target.split("#", 1)[0].split("?", 1)[0]
    return target


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bad = []
    for md in root.rglob("*.md"):
        if should_exclude(md):
            continue
        text = md.read_text(encoding="utf-8", errors="ignore")
        for m in MD_LINK.finditer(text):
            raw = m.group(1)
            target = normalize_link(raw)
            if not target:
                continue
            # Allow pure anchors like (#section)
            if target.startswith("#") or target == "":
                continue
            # Resolve path relative to file
            resolved = (md.parent / target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                # Link points outside repo
                bad.append((str(md.relative_to(root)), raw, "points outside repo"))
                continue
            if not resolved.exists():
                bad.append((str(md.relative_to(root)), raw, "missing"))
    if bad:
        for src, raw, why in bad:
            print(f"BAD_LINK in {src}: ({raw}) -> {why}")
        return 2
    print("OK: markdown links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
