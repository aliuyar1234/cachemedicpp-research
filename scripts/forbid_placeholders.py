#!/usr/bin/env python3
"""Fail the check if forbidden placeholder tokens exist anywhere in the repo."""

from pathlib import Path

FORBIDDEN = ("TO" + "DO", "TB" + "D", "FI" + "XME")
EXCLUDE_DIRS = {".git", "__pycache__"}
EXCLUDE_FILES = {"MANIFEST.sha256"}
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".pt",
    ".pth",
    ".bin",
    ".onnx",
}


def should_skip(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return True
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def read_text_best_effort(path: Path) -> str:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return ""
    data = path.read_bytes()
    if b"\x00" in data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if should_skip(f):
            continue
        text = read_text_best_effort(f)
        for token in FORBIDDEN:
            if token in text:
                bad.append((str(f.relative_to(root)), token))
    if bad:
        for rel, token in bad:
            print(f"FORBIDDEN_TOKEN {token} in {rel}")
        return 2
    print("OK: no forbidden placeholder tokens found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
