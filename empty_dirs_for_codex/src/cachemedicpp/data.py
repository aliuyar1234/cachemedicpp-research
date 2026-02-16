"""Dataset loading with deterministic offline fallbacks and prompt helpers."""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path
from typing import Any

from .config import resolve_repo_root


def _load_online_wikitext2(num_examples: int) -> list[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [str(x) for x in ds["text"] if str(x).strip()]
    if not texts:
        raise RuntimeError("Online Wikitext-2 dataset appears empty.")
    return [texts[i % len(texts)] for i in range(num_examples)]


def _load_online_sst2(num_examples: int) -> list[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("glue", "sst2", split="validation")
    out = []
    for i in range(num_examples):
        item = ds[i % len(ds)]
        out.append({"text": str(item["sentence"]), "label": int(item["label"])})
    return out


def _load_offline_wikitext2(path: Path, num_examples: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Offline Wikitext file has no non-empty lines: {path}")
    return [lines[i % len(lines)] for i in range(num_examples)]


def _load_offline_sst2(path: Path, num_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        rows.append({"text": str(obj["text"]), "label": int(obj["label"])})
    if not rows:
        raise RuntimeError(f"Offline SST-2 file has no records: {path}")
    return [rows[i % len(rows)] for i in range(num_examples)]


def load_wikitext2(
    data_cfg: dict[str, Any],
    *,
    num_examples: int,
    repo_root: Path | None = None,
) -> list[str]:
    root = repo_root or resolve_repo_root()
    use_offline = bool(data_cfg.get("use_offline", False))
    offline_path = root / str(data_cfg["offline_paths"]["wikitext2"])
    if use_offline:
        return _load_offline_wikitext2(offline_path, num_examples)
    try:
        return _load_online_wikitext2(num_examples)
    except Exception:
        return _load_offline_wikitext2(offline_path, num_examples)


def load_sst2(
    data_cfg: dict[str, Any],
    *,
    num_examples: int,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = repo_root or resolve_repo_root()
    use_offline = bool(data_cfg.get("use_offline", False))
    offline_path = root / str(data_cfg["offline_paths"]["sst2"])
    if use_offline:
        return _load_offline_sst2(offline_path, num_examples)
    try:
        return _load_online_sst2(num_examples)
    except Exception:
        return _load_offline_sst2(offline_path, num_examples)


def build_sst2_prompt(text: str) -> str:
    return f"Review: {text}\nSentiment:"


SST2_LABEL_NEG = " negative"
SST2_LABEL_POS = " positive"


def _random_sentence(rng: random.Random) -> str:
    vocab = [
        "cache",
        "state",
        "token",
        "context",
        "stable",
        "window",
        "layer",
        "head",
        "attention",
        "repair",
        "signal",
        "drift",
        "deterministic",
    ]
    length = rng.randint(7, 16)
    words = [vocab[rng.randrange(len(vocab))] for _ in range(length)]
    sent = " ".join(words)
    return sent[:1].upper() + sent[1:] + "."


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def generate_needle_sample(long_cfg: dict[str, Any], index: int) -> dict[str, str]:
    seed = int(long_cfg["seed"]) + int(index)
    rng = random.Random(seed)
    context_len = int(long_cfg["context_len"])
    needle_token = str(long_cfg["needle_token"])
    needle_value = str(long_cfg["needle_value"])
    position_frac = float(long_cfg["position_frac"])

    needle_sentence = f"NEEDLE: {needle_token} = {needle_value}."
    filler = []
    while len(" ".join(filler)) < max(256, context_len * 3):
        filler.append(_random_sentence(rng))
    n = len(filler)
    insert_idx = min(n - 1, max(0, int(position_frac * n)))
    filler.insert(insert_idx, needle_sentence)
    context = " ".join(filler)

    # approximate token budget by whitespace.
    words = context.split()
    if len(words) > context_len:
        words = words[:context_len]
    context = " ".join(words)

    question_template = str(long_cfg["question_template"])
    question = question_template.format(needle_token=needle_token)
    return {"context": context, "question": question, "answer": needle_value}


def generate_needle_dataset(long_cfg: dict[str, Any], num_examples: int) -> list[dict[str, str]]:
    return [generate_needle_sample(long_cfg, i) for i in range(num_examples)]

