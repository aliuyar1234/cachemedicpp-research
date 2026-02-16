#!/usr/bin/env python3
"""Verify that the paper skeleton references the locked figure/table filenames.

This prevents drift between experiments and the LaTeX paper.
"""

from pathlib import Path

REQUIRED = [
    "figures/figA_score_vs_eps.png",
    "figures/figB_clean_vs_overhead_pareto.png",
    "figures/figC_logit_sensitivity.png",
    "figures/figD_amplification_map.png",
    "tables/table_main_results.tex",
    "tables/table_ablation_summary.tex",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paper_dir = root / "paper"
    if not paper_dir.exists():
        print("paper/ directory is missing")
        return 2
    text = ""
    for tex in sorted(paper_dir.rglob("*.tex")):
        text += tex.read_text(encoding="utf-8", errors="ignore") + "\n"
    missing = [p for p in REQUIRED if p not in text]
    if missing:
        for m in missing:
            print(f"PAPER_CONTRACT_MISSING {m}")
        return 2
    print("OK: paper skeleton references locked artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
