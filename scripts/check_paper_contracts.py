#!/usr/bin/env python3
"""Verify paper contracts and claim-to-evidence integrity.

This check prevents drift between:
- locked paper artifact names,
- evidence paths referenced in LaTeX,
- and key numeric claims in table_lambda_sensitivity.tex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED = [
    "figures/figA_score_vs_eps.png",
    "figures/figB_clean_vs_overhead_pareto.png",
    "figures/figC_logit_sensitivity.png",
    "figures/figD_amplification_map.png",
    "tables/table_main_results.tex",
    "tables/table_ablation_summary.tex",
]

PATH_MACRO_RE = re.compile(r"\\(?:path|texttt)\{([^}]+)\}")

LAMBDA_EXPECTED_RUNS = {
    0.0: "evidence/canonical/medium_no_contr",
    3.0: "evidence/canonical/medium_contr",
    5.0: "evidence/lambda/lambda5_tuned_single",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _strip_latex_number(cell: str) -> float:
    cleaned = cell.strip()
    cleaned = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not m:
        raise ValueError(f"no numeric token in cell: {cell!r}")
    return float(m.group(0))


def _load_lambda_metrics(run_dir: Path) -> tuple[float, float, float, float]:
    task = json.loads((run_dir / "metrics" / "task_metrics.json").read_text(encoding="utf-8"))
    stab = json.loads((run_dir / "metrics" / "stability_metrics.json").read_text(encoding="utf-8"))
    sens = stab["metrics"]["stability.logit_sensitivity"]
    d1 = next(x for x in sens if float(x["delta_norm"]) == 1.0)
    d2 = next(x for x in sens if float(x["delta_norm"]) == 2.0)
    return (
        float(task["metrics"]["clean_regression"]["cachemedic"]),
        float(task["metrics"]["robustness_auc"]["cachemedic"]),
        float(d1["cachemedic"]),
        float(d2["cachemedic"]),
    )


def _parse_lambda_table(table_path: Path) -> dict[float, tuple[float, float, float, float]]:
    rows: dict[float, tuple[float, float, float, float]] = {}
    for raw in _read_text(table_path).splitlines():
        line = raw.strip()
        if "&" not in line or not line.endswith("\\\\"):
            continue
        parts = [p.strip() for p in line[:-2].split("&")]
        if len(parts) != 5:
            continue
        try:
            lam = _strip_latex_number(parts[0])
            rows[lam] = tuple(_strip_latex_number(p) for p in parts[1:5])  # type: ignore[assignment]
        except ValueError:
            continue
    return rows


def _check_locked_contracts(paper_dir: Path) -> list[str]:
    text = ""
    for tex in sorted(paper_dir.rglob("*.tex")):
        text += _read_text(tex) + "\n"
    return [p for p in REQUIRED if p not in text]


def _collect_path_refs(paper_dir: Path) -> list[str]:
    refs: list[str] = []
    for tex in sorted(paper_dir.rglob("*.tex")):
        for m in PATH_MACRO_RE.finditer(_read_text(tex)):
            ref = m.group(1).replace(r"\_", "_").strip()
            if "/" not in ref:
                continue
            if ref.startswith("http://") or ref.startswith("https://"):
                continue
            refs.append(ref)
    return refs


def _check_referenced_paths(root: Path, refs: list[str]) -> list[str]:
    bad: list[str] = []
    evidence_dir_exists = (root / "evidence").exists()
    for ref in sorted(set(refs)):
        if ref.startswith("evidence/") and not evidence_dir_exists:
            bad.append(f"STALE_PREFIX {ref} (missing repo root evidence/ directory)")
            continue
        if "*" in ref:
            if not any(root.glob(ref)):
                bad.append(f"MISSING_GLOB {ref}")
            continue
        if not (root / ref).exists():
            bad.append(f"MISSING_PATH {ref}")
    return bad


def _check_lambda_table(root: Path) -> list[str]:
    table_path = root / "paper" / "tables" / "table_lambda_sensitivity.tex"
    if not table_path.exists():
        return []
    rows = _parse_lambda_table(table_path)
    bad: list[str] = []
    for lam, rel_run in LAMBDA_EXPECTED_RUNS.items():
        if lam not in rows:
            bad.append(f"LAMBDA_ROW_MISSING lambda={lam:g}")
            continue
        run_dir = root / rel_run
        if not run_dir.exists():
            bad.append(f"LAMBDA_RUN_MISSING lambda={lam:g} path={rel_run}")
            continue
        expected = _load_lambda_metrics(run_dir)
        observed = rows[lam]
        labels = ("clean", "auc", "sens_d1", "sens_d2")
        for idx, label in enumerate(labels):
            if round(expected[idx], 4) != round(observed[idx], 4):
                bad.append(
                    "LAMBDA_VALUE_MISMATCH "
                    f"lambda={lam:g} field={label} table={observed[idx]:.4f} "
                    f"expected={expected[idx]:.4f} source={rel_run}"
                )
    return bad


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paper_dir = root / "paper"
    if not paper_dir.exists():
        print("paper/ directory is missing")
        return 2

    bad: list[str] = []

    missing_locked = _check_locked_contracts(paper_dir)
    for m in missing_locked:
        bad.append(f"PAPER_CONTRACT_MISSING {m}")

    refs = _collect_path_refs(paper_dir)
    bad.extend(f"PAPER_EVIDENCE_BAD {x}" for x in _check_referenced_paths(root, refs))
    bad.extend(_check_lambda_table(root))

    if bad:
        for msg in bad:
            print(msg)
        return 2

    print("OK: paper contracts and evidence links are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
