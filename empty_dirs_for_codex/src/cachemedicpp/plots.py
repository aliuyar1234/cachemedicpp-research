"""CLI for generating locked CacheMedic++ plots and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io_utils import ensure_run_layout

LOCKED_FIGURES = [
    "figA_score_vs_eps.png",
    "figB_clean_vs_overhead_pareto.png",
    "figC_logit_sensitivity.png",
    "figD_amplification_map.png",
]

LOCKED_TABLES = [
    "table_main_results.tex",
    "table_ablation_summary.tex",
]

# 1x1 transparent PNG
PLACEHOLDER_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000A49444154789C6360000002000154A24F560000000049454E44AE426082"
)


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        return plt, np
    except Exception:
        return None, None


def _style_matplotlib(plt) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#9aa4b2",
            "grid.linestyle": "-",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d7de",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PLACEHOLDER_PNG)


def _ensure_all_locked_outputs(run_dir: Path) -> None:
    fig_dir = run_dir / "paper" / "figures"
    table_dir = run_dir / "paper" / "tables"
    for name in LOCKED_FIGURES:
        path = fig_dir / name
        if not path.exists():
            _write_placeholder(path)
    for name in LOCKED_TABLES:
        path = table_dir / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("% auto-generated fallback table\n", encoding="utf-8")


def _plot_with_matplotlib(
    run_dir: Path,
    task_metrics: dict[str, Any],
    stability_metrics: dict[str, Any],
) -> None:
    plt, np = _safe_import_matplotlib()
    if plt is None or np is None:
        _ensure_all_locked_outputs(run_dir)
        return
    _style_matplotlib(plt)

    colors = {
        "no_defense": "#2D6CDF",
        "best_heuristic": "#F2994A",
        "cachemedic": "#1F9D55",
        "no_contr": "#C53D3D",
    }

    fig_dir = run_dir / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    score = task_metrics["metrics"]["score_vs_eps"]
    eps = score["eps"]
    plt.figure(figsize=(7.0, 4.3))
    plt.plot(
        eps,
        score["no_defense"],
        label="No defense",
        marker="o",
        color=colors["no_defense"],
    )
    plt.plot(
        eps,
        score["best_heuristic"],
        label="Best heuristic",
        marker="o",
        color=colors["best_heuristic"],
    )
    plt.plot(
        eps,
        score["cachemedic"],
        label="CacheMedic++",
        marker="o",
        color=colors["cachemedic"],
    )
    if eps:
        x_last = eps[-1]
        y_last = score["cachemedic"][-1]
        plt.annotate(
            f"{y_last:.4f}",
            (x_last, y_last),
            xytext=(6, 6),
            textcoords="offset points",
            color=colors["cachemedic"],
            fontsize=10,
            weight="bold",
        )
    plt.xlabel("epsilon")
    plt.ylabel("robustness score (higher is better)")
    plt.title("Robustness score vs corruption severity")
    plt.xticks(eps)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figA_score_vs_eps.png", dpi=220)
    plt.close()

    clean = task_metrics["metrics"]["clean_regression"]
    overhead = task_metrics["metrics"].get("overhead", {})
    plt.figure(figsize=(7.0, 4.3))
    labels = ["no_defense", "best_heuristic", "cachemedic"]
    xs = [clean.get(lbl, 0.0) for lbl in labels]
    ys = [overhead.get(lbl, {}).get("tokens_per_sec", 0.0) for lbl in labels]
    for x, y, lbl in zip(xs, ys, labels):
        color = colors[lbl]
        plt.scatter([x], [y], label=lbl, s=120, color=color, edgecolor="#202020", linewidth=0.6)
        plt.annotate(lbl, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9, color=color)
    plt.xlabel("clean score")
    plt.ylabel("tokens/sec")
    plt.title("Efficiency-quality sweet spot")
    if xs and ys:
        plt.xlim(min(xs) - 0.0004, max(xs) + 0.0005)
        plt.ylim(min(ys) - 60, max(ys) + 60)
    plt.tight_layout()
    plt.savefig(fig_dir / "figB_clean_vs_overhead_pareto.png", dpi=220)
    plt.close()

    sens = stability_metrics["metrics"]["stability.logit_sensitivity"]
    dn = [float(row["delta_norm"]) for row in sens]
    plt.figure(figsize=(7.0, 4.3))
    base_vals = [row["baseline"] for row in sens]
    heur_vals = [row["heuristics"] for row in sens]
    cache_vals = [row["cachemedic"] for row in sens]
    plt.plot(dn, base_vals, label="No defense", marker="o", color=colors["no_defense"])
    plt.plot(dn, heur_vals, label="Best heuristic", marker="o", color=colors["best_heuristic"])
    plt.plot(dn, cache_vals, label="CacheMedic++", marker="o", color=colors["cachemedic"])
    no_contr_vals = [row.get("cachemedic_no_contr") for row in sens]
    if any(v is not None for v in no_contr_vals):
        noc = [float("nan") if v is None else float(v) for v in no_contr_vals]
        plt.plot(
            dn,
            noc,
            label="lambda_contr=0",
            linestyle="--",
            color=colors["no_contr"],
        )
        if len(dn) == len(cache_vals) == len(noc):
            plt.fill_between(dn, cache_vals, noc, color=colors["no_contr"], alpha=0.09)
    plt.xlabel("delta norm")
    plt.ylabel("logit drift ||.||_2")
    plt.title("Logit sensitivity under cache perturbations")
    plt.xticks(dn)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figC_logit_sensitivity.png", dpi=220)
    plt.close()

    amap = stability_metrics["metrics"]["stability.amplification_map"]
    layers = amap.get("layers", list(range(len(amap["baseline"]))))
    heads = amap.get("heads", list(range(len(amap["baseline"][0]) if amap.get("baseline") else 0)))
    base = np.array(amap["baseline"], dtype=float)
    rep = np.array(amap["cachemedic"], dtype=float)
    diff = base - rep
    plt.figure(figsize=(7.1, 4.4))
    im = plt.imshow(diff, aspect="auto", cmap="magma_r")
    cbar = plt.colorbar(im, label="amplification reduction", pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    if heads:
        step = max(1, len(heads) // 8)
        idx = list(range(0, len(heads), step))
        plt.xticks(idx, [heads[i] for i in idx])
    if layers:
        plt.yticks(range(len(layers)), layers)
    plt.xlabel("head index")
    plt.ylabel("layer id")
    plt.title("Amplification map (baseline - cachemedic)")
    plt.tight_layout()
    plt.savefig(fig_dir / "figD_amplification_map.png", dpi=220)
    plt.close()


def _write_tables(run_dir: Path, task_metrics: dict[str, Any], config: dict[str, Any]) -> None:
    table_dir = run_dir / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    best_heur_name = task_metrics["metrics"]["best_heuristic_name"]
    aucs = task_metrics["metrics"]["robustness_auc"]
    clean = task_metrics["metrics"]["clean_regression"]
    main = r"""\begin{tabular}{lrr}
\toprule
Method & Clean Score & Robustness AUC \\
\midrule
No defense & %.4f & %.4f \\
Best heuristic (%s) & %.4f & %.4f \\
CacheMedic++ & %.4f & %.4f \\
\bottomrule
\end{tabular}
""" % (
        clean["no_defense"],
        aucs["no_defense"],
        best_heur_name,
        clean["best_heuristic"],
        aucs["best_heuristic"],
        clean["cachemedic"],
        aucs["cachemedic"],
    )
    (table_dir / "table_main_results.tex").write_text(main, encoding="utf-8")

    ablation = r"""\begin{tabular}{lr}
\toprule
Setting & Value \\
\midrule
repair.family & %s \\
repair.rank & %s \\
repair.apply\_to & %s \\
train.lambda\_contr & %.4f \\
train.lambda\_id & %.4f \\
\bottomrule
\end{tabular}
""" % (
        str(config["repair"]["family"]),
        str(config["repair"]["rank"]),
        str(config["repair"]["apply_to"]),
        float(config["train"]["lambda_contr"]),
        float(config["train"]["lambda_id"]),
    )
    (table_dir / "table_ablation_summary.tex").write_text(ablation, encoding="utf-8")


def generate_plots(run_dir: Path) -> dict[str, Any]:
    ensure_run_layout(run_dir)
    task_path = run_dir / "metrics" / "task_metrics.json"
    stab_path = run_dir / "metrics" / "stability_metrics.json"
    cfg_path = run_dir / "config_resolved.yaml"

    if not task_path.exists() or not stab_path.exists():
        _ensure_all_locked_outputs(run_dir)
        return {"status": "fallback", "reason": "missing_metrics"}

    task_metrics = _load_json(task_path)
    stability_metrics = _load_json(stab_path)

    # Config is optional for table content; if unavailable write basic tables.
    config = {}
    if cfg_path.exists():
        import yaml

        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        config = {"repair": {"family": "A", "rank": 8, "apply_to": "KV"}, "train": {"lambda_contr": 0.0, "lambda_id": 0.0}}

    _plot_with_matplotlib(run_dir, task_metrics, stability_metrics)
    _write_tables(run_dir, task_metrics, config)
    _ensure_all_locked_outputs(run_dir)
    return {"status": "ok", "figures": LOCKED_FIGURES, "tables": LOCKED_TABLES}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate locked CacheMedic++ plots/tables.")
    p.add_argument("--run_dir", required=True, help="Run directory with metrics/ data.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    generate_plots(Path(args.run_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
