"""Three figures from the JSON reports. All data is synthetic and the figures say so."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPORT_DIR = Path("reports")

# Fixed per-entity colours: categorical slots 1, 2, 3, then 5 and 7 of the reference palette,
# validated as an adjacent sequence (needs direct value labels, which every chart here has).
COLORS = {
    "trust": "#2a78d6",
    "global_multiplier": "#eb6834",
    "type_median": "#1baf7a",
    "gbm": "#e87ba4",
    "gbm_q80": "#6250d6",
    "oracle": "#8a8985",
}
LABELS = {
    "trust": "Trust estimate",
    "global_multiplier": "Global multiplier",
    "type_median": "Per-type median",
    "gbm": "GBM (ML)",
    "gbm_q80": "GBM q80 (ML, cautious)",
    "oracle": "Oracle (true durations)",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
NOTE = "Synthetic data: shows how the method behaves when its assumptions hold, not real-world benefit."


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(*args: Any, **kwargs: Any) -> Any:
    fig = plt.figure(*args, facecolor=SURFACE, **kwargs)
    return fig


def _footer(fig: Any) -> None:
    fig.text(0.01, 0.01, NOTE, fontsize=8, color=INK_2, ha="left", va="bottom")


def fig_estimation_error(metrics: dict[str, Any], out: Path) -> None:
    sl = metrics["mae_minutes"]["eval_period"]
    names = ["trust", "global_multiplier", "type_median", "gbm"]
    fig = _fig(figsize=(8, 3.6))
    ax = fig.add_axes((0.25, 0.2, 0.7, 0.58))
    _style(ax)
    for i, name in enumerate(names):
        e = sl[name]
        ax.barh(i, e["mae"], height=0.5, color=COLORS[name], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.plot(e["ci95"], [i, i], color=INK, linewidth=1.2, zorder=4)
        ax.text(e["ci95"][1] + 0.4, i, f"{e['mae']:.1f}", va="center", fontsize=10, color=INK)
    floor = sl["oracle_median"]["mae"]
    ax.axvline(floor, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        floor + 0.3,
        -0.62,
        f"noise floor {floor:.1f} min (true conditional median)",
        fontsize=8,
        color=INK_2,
        va="center",
    )
    ax.set_ylim(len(names) - 0.5, -0.85)
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names], color=INK, fontsize=10)
    ax.set_xlim(0, max(sl[n]["ci95"][1] for n in names) * 1.15)
    ax.set_xlabel("Mean absolute error of predicted task duration (minutes)", color=INK_2, fontsize=9)
    fig.text(
        0.01, 0.94, "Duration-estimate error on unseen synthetic users", fontsize=13, color=INK, weight="bold"
    )
    fig.text(
        0.01,
        0.875,
        f"Test users, evaluation period ({sl['n_users']} users, {sl['n_rows']:,} tasks). Bars: MAE; lines: 95% CI over users.",
        fontsize=9,
        color=INK_2,
    )
    _footer(fig)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_calibration(metrics: dict[str, Any], out: Path) -> None:
    cal = metrics["quantile_calibration"]
    fig = _fig(figsize=(9, 4.2))
    for k, (name, color) in enumerate((("gbm_q50", COLORS["gbm"]), ("gbm_q80", COLORS["gbm_q80"]))):
        ax = fig.add_axes((0.08 + 0.46 * k, 0.2, 0.4, 0.58))
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        c = cal[name]
        xs = [r["mean_ratio"] for r in c["reliability"]]
        ys = [r["coverage"] for r in c["reliability"]]
        ax.axhline(c["nominal"], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(
            max(xs), c["nominal"] + 0.025, f"nominal {c['nominal']:.0%}", fontsize=8, color=INK_2, ha="right"
        )
        ax.plot(xs, ys, color=color, linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Predicted quantile ÷ user's estimate (binned)", color=INK_2, fontsize=9)
        ax.set_title(
            f"GBM {name[4:]}: overall coverage {c['coverage']:.1%}", fontsize=10, color=INK, loc="left"
        )
        if k == 0:
            ax.set_ylabel("Share of tasks finishing within the prediction", color=INK_2, fontsize=9)
    fig.text(
        0.01, 0.94, "Quantile calibration on unseen synthetic users", fontsize=13, color=INK, weight="bold"
    )
    fig.text(
        0.01,
        0.885,
        "A calibrated q-quantile is exceeded by the actual duration (1 − q) of the time; points should sit on the dashed line.",
        fontsize=9,
        color=INK_2,
    )
    _footer(fig)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


PANELS = [
    ("deadlines_missed", "Deadlines missed per day"),
    ("replans", "Replans per day"),
    ("idle_minutes", "Idle minutes per day"),
    ("tasks_dropped", "Tasks dropped per day"),
    ("event_overlap_minutes", "Minutes spilling into fixed events"),
    ("overtime_minutes", "Overtime minutes per day"),
]


def fig_schedule_outcomes(results: dict[str, Any], out: Path, reclaim: bool = False) -> None:
    names = list(LABELS)
    fig = _fig(figsize=(12, 6.6))
    axes = fig.subplots(
        2,
        3,
        sharey=True,
        gridspec_kw={"left": 0.17, "right": 0.97, "top": 0.82, "bottom": 0.14, "wspace": 0.12, "hspace": 0.5},
    )
    for ax, (metric, title) in zip(axes.flat, PANELS, strict=True):
        _style(ax)
        hi_max = max(results[n][metric]["ci95"][1] for n in names)
        for i, name in enumerate(names):
            e = results[name][metric]
            ax.barh(
                i,
                e["mean"],
                height=0.55,
                color=COLORS[name],
                edgecolor=SURFACE,
                linewidth=2,
                hatch="////" if name == "oracle" else None,
                zorder=3,
            )
            ax.plot(e["ci95"], [i, i], color=INK, linewidth=1.1, zorder=4)
            ax.text(e["ci95"][1] + hi_max * 0.02, i, f"{e['mean']:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, hi_max * 1.22)
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names], color=INK, fontsize=9)
    axes[0, 0].invert_yaxis()
    fig.text(
        0.01,
        0.95,
        f"End-to-end results: the scheduler run inside the simulator with each estimator{', early finishes reclaimed' if reclaim else ''}",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.01,
        0.905,
        f"Per simulated day, mean over {results['n_users']} unseen users × {results['n_days']} days; lines are 95% CIs from resampling users. Lower is better in every panel, but panels trade off against each other.",
        fontsize=9,
        color=INK_2,
    )
    fig.text(
        0.01,
        0.865,
        "Idle minutes are unreclaimed slack: a cautious plan trades fewer overruns for more of it. The oracle is an unattainable reference.",
        fontsize=9,
        color=INK_2,
    )
    handles = [
        Patch(
            facecolor=COLORS[n], hatch="////" if n == "oracle" else None, edgecolor=SURFACE, label=LABELS[n]
        )
        for n in names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.57, 0.035),
        labelcolor=INK_2,
    )
    _footer(fig)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


FRONTIER_PANELS = [
    ("deadlines_missed", "Deadlines missed per day"),
    ("replans", "Replans per day"),
    ("tasks_dropped", "Tasks dropped per day"),
]
QUANTILE_NAMES = ["gbm_q50", "gbm_q60", "gbm_q70", "gbm_q80", "gbm_q90"]


def fig_frontier(frontier: dict[str, Any], out: Path) -> None:
    """The price of caution: each point is a planning quantile (q50 to q90); x is the idle time it costs."""
    fig = _fig(figsize=(12, 4.6))
    axes = fig.subplots(
        1, 3, gridspec_kw={"left": 0.06, "right": 0.98, "top": 0.74, "bottom": 0.2, "wspace": 0.28}
    )
    plain, reclaim = frontier["no_reclaim"], frontier["reclaim"]
    for ax, (metric, title) in zip(axes, FRONTIER_PANELS, strict=True):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for summary, dashed in ((plain, False), (reclaim, True)):
            xs = [summary[n]["idle_minutes"]["mean"] for n in QUANTILE_NAMES]
            ys = [summary[n][metric]["mean"] for n in QUANTILE_NAMES]
            ax.plot(
                xs,
                ys,
                color=COLORS["gbm_q80"],
                linewidth=2,
                linestyle=(0, (4, 3)) if dashed else "-",
                zorder=3,
            )
            ax.scatter(xs, ys, s=56, color=COLORS["gbm_q80"], edgecolor=SURFACE, linewidth=2, zorder=4)
            if not dashed:
                for n, x, y in zip(QUANTILE_NAMES, xs, ys, strict=True):
                    ax.annotate(
                        n[-2:], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, color=INK_2
                    )
        for name, marker in (("trust", "s"), ("type_median", "D"), ("oracle", "^")):
            ax.scatter(
                plain[name]["idle_minutes"]["mean"],
                plain[name][metric]["mean"],
                s=70,
                marker=marker,
                color=COLORS[name],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=5,
            )
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Idle minutes per day", color=INK_2, fontsize=9)
        ax.set_ylim(bottom=0)
    handles = [
        Line2D(
            [],
            [],
            color=COLORS["gbm_q80"],
            linewidth=2,
            marker="o",
            label="GBM, planning quantile q50 to q90",
        ),
        Line2D(
            [],
            [],
            color=COLORS["gbm_q80"],
            linewidth=2,
            linestyle=(0, (4, 3)),
            label="same, early finishes reclaimed",
        ),
        Line2D([], [], color=COLORS["trust"], marker="s", linestyle="", label="Trust estimate"),
        Line2D([], [], color=COLORS["type_median"], marker="D", linestyle="", label="Per-type median"),
        Line2D([], [], color=COLORS["oracle"], marker="^", linestyle="", label="Oracle (reference)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.01,
        0.94,
        "The price of caution: planning at a higher percentile",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.01,
        0.875,
        "Moving right plans for longer tasks: fewer misses and replans, more idle time and dropped tasks. Reclaiming early finishes shifts the curve left.",
        fontsize=9,
        color=INK_2,
    )
    _footer(fig)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_figures(report_dir: Path = REPORT_DIR) -> list[Path]:
    out_dir = report_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    metrics_path = report_dir / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        fig_estimation_error(metrics, out_dir / "fig1_estimation_error.png")
        fig_calibration(metrics, out_dir / "fig2_quantile_calibration.png")
        made += [out_dir / "fig1_estimation_error.png", out_dir / "fig2_quantile_calibration.png"]
    results_path = report_dir / "schedule_results.json"
    if results_path.exists():
        fig_schedule_outcomes(json.loads(results_path.read_text()), out_dir / "fig3_schedule_outcomes.png")
        made.append(out_dir / "fig3_schedule_outcomes.png")
    reclaim_path = report_dir / "schedule_results_reclaim.json"
    if reclaim_path.exists():
        target = out_dir / "fig3b_schedule_outcomes_reclaim.png"
        fig_schedule_outcomes(json.loads(reclaim_path.read_text()), target, reclaim=True)
        made.append(target)
    sens_path = report_dir / "sensitivity.json"
    if sens_path.exists():
        sens = json.loads(sens_path.read_text())
        if "frontier" in sens:
            target = out_dir / "fig4_caution_frontier.png"
            fig_frontier(sens["frontier"], target)
            made.append(target)
    return made


if __name__ == "__main__":
    for path in make_figures():
        print(path)
