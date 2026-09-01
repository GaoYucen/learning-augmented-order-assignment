#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rood_dasfaa2019.learning.runner import run_learning_augmented_instance
from rood_dasfaa2019.learning.synthetic import apply_bottleneck_config, generate_bottleneck_instance
from rood_dasfaa2019.utils.config import load_experiment


ROBUSTNESS_SCENARIOS = [
    {
        "scenario": "scale_under_0.5",
        "corruption": "scale",
        "prediction_scales": [0.5],
        "corruption_strengths": [0.5],
    },
    {
        "scenario": "scale_over_1.5",
        "corruption": "scale",
        "prediction_scales": [1.5],
        "corruption_strengths": [0.5],
    },
    {
        "scenario": "destination_permutation_0.75",
        "corruption": "permutation",
        "prediction_scales": [1.0],
        "corruption_strengths": [0.75],
    },
    {
        "scenario": "adversarial_concentration_0.75",
        "corruption": "adversarial_concentration",
        "prediction_scales": [1.0],
        "corruption_strengths": [0.75],
    },
]


def _ci95(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return 1.96 * float(series.std()) / (len(series) ** 0.5)


def _run_scenario(cfg, seed, slot_id, thetas, scenario):
    stations, orders, buses = generate_bottleneck_instance(cfg, seed)
    rows = run_learning_augmented_instance(
        stations=stations,
        orders=orders,
        buses=buses,
        cfg=cfg,
        seed=seed,
        prediction_scales=scenario["prediction_scales"],
        corruption_strengths=scenario["corruption_strengths"],
        corruption=scenario["corruption"],
        thetas=thetas,
        slot_id=slot_id,
    )
    for row in rows:
        row["scenario"] = scenario["scenario"]
        row["scenario_kind"] = scenario["scenario_kind"]
    return rows


def build_frontier(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rp = raw.loc[raw["method"] == "RP-LAIPD"].copy()
    rp["theta"] = rp["theta"].astype(float)

    consistency_rows = rp.loc[rp["scenario_kind"] == "consistency"]
    consistency = consistency_rows.groupby("theta")["alg_over_opt"].agg(
        consistency="mean",
        consistency_std="std",
        consistency_runs="count",
    ).reset_index()
    consistency["consistency_std"] = consistency["consistency_std"].fillna(0.0)
    consistency["consistency_ci95"] = consistency_rows.groupby("theta")["alg_over_opt"].apply(_ci95).to_numpy()

    robust_rows = rp.loc[rp["scenario_kind"] == "robustness"]
    scenario_summary = robust_rows.groupby(["theta", "scenario"], as_index=False).agg(
        prediction_error=("prediction_error", "mean"),
        advice_error=("advice_error", "mean"),
        alg_over_opt_mean=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"),
        runs=("slot_id", "count"),
    )
    scenario_summary["alg_over_opt_std"] = scenario_summary["alg_over_opt_std"].fillna(0.0)
    idx = scenario_summary.groupby("theta")["alg_over_opt_mean"].idxmin()
    robustness = scenario_summary.loc[idx, ["theta", "scenario", "alg_over_opt_mean"]].rename(
        columns={"scenario": "worst_scenario", "alg_over_opt_mean": "robustness"}
    )
    q05 = robust_rows.groupby("theta")["alg_over_opt"].quantile(0.05).rename("robustness_q05").reset_index()

    frontier = consistency.merge(robustness, on="theta").merge(q05, on="theta")
    return frontier.sort_values("theta").reset_index(drop=True), scenario_summary.sort_values(["theta", "scenario"]).reset_index(drop=True)


def plot_frontier(frontier: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=180)
    plot_df = frontier.sort_values("theta")
    ax.plot(plot_df["consistency"], plot_df["robustness"], marker="o", linewidth=1.8)
    offsets = {
        0.0: (-38, -2),
        0.1: (8, 14),
        0.2: (8, -18),
        0.4: (8, 14),
        0.6: (8, 14),
        0.8: (8, 10),
        1.0: (8, 10),
    }
    for row in plot_df.itertuples():
        dx, dy = offsets.get(round(float(row.theta), 1), (8, 6))
        ax.annotate(
            f"theta={row.theta:g}",
            (row.consistency, row.robustness),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha="right" if dx < 0 else "left",
            va="top" if dy < 0 else "bottom",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.6, "shrinkA": 0, "shrinkB": 4},
        )
    ax.set_title("Consistency-Robustness Frontier")
    ax.set_xlabel("Consistency: ALG/OPT under accurate prediction")
    ax.set_ylabel("Robustness: worst-case ALG/OPT under corrupted prediction")
    fig.text(
        0.08,
        0.02,
        "theta = robust IPD resource share; larger theta trusts prediction less",
        fontsize=7,
        color="dimgray",
    )
    ax.grid(alpha=0.25)
    ax.margins(x=0.08, y=0.18)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A3 consistency-robustness frontier experiment.")
    parser.add_argument("--experiment", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--slots", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "consistency_robustness")
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Robust/IPD resource fractions from the revised draft. theta=1 ignores prediction; theta=0 follows advice.",
    )
    args = parser.parse_args()

    cfg = apply_bottleneck_config(load_experiment(args.experiment))
    scenarios = [
        {
            "scenario": "accurate_prediction",
            "scenario_kind": "consistency",
            "corruption": "scale",
            "prediction_scales": [1.0],
            "corruption_strengths": [0.0],
        },
        *[{**scenario, "scenario_kind": "robustness"} for scenario in ROBUSTNESS_SCENARIOS],
    ]

    rows = []
    for slot in range(args.slots):
        seed = args.seed + args.experiment * 100 + slot
        for scenario in scenarios:
            rows.extend(_run_scenario(cfg, seed, slot, args.thetas, scenario))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    frontier, scenario_summary = build_frontier(raw)

    raw_path = args.output_dir / "consistency_robustness_raw.csv"
    frontier_path = args.output_dir / "consistency_robustness_frontier.csv"
    scenario_path = args.output_dir / "consistency_robustness_scenarios.csv"
    fig_path = args.output_dir / "consistency_robustness_frontier.png"

    raw.to_csv(raw_path, index=False)
    frontier.to_csv(frontier_path, index=False)
    scenario_summary.to_csv(scenario_path, index=False)
    plot_frontier(frontier, fig_path)

    print(frontier.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {frontier_path}")
    print(f"Wrote {scenario_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
