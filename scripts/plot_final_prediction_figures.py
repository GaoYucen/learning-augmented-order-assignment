#!/usr/bin/env python
"""Redraw the two final figures using saved CSV files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_STYLES = {
    "HistoricalAverage": ("Historical Average", "#F58518", "s"),
    "RidgeAR": ("RidgeAR", "#4C78A8", "D"),
    "HGB": ("HGB", "#54A24B", "o"),
}


def plot_forecast_horizon(input_dir: Path) -> None:
    metrics = pd.read_csv(input_dir / "forecast_metrics.csv")
    test = metrics.loc[metrics["split"].eq("test")].copy()
    fig, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for model, (label, color, marker) in MODEL_STYLES.items():
        selected = test.loc[test["model"].eq(model)].sort_values("horizon")
        lower = selected["mae"] - selected["mae_ci95_lower"]
        upper = selected["mae_ci95_upper"] - selected["mae"]
        axis.errorbar(
            selected["horizon"],
            selected["mae"],
            yerr=np.vstack([lower.clip(lower=0), upper.clip(lower=0)]),
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            capsize=4,
        )
    horizons = [1, 2, 4, 8]
    axis.set_xticks(horizons, ["1\n30 min", "2\n60 min", "4\n120 min", "8\n240 min"])
    axis.set_xlabel("Forecast horizon h (30-minute service slots)")
    axis.set_ylabel("MAE (lower is better)")
    axis.set_title("Forecast horizon versus station-demand error")
    axis.grid(alpha=0.22)
    axis.legend(frameon=True)
    axis.text(
        0.01,
        0.01,
        "Sampled experimental demand: max 300 orders/slot; one passenger/order.\n"
        "95% CI: service-date block bootstrap over common Test targets.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    fig.savefig(input_dir / "fig1_forecast_horizon_mae.png", dpi=240)
    fig.savefig(input_dir / "fig1_forecast_horizon_mae.pdf")
    plt.close(fig)


def plot_station_and_theta(input_dir: Path) -> None:
    source = pd.read_csv(input_dir / "representative_station_predictions.csv")
    theta = pd.read_csv(input_dir / "theta_error_summary.csv")
    with (input_dir / "experiment_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    slot_id = str(manifest["selected_representative_slot"])
    selected_predictor = str(manifest["selected_theta_predictor"])

    observed = (
        source.loc[source["model"].eq("HistoricalAverage"), ["type_id", "observed_count"]]
        .sort_values("type_id")
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)
    left.bar(
        observed["type_id"],
        observed["observed_count"],
        color="#B9BDC5",
        alpha=0.8,
        label="Observed",
    )
    for model, (label, color, marker) in MODEL_STYLES.items():
        selected = source.loc[source["model"].eq(model)].sort_values("type_id")
        left.plot(
            selected["type_id"],
            selected["predicted_count"],
            label=label,
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=1.6,
        )
    left.set_xlabel("Destination station ID")
    left.set_ylabel("Sampled demand (passengers)")
    left.set_title(f"Observed and h=1 predicted demand by station\nrepresentative Test slot: {slot_id}")
    left.set_xticks(observed["type_id"].iloc[::2])
    left.grid(axis="y", alpha=0.22)
    left.legend(ncol=2, fontsize=9)

    theta_values = sorted(theta["theta"].unique())
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(theta_values)))
    for theta_value, color in zip(theta_values, colors):
        selected = theta.loc[theta["theta"].eq(theta_value)].sort_values("actual_mae")
        lower = selected["alg_over_opt"] - selected["alg_over_opt_ci95_lower"]
        upper = selected["alg_over_opt_ci95_upper"] - selected["alg_over_opt"]
        right.errorbar(
            selected["actual_mae"],
            selected["alg_over_opt"],
            yerr=np.vstack([lower.clip(lower=0), upper.clip(lower=0)]),
            label=f"theta={theta_value:g}",
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            capsize=3,
        )
    right.set_xlabel("Measured MAE after perturbation")
    right.set_ylabel("Dispatch performance (ALG / OPT)")
    right.set_title(f"Theta sensitivity with fixed predictor: {selected_predictor}")
    right.grid(alpha=0.22)
    right.legend(ncol=2, fontsize=9)
    right.set_ylim(max(0.0, float(theta["alg_over_opt_ci95_lower"].min()) - 0.03), 1.01)
    right.text(
        0.01,
        0.01,
        "theta=0: Prediction-only | theta=1: IPD. x-axis is measured MAE, not lambda.\n"
        "Empirical Gaussian-noise sensitivity; not a theoretical worst-case guarantee.",
        transform=right.transAxes,
        fontsize=8.3,
        color="#444444",
    )

    fig.savefig(input_dir / "fig2_station_prediction_theta.png", dpi=240)
    fig.savefig(input_dir / "fig2_station_prediction_theta.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/chengdu/final_experiment"))
    args = parser.parse_args()
    plot_forecast_horizon(args.input_dir)
    plot_station_and_theta(args.input_dir)
    print(f"Figures redrawn from saved CSV files in {args.input_dir}")


if __name__ == "__main__":
    main()
