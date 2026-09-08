#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "chengdu" / "a4_baseline_comparison" / "a4_baseline_summary.csv"
OUTPUT_DIR = ROOT / "outputs" / "chengdu" / "a4_baseline_comparison"


def label_method(row: pd.Series) -> str:
    method = str(row["method"])
    if method == "RP-LAIPD":
        return f"RP-LAIPD\n(theta={float(row['theta']):.1f})"
    if method == "Static LP / bid-price":
        return "Static LP\nbid-price"
    if method == "Prediction-only":
        return "Prediction\nonly"
    return method


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    order = [
        "Offline OPT",
        "Greedy",
        "Random",
        "IPD",
        "Prediction-only",
        "RP-LAIPD",
        "Static LP / bid-price",
    ]
    df["order"] = df["method"].map({method: idx for idx, method in enumerate(order)})
    df = df.sort_values("order").reset_index(drop=True)
    labels = [label_method(row) for _, row in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=220, constrained_layout=True)
    colors = ["#111827", "#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B", "#B279A2"]

    metrics = [
        ("alg_over_opt_mean", "alg_over_opt_ci95", "ALG / OPT", "Online performance against offline optimum"),
        ("accepted_value_mean", "accepted_value_ci95", "Accepted value", "Accepted value by method"),
    ]
    for axis, (mean_col, ci_col, y_label, title) in zip(axes, metrics):
        y = df[mean_col].astype(float).to_numpy()
        yerr = df[ci_col].astype(float).fillna(0.0).to_numpy()
        bars = axis.bar(labels, y, yerr=yerr, color=colors[: len(df)], capsize=4, edgecolor="#2F3640", linewidth=0.7)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(y_label, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=0, labelsize=8)
        axis.margins(y=0.12)
        for bar, value in zip(bars, y):
            axis.annotate(
                f"{value:.3f}" if mean_col == "alg_over_opt_mean" else f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylim(0.0, 1.08)
    fig.suptitle("A4 baseline comparison on Chengdu test slots", fontsize=14)
    fig.savefig(OUTPUT_DIR / "a4_baseline_comparison.png")
    fig.savefig(OUTPUT_DIR / "a4_baseline_comparison.pdf")
    print(OUTPUT_DIR / "a4_baseline_comparison.png")


if __name__ == "__main__":
    main()
