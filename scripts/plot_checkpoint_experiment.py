#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


COLORS = {"IPD": "#4C78A8", "Prediction-only": "#E45756", "RP-LAIPD": "#54A24B"}


def save(fig, figure_dir: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot checkpoint accuracy-to-dispatch experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["data"]["output_dir"])
    metrics = pd.read_csv(root / "prediction" / "checkpoint_prediction_metrics.csv")
    dispatch = pd.read_csv(root / "summary" / "checkpoint_dispatch_summary.csv")
    figure_dir = root / "figures"
    data_dir = figure_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False,
    })

    training = metrics[["split", "model", "max_iter", "mae", "rmse", "wape"]].copy()
    training.to_csv(data_dir / "fig_checkpoint_training_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for split in ["validation", "test", "shifted_test"]:
        group = training[training["split"].eq(split)].sort_values("max_iter")
        ax.plot(group["max_iter"], group["wape"], marker="o", label=split.replace("_", " ").title())
    ax.set_xscale("log")
    ax.set(xlabel="HGB max_iter", ylabel="WAPE (lower is better)", title="Prediction checkpoint training curve")
    ax.legend(); save(fig, figure_dir, "fig_checkpoint_training_curve")

    test_metrics = metrics[metrics["split"].eq("test")][["model", "max_iter", "wape", "mae"]]
    performance = dispatch.merge(
        test_metrics, left_on=["prediction_model", "max_iter"], right_on=["model", "max_iter"], how="left"
    )
    performance.to_csv(data_dir / "fig_checkpoint_accuracy_performance.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method in ["IPD", "Prediction-only", "RP-LAIPD"]:
        group = performance[performance["method"].eq(method)].sort_values("wape")
        ax.errorbar(group["wape"], group["alg_over_opt_mean"], yerr=group["alg_over_opt_ci95"],
                    marker="o", label=method, color=COLORS[method])
    ax.set(xlabel="Test WAPE (lower is more accurate)", ylabel="ALG / OPT",
           title="Natural prediction accuracy versus dispatch performance")
    ax.legend(); save(fig, figure_dir, "fig_checkpoint_accuracy_performance")

    decision = performance[performance["method"].eq("RP-LAIPD")].copy()
    decision.to_csv(data_dir / "fig_checkpoint_demand_advice_error.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    points = ax.scatter(decision["prediction_error_mean"], decision["advice_error_mean"],
                        c=decision["alg_over_opt_mean"], cmap="viridis", s=55)
    for row in decision.itertuples():
        ax.annotate(str(int(row.max_iter)), (row.prediction_error_mean, row.advice_error_mean),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    fig.colorbar(points, ax=ax, label="RP-LAIPD ALG / OPT")
    ax.set(xlabel="Demand prediction error", ylabel="Advice error η",
           title="Demand error is not the same as decision error")
    save(fig, figure_dir, "fig_checkpoint_demand_advice_error")
    print(f"Checkpoint figures written to {figure_dir}")


if __name__ == "__main__":
    main()
