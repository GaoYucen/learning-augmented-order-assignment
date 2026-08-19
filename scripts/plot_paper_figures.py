#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


COLORS = {"IPD": "#4C78A8", "Prediction-only": "#E45756", "RP-LAIPD": "#54A24B"}
METHOD_ORDER = ["IPD", "Prediction-only", "RP-LAIPD"]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 140,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, figure_dir: Path, name: str) -> None:
    fig.tight_layout()
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    out = frame.groupby(groups, dropna=False).agg(
        runs=("slot_id", "size"), mean=("alg_over_opt", "mean"), std=("alg_over_opt", "std"),
        prediction_error=("prediction_error", "mean"), advice_error=("advice_error", "mean"),
    ).reset_index()
    out["ci95"] = 1.96 * out["std"].fillna(0) / np.sqrt(out["runs"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 regenerate all paper figures and figure-level CSV files.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["data"]["output_dir"])
    raw = pd.read_csv(root / "raw" / "distribution_shift_results.csv")
    runtime = pd.read_csv(root / "raw" / "runtime_scalability.csv")
    ablation = pd.read_csv(root / "raw" / "ablation_results.csv")
    figure_dir = root / "figures"
    data_dir = figure_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    # Main smoothness: theta controls how prediction trust changes under count error.
    frame = raw[(raw["prediction_corruption"].isin(["scale", "none"])) & raw["method"].eq("RP-LAIPD")].copy()
    frame["scale"] = np.where(frame["prediction_corruption"].eq("none"), 1.0, frame["corruption_strength"])
    table = aggregate(frame, ["scale", "theta"])
    table.to_csv(data_dir / "fig_main_smoothness.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for theta, group in table.groupby("theta"):
        group = group.sort_values("scale")
        ax.errorbar(group["scale"], group["mean"], yerr=group["ci95"], marker="o", label=f"θ={theta:g}")
    ax.set(xlabel="Count scaling", ylabel="ALG / OPT", title="Smooth degradation under count misspecification")
    ax.legend(ncol=2); save(fig, figure_dir, "fig_main_smoothness")

    # Consistency/robustness: performance versus measured prediction error.
    selected_theta = float(config["experiment"]["selected_theta"])
    chosen = raw[(raw["method"].isin(METHOD_ORDER)) & ((~raw["method"].eq("RP-LAIPD")) | raw["theta"].eq(selected_theta))]
    table = aggregate(chosen, ["prediction_corruption", "corruption_strength", "method"])
    table.to_csv(data_dir / "fig_consistency_robustness.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method in METHOD_ORDER:
        group = table[table["method"].eq(method)].sort_values("prediction_error")
        ax.scatter(group["prediction_error"], group["mean"], s=22, alpha=.8, label=method, color=COLORS[method])
    ax.set(xlabel="Normalized demand prediction error", ylabel="ALG / OPT", title="Consistency and robust floor")
    ax.legend(); save(fig, figure_dir, "fig_consistency_robustness")

    # Four controlled distribution shifts.
    table = aggregate(chosen, ["prediction_corruption", "corruption_strength", "method"])
    table.to_csv(data_dir / "fig_distribution_shift.csv", index=False)
    corruptions = ["scale", "permutation", "temporal_shift", "scarce_resource"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharey=True)
    for ax, corruption in zip(axes.flat, corruptions):
        panel = table[table["prediction_corruption"].eq(corruption)]
        if corruption == "scale":
            panel = pd.concat([panel, table[table["prediction_corruption"].eq("none")].assign(corruption_strength=1.0)])
        for method in METHOD_ORDER:
            group = panel[panel["method"].eq(method)].sort_values("corruption_strength")
            ax.plot(group["corruption_strength"], group["mean"], marker="o", label=method, color=COLORS[method])
        ax.set_title(corruption.replace("_", " ").title()); ax.set_xlabel("Corruption level")
    axes[0, 0].set_ylabel("ALG / OPT"); axes[1, 0].set_ylabel("ALG / OPT")
    axes[0, 1].legend(); save(fig, figure_dir, "fig_distribution_shift")

    # Overall clean comparison.
    clean = chosen[chosen["prediction_corruption"].eq("none")]
    table = aggregate(clean, ["method"])
    table.to_csv(data_dir / "fig_overall_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    table["order"] = table["method"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    table = table.sort_values("order")
    ax.bar(table["method"], table["mean"], yerr=table["ci95"], color=[COLORS[x] for x in table["method"]], capsize=3)
    ax.set(ylabel="ALG / OPT", title="Overall comparison on clean Test slots"); ax.tick_params(axis="x", rotation=15)
    save(fig, figure_dir, "fig_overall_comparison")

    # Runtime/scalability.
    measures = runtime.groupby(["dimension", "level", "method"], as_index=False).agg(
        lp_ms=("predictive_lp_time_ms", "mean"), runtime_ms=("slot_runtime_ms", "mean"),
        avg_latency_ms=("avg_request_latency_ms", "mean"), p95_latency_ms=("p95_request_latency_ms", "mean"),
    )
    measures.to_csv(data_dir / "fig_runtime.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for ax, dimension in zip(axes, ["N", "M", "K"]):
        panel = measures[measures["dimension"].eq(dimension)]
        for method in ["IPD", "RP-LAIPD"]:
            group = panel[panel["method"].eq(method)].sort_values("level")
            ax.plot(group["level"], group["avg_latency_ms"], marker="o", label=method, color=COLORS[method])
        ax.set(xlabel=dimension, ylabel="Mean request latency (ms)", title=f"Scale by {dimension}")
        if dimension in {"N", "K"}: ax.set_xscale("log")
    axes[-1].legend(); save(fig, figure_dir, "fig_runtime")

    # Paired ablations.
    table = aggregate(ablation, ["prediction_corruption", "corruption_strength", "ablation_component", "method"])
    table.to_csv(data_dir / "fig_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    panels = [("without_robust_branch", "Robust branch"), ("without_prediction_advice", "Prediction advice")]
    for ax, (component, title) in zip(axes, panels):
        panel = table[(table["prediction_corruption"].eq("scarce_resource")) &
                      (table["ablation_component"].isin(["full_model", component]))]
        for name, group in panel.groupby("ablation_component"):
            ax.plot(group["corruption_strength"], group["mean"], marker="o", label=name.replace("_", " "))
        ax.set(xlabel="Scarce-resource corruption", title=title)
    axes[0].set_ylabel("ALG / OPT"); axes[1].legend(); save(fig, figure_dir, "fig_ablation")

    # Representative slot: cumulative arrivals and station-level demand prediction.
    orders = pd.read_csv(root / "processed" / "experiment_orders.csv.gz")
    predictions = pd.read_csv(root / "prediction" / "predictions.csv")
    slot_id = clean.sort_values("slot_id")["slot_id"].iloc[0]
    slot_orders = orders[orders["slot_id"].eq(slot_id)].sort_values("arrival_minute")
    pred = predictions[(predictions["slot_id"].eq(slot_id)) & predictions["model"].eq("HistGradientBoosting")]
    observed = slot_orders.groupby("station_id").size().rename("observed_count").reset_index()
    station = pred[["type_id", "predicted_count"]].rename(columns={"type_id": "station_id"}).merge(observed, how="left").fillna(0)
    station["slot_id"] = slot_id
    station.to_csv(data_dir / "fig_representative_slot.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    axes[0].step(slot_orders["arrival_minute"], np.arange(1, len(slot_orders) + 1), where="post", label="Observed")
    axes[0].plot([0, 30], [0, pred["predicted_count"].sum()], "--", label="Predicted total (uniform within slot)")
    axes[0].set(xlabel="Minute in slot", ylabel="Cumulative requests", title=str(slot_id)); axes[0].legend()
    top = station.sort_values("observed_count", ascending=False).head(10)
    x = np.arange(len(top)); axes[1].bar(x-.2, top["observed_count"], .4, label="Observed")
    axes[1].bar(x+.2, top["predicted_count"], .4, label="Predicted")
    axes[1].set_xticks(x, top["station_id"].astype(int)); axes[1].set(xlabel="Destination station", ylabel="Requests", title="Top destinations")
    axes[1].legend(); save(fig, figure_dir, "fig_representative_slot")
    print(f"Figures and CSVs written to {figure_dir}")


if __name__ == "__main__":
    main()
