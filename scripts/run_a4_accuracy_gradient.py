#!/usr/bin/env python
"""A4 accuracy-gradient experiment on Chengdu predictions.

The plot answers: as prediction accuracy changes, how do ALG/OPT curves move
for IPD, Prediction-only, Static LP, and RP-LAIPD with different theta values?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import (
    greedy_dispatch,
    ipd_dispatch,
    offline_opt,
    prediction_only_dispatch,
    random_dispatch,
    rp_laipd_dispatch,
    static_bid_price_dispatch,
)
from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice
from rood_dasfaa2019.experiments import ChengduExperimentContext
from rood_dasfaa2019.learning.metrics import advice_error, dispatch_row, opt_type_bus_allocation, prediction_error
from scripts.run_final_prediction_theta_figures import CALIBRATION_SCALES, load_model_predictions


DEFAULT_THETAS = (0.2, 0.4, 0.6, 0.8)
HIGH_ACCURACY_TARGETS = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
FAMILY_BASE_MODELS = {
    "Historical average": "HistoricalAverage_scale_1.00",
    "RidgeAR": "RidgeAR_scale_1.00",
    "HGB": "HGB_iter_100",
}


def timed(fn):
    started = perf_counter()
    result = fn()
    return result, (perf_counter() - started) * 1000.0


def evenly_spaced_test_slots(context: ChengduExperimentContext, count: int) -> list[str]:
    slots = (
        context.demand.loc[
            context.demand["split"].eq("test"),
            ["slot_id", "service_date", "slot_in_day"],
        ]
        .drop_duplicates()
        .sort_values(["service_date", "slot_in_day"])["slot_id"]
        .astype(str)
        .tolist()
    )
    if count >= len(slots):
        return slots
    positions = np.linspace(0, len(slots) - 1, count, dtype=int)
    return [slots[position] for position in positions]


def append_high_accuracy_variants(
    context: ChengduExperimentContext,
    predictions: pd.DataFrame,
    catalog: pd.DataFrame,
    slot_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add oracle-blended variants for the 75%-100% accuracy range.

    The blend is computed against observed test-slot demand:
    new_prediction = (1 - alpha) * base_prediction + alpha * observed_count.
    At alpha=1 this becomes the 100% oracle prediction.
    """
    demand = context.demand.loc[context.demand["slot_id"].astype(str).isin(slot_ids)].copy()
    truth = demand[["slot_id", "type_id", "observed_count"]].copy()
    truth = truth.rename(columns={"observed_count": "truth_count"})
    truth["slot_id"] = truth["slot_id"].astype(str)
    parts = [predictions]
    rows = []

    for family, base_model in FAMILY_BASE_MODELS.items():
        base = predictions.loc[
            predictions["slot_id"].astype(str).isin(slot_ids) & predictions["model"].eq(base_model)
        ].copy()
        if base.empty:
            continue
        merged = base.merge(truth, on=["slot_id", "type_id"], how="left")
        base_error = (
            (merged["predicted_count"] - merged["truth_count"]).abs().sum()
            / max(float(merged["truth_count"].sum()), 1.0)
        )
        base_accuracy = max(0.0, 1.0 - float(base_error))

        for target in HIGH_ACCURACY_TARGETS:
            if target <= base_accuracy + 1e-9 and target < 1.0:
                continue
            alpha = 1.0 if np.isclose(target, 1.0) else (target - base_accuracy) / max(1.0 - base_accuracy, 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            variant = merged.copy()
            model_name = f"{base_model}_oracle_{int(round(target * 100)):03d}"
            variant["model"] = model_name
            variant["predicted_count"] = (
                (1.0 - alpha) * variant["predicted_count"] + alpha * variant["truth_count"]
            ).clip(lower=0.0)
            parts.append(variant.drop(columns=["truth_count"]))
            rows.append(
                {
                    "model": model_name,
                    "model_family": family,
                    "variant_label": f"oracle {int(round(target * 100))}%",
                    "variant_order": 1000 + int(round(target * 100)),
                    "max_iter": np.nan,
                    "is_reference_variant": False,
                }
            )

    if rows:
        catalog = pd.concat([catalog, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return pd.concat(parts, ignore_index=True, sort=False), catalog.reset_index(drop=True)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(
        [
            "prediction_model",
            "model_family",
            "variant_label",
            "variant_order",
            "method",
            "theta",
        ],
        dropna=False,
        as_index=False,
    )
    summary = grouped.agg(
        runs=("slot_id", "nunique"),
        prediction_error_mean=("prediction_error", "mean"),
        advice_error_mean=("advice_error", "mean"),
        alg_over_opt_mean=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"),
        accepted_value_mean=("accepted_value", "mean"),
        served_orders_mean=("served_orders", "mean"),
        served_passengers_mean=("served_passengers", "mean"),
        occupancy_mean=("occupancy", "mean"),
        violations_mean=("violations", "mean"),
        slot_runtime_ms_mean=("slot_runtime_ms", "mean"),
        avg_request_latency_ms_mean=("avg_request_latency_ms", "mean"),
    )
    summary["alg_over_opt_std"] = summary["alg_over_opt_std"].fillna(0.0)
    summary["alg_over_opt_ci95"] = (
        1.96 * summary["alg_over_opt_std"] / np.sqrt(summary["runs"].clip(lower=1))
    )
    summary["prediction_accuracy_mean"] = (1.0 - summary["prediction_error_mean"]).clip(lower=0.0)
    return summary.sort_values(["prediction_accuracy_mean", "method", "theta"]).reset_index(drop=True)


def best_theta_by_accuracy(summary: pd.DataFrame) -> pd.DataFrame:
    rp = summary.loc[summary["method"].eq("RP-LAIPD")].copy()
    idx = rp.groupby("prediction_model")["alg_over_opt_mean"].idxmax()
    best = rp.loc[idx].sort_values("prediction_accuracy_mean").reset_index(drop=True)
    return best[
        [
            "prediction_model",
            "model_family",
            "variant_label",
            "prediction_accuracy_mean",
            "prediction_error_mean",
            "theta",
            "alg_over_opt_mean",
            "alg_over_opt_ci95",
        ]
    ]


def curve_label(row: pd.Series) -> str:
    method = str(row["method"])
    if method == "RP-LAIPD":
        return f"RP-LAIPD theta={float(row['theta']):.1f}"
    return method


def save_figure(fig, path: Path) -> None:
    try:
        fig.savefig(path)
    except PermissionError:
        stamped = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        fig.savefig(stamped)
        print(f"{path} is locked; wrote {stamped} instead", flush=True)


def plot_accuracy_gradient(
    summary: pd.DataFrame,
    best_theta: pd.DataFrame,
    output_dir: Path,
    predictor_label: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_df = summary.copy()
    plot_df["curve"] = plot_df.apply(curve_label, axis=1)

    curve_order = [
        "Offline OPT",
        "Greedy",
        "Random",
        "IPD",
        "Prediction-only",
        "Static LP / bid-price",
        "RP-LAIPD theta=0.2",
        "RP-LAIPD theta=0.4",
        "RP-LAIPD theta=0.6",
        "RP-LAIPD theta=0.8",
    ]
    styles = {
        "Offline OPT": dict(color="#111827", linestyle="--", marker=None, linewidth=1.8, alpha=0.75),
        "Greedy": dict(color="#4C78A8", linestyle="-", marker="s", linewidth=1.8),
        "Random": dict(color="#72B7B2", linestyle="-", marker="^", linewidth=1.8),
        "IPD": dict(color="#F58518", linestyle="-", marker="D", linewidth=2.0),
        "Prediction-only": dict(color="#E45756", linestyle="-", marker="v", linewidth=2.0),
        "Static LP / bid-price": dict(color="#B279A2", linestyle="-", marker="P", linewidth=1.8),
        "RP-LAIPD theta=0.2": dict(color="#2F7D32", linestyle="-", marker="o", linewidth=2.0),
        "RP-LAIPD theta=0.4": dict(color="#59A14F", linestyle="-", marker="o", linewidth=2.0),
        "RP-LAIPD theta=0.6": dict(color="#8CD17D", linestyle="-", marker="o", linewidth=2.0),
        "RP-LAIPD theta=0.8": dict(color="#B6992D", linestyle="-", marker="o", linewidth=2.0),
    }

    fig, ax = plt.subplots(figsize=(14.8, 7.6), dpi=220, constrained_layout=True)
    for curve in curve_order:
        sub = plot_df.loc[plot_df["curve"].eq(curve)].sort_values("prediction_accuracy_mean")
        if sub.empty:
            continue
        style = styles[curve]
        ax.errorbar(
            sub["prediction_accuracy_mean"],
            sub["alg_over_opt_mean"],
            yerr=sub["alg_over_opt_ci95"],
            capsize=3,
            markersize=5 if style.get("marker") else 0,
            label=curve,
            **style,
        )

    ax.plot(
        best_theta["prediction_accuracy_mean"],
        best_theta["alg_over_opt_mean"],
        color="#000000",
        linestyle=":",
        marker="*",
        markersize=9,
        linewidth=2.5,
        label="Best RP-LAIPD theta",
    )

    title = "A4：不同预测准确率下的算法最优比对比"
    if predictor_label:
        title += f"（{predictor_label}）"
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("预测准确率 = 1 - WAPE（越高越准）", fontsize=11)
    ax.set_ylabel("ALG / OPT（越高越好）", fontsize=11)
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.set_ylim(max(0.0, float(plot_df["alg_over_opt_mean"].min()) - 0.05), 1.04)
    xmin = max(0.0, float(plot_df["prediction_accuracy_mean"].min()) - 0.03)
    xmax = min(1.0, float(plot_df["prediction_accuracy_mean"].max()) + 0.03)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(ncol=1, fontsize=8.5, frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.text(
        0.01,
        0.01,
        "theta=0 完全使用预测建议；theta=1 退化为 IPD。误差棒为 test slots 上的 95% CI。",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    save_figure(fig, output_dir / "a4_prediction_accuracy_vs_alg_over_opt.png")
    save_figure(fig, output_dir / "a4_prediction_accuracy_vs_alg_over_opt.pdf")
    plt.close(fig)


def plot_accuracy_gradient_by_predictor(summary: pd.DataFrame, best_theta: pd.DataFrame, output_dir: Path) -> None:
    by_dir = output_dir / "by_predictor"
    by_dir.mkdir(parents=True, exist_ok=True)
    for family in ["Historical average", "RidgeAR", "HGB"]:
        family_summary = summary.loc[summary["model_family"].eq(family)].copy()
        family_best = best_theta.loc[best_theta["model_family"].eq(family)].copy()
        if family_summary.empty:
            continue
        safe_name = family.lower().replace(" ", "_")
        plot_accuracy_gradient(family_summary, family_best, by_dir / safe_name, predictor_label=family)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--thetas", nargs="+", type=float, default=list(DEFAULT_THETAS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    context = ChengduExperimentContext.load(config)
    root = Path(config["data"]["output_dir"])
    predictions, catalog = load_model_predictions(root)
    output_dir = args.output_dir or root / "a4_accuracy_gradient"
    output_dir.mkdir(parents=True, exist_ok=True)
    slot_ids = evenly_spaced_test_slots(context, args.slots)
    predictions, catalog = append_high_accuracy_variants(context, predictions, catalog, slot_ids)
    context.predictions = predictions
    catalog_lookup = catalog.set_index("model").to_dict("index")
    seed = int(config["data"]["random_seed"])
    rows = []
    for slot_number, slot_id in enumerate(slot_ids, start=1):
        print(f"[{slot_number}/{len(slot_ids)}] Test slot {slot_id}", flush=True)
        orders = context.slot_orders(slot_id)
        opt_result, opt_ms = timed(lambda: offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg))
        total_capacity = sum(bus.capacity for bus in context.buses)

        for model in catalog["model"]:
            model = str(model)
            predicted = context.prediction_for_slot(slot_id, model)
            advice, lp_ms = timed(
                lambda: build_predictive_lp_advice(
                    orders,
                    context.buses,
                    len(context.stations),
                    context.algorithm_cfg,
                    predicted_counts=predicted,
                    type_values=context.type_values,
                )
            )
            pred_error = prediction_error(orders, len(context.stations), predicted)
            adv_error = advice_error(advice, opt_type_bus_allocation(orders, opt_result))
            meta = catalog_lookup[model]

            methods = [
                ("Random", np.nan, lambda: random_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg, seed)),
                ("Greedy", np.nan, lambda: greedy_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
                ("IPD", np.nan, lambda: ipd_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
                (
                    "Prediction-only",
                    np.nan,
                    lambda: prediction_only_dispatch(
                        orders,
                        context.buses,
                        len(context.stations),
                        context.algorithm_cfg,
                        advice=advice,
                    ),
                ),
                (
                    "Static LP / bid-price",
                    np.nan,
                    lambda: static_bid_price_dispatch(
                        orders,
                        context.buses,
                        len(context.stations),
                        context.algorithm_cfg,
                        predicted_counts=predicted,
                        type_values=context.type_values,
                        avg_passengers=context.type_passengers,
                    ),
                ),
            ]
            for theta in args.thetas:
                theta_cfg = {**context.algorithm_cfg, "theta": float(theta)}
                methods.append(
                    (
                        "RP-LAIPD",
                        float(theta),
                        lambda cfg=theta_cfg: rp_laipd_dispatch(
                            orders,
                            context.buses,
                            len(context.stations),
                            cfg,
                            advice=advice,
                        ),
                    )
                )
            methods.append(("Offline OPT", np.nan, lambda: opt_result))

            for method, theta, fn in methods:
                result, runtime_ms = timed(fn)
                if method == "Offline OPT":
                    runtime_ms = opt_ms
                violations = count_constraint_violations(
                    result.accepted,
                    orders,
                    context.buses,
                    len(context.stations),
                    context.algorithm_cfg,
                )
                row = dispatch_row(
                    slot_id=slot_id,
                    method=method,
                    theta=theta,
                    prediction_scale=1.0,
                    corruption="none",
                    corruption_strength=0.0,
                    pred_error=pred_error,
                    adv_error=adv_error,
                    orders=orders,
                    result=result,
                    opt_result=opt_result,
                    elapsed_ms=runtime_ms,
                    total_capacity=total_capacity,
                    violations=violations,
                    seed=seed,
                    split="test",
                    prediction_model=model,
                    prediction_time_ms=0.0,
                    predictive_lp_time_ms=lp_ms,
                    slot_preparation_ms=0.0,
                )
                row["model_family"] = meta["model_family"]
                row["variant_label"] = meta["variant_label"]
                row["variant_order"] = meta["variant_order"]
                rows.append(row)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    best_theta = best_theta_by_accuracy(summary)

    raw.to_csv(output_dir / "a4_accuracy_gradient_raw.csv", index=False)
    summary.to_csv(output_dir / "a4_accuracy_gradient_summary.csv", index=False)
    best_theta.to_csv(output_dir / "a4_best_theta_by_accuracy.csv", index=False)
    plot_accuracy_gradient(summary, best_theta, output_dir)
    plot_accuracy_gradient_by_predictor(summary, best_theta, output_dir)

    print(summary[["method", "theta", "prediction_accuracy_mean", "alg_over_opt_mean"]].to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
