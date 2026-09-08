#!/usr/bin/env python
"""Run the final paired Chengdu experiment and produce exactly two figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import offline_opt
from rood_dasfaa2019.experiments import ChengduExperimentContext, run_chengdu_slot


THETAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CALIBRATION_SCALES = (0.40, 0.55, 0.70, 0.85, 1.00)


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


def load_model_predictions(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(root / "prediction" / "predictions.csv")
    base = base.loc[base["model"].isin(["HistoricalAverage", "RidgeAR"])].copy()
    base["max_iter"] = np.nan
    checkpoints = pd.read_csv(root / "prediction" / "checkpoint_predictions.csv")
    prediction_parts = [base, checkpoints]
    catalog_rows = []

    family_labels = {
        "HistoricalAverage": "Historical average",
        "RidgeAR": "RidgeAR",
    }
    for family, family_label in family_labels.items():
        family_rows = base.loc[base["model"].eq(family)]
        for order, scale in enumerate(CALIBRATION_SCALES):
            model_name = f"{family}_scale_{scale:.2f}"
            variant = family_rows.copy()
            variant["model"] = model_name
            variant["predicted_count"] = variant["predicted_count"] * scale
            prediction_parts.append(variant)
            catalog_rows.append(
                {
                    "model": model_name,
                    "model_family": family_label,
                    "variant_label": f"scale={scale:.2f}",
                    "variant_order": order,
                    "max_iter": np.nan,
                    "is_reference_variant": bool(np.isclose(scale, 1.0)),
                }
            )

    for row in checkpoints[["model", "max_iter"]].drop_duplicates().sort_values("max_iter").itertuples(index=False):
        catalog_rows.append(
            {
                "model": str(row.model),
                "model_family": "HGB",
                "variant_label": f"iter={int(row.max_iter)}",
                "variant_order": int(row.max_iter),
                "max_iter": int(row.max_iter),
                "is_reference_variant": True,
            }
        )

    combined = pd.concat(prediction_parts, ignore_index=True, sort=False)
    catalog = pd.DataFrame(catalog_rows)
    return combined, catalog.reset_index(drop=True)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(
        [
            "prediction_model", "model_family", "variant_label", "variant_order",
            "max_iter", "is_reference_variant", "method", "theta",
        ],
        dropna=False,
        as_index=False,
    ).agg(
        effective_slots=("slot_id", "nunique"),
        prediction_wape=("prediction_error", "mean"),
        advice_error=("advice_error", "mean"),
        alg_over_opt=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"),
        accepted_value=("accepted_value", "mean"),
        violations=("violations", "sum"),
    )
    grouped["alg_over_opt_std"] = grouped["alg_over_opt_std"].fillna(0.0)
    grouped["alg_over_opt_ci95"] = (
        1.96
        * grouped["alg_over_opt_std"]
        / np.sqrt(grouped["effective_slots"].clip(lower=1))
    )
    return grouped.sort_values(["prediction_wape", "method", "theta"]).reset_index(drop=True)


def validate_theta_endpoints(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["slot_id", "prediction_model"]
    prediction_only = raw.loc[raw["method"].eq("Prediction-only"), keys + ["accepted_value"]].rename(
        columns={"accepted_value": "prediction_only_value"}
    )
    theta_zero = raw.loc[
        raw["method"].eq("RP-LAIPD") & raw["theta"].eq(0.0),
        keys + ["accepted_value"],
    ].rename(columns={"accepted_value": "theta_0_value"})
    ipd = raw.loc[raw["method"].eq("IPD"), keys + ["accepted_value"]].rename(
        columns={"accepted_value": "ipd_value"}
    )
    theta_one = raw.loc[
        raw["method"].eq("RP-LAIPD") & raw["theta"].eq(1.0),
        keys + ["accepted_value"],
    ].rename(columns={"accepted_value": "theta_1_value"})
    result = prediction_only.merge(theta_zero, on=keys).merge(ipd, on=keys).merge(theta_one, on=keys)
    result["theta_0_matches_prediction_only"] = np.isclose(
        result["theta_0_value"], result["prediction_only_value"]
    )
    result["theta_1_matches_ipd"] = np.isclose(result["theta_1_value"], result["ipd_value"])
    if not result[["theta_0_matches_prediction_only", "theta_1_matches_ipd"]].all().all():
        raise AssertionError("Theta endpoint semantics failed; figures were not generated.")
    return result


def style_axis(axis, x_values: pd.Series, y_values: pd.Series) -> None:
    axis.grid(alpha=0.22, linewidth=0.8)
    # Use the observed WAPE range instead of wasting space below the data.
    x_lower = float(np.floor(float(x_values.min()) * 20.0) / 20.0)
    x_upper = float(np.ceil(float(x_values.max()) * 20.0) / 20.0)
    axis.set_xlim(x_lower, x_upper)
    axis.set_xticks(np.arange(x_lower, x_upper + 0.001, 0.05))
    lower = max(0.0, float(y_values.min()) - 0.04)
    axis.set_ylim(lower, min(1.02, max(1.0, float(y_values.max()) + 0.03)))
    axis.set_ylabel("Dispatch performance (ALG / OPT)")
    axis.set_xlabel("Test WAPE (lower = more accurate prediction)")


def plot_model_accuracy(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    selected_theta: float,
    output_dir: Path,
) -> None:
    """Plot prediction accuracy across variants for each predictor family."""
    fig, axis = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    selected = summary.loc[
        summary["method"].eq("RP-LAIPD") & summary["theta"].eq(selected_theta)
    ].copy()
    selected["prediction_accuracy"] = (1.0 - selected["prediction_wape"]).clip(lower=0.0)
    family_styles = {
        "Historical average": ("#F58518", "s"),
        "RidgeAR": ("#4C78A8", "D"),
        "HGB": ("#54A24B", "o"),
    }
    for family, (color, marker) in family_styles.items():
        family_rows = selected.loc[selected["model_family"].eq(family)].sort_values("variant_order")
        stages = np.arange(1, len(family_rows) + 1)
        axis.plot(
            stages,
            family_rows["prediction_accuracy"],
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=6,
            label=family,
        )
        for stage, row in zip(stages, family_rows.itertuples()):
            axis.annotate(
                row.variant_label.replace("scale=", "x").replace("iter=", "i"),
                (stage, row.prediction_accuracy),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    axis.set_xlim(0.7, 8.3)
    axis.set_xticks(np.arange(1, 9))
    axis.set_xlabel("Model variant / training stage")
    axis.set_ylabel("Prediction accuracy (1 - Test WAPE)")
    axis.set_ylim(
        max(0.0, float(selected["prediction_accuracy"].min()) - 0.05),
        min(1.0, float(selected["prediction_accuracy"].max()) + 0.06),
    )
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.set_title("Prediction accuracy across model variants")
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.01,
        0.01,
        "Higher is more accurate. x = calibration scale; i = HGB training iterations.",
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.savefig(output_dir / "figure_1_model_accuracy_dispatch.png", dpi=220)
    fig.savefig(output_dir / "figure_1_model_accuracy_dispatch.pdf")
    plt.close(fig)


def station_frame(
    context: ChengduExperimentContext,
    slot_id: str,
    models: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    observed = context.observed_counts(slot_id)
    rows = []
    for station in range(len(context.stations)):
        row = {"slot_id": slot_id, "station_id": station, "observed": observed.get(station, 0.0)}
        for model in models:
            row[labels[model]] = context.prediction_for_slot(slot_id, model).get(station, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_station_and_theta(
    stations: pd.DataFrame,
    summary: pd.DataFrame,
    slot_id: str,
    output_dir: Path,
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 6.4), constrained_layout=True)

    left.bar(
        stations["station_id"], stations["observed"],
        color="#B9BDC5", alpha=0.8, label="Observed",
    )
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for column, color in zip(
        [column for column in stations.columns if column not in {"slot_id", "station_id", "observed"}],
        colors,
    ):
        left.plot(
            stations["station_id"], stations[column], marker="o",
            markersize=3.5, linewidth=1.5, color=color, label=column,
        )
    left.set_title(f"Observed and predicted demand by station\nrepresentative Test slot: {slot_id}")
    left.set_xlabel("Destination station ID")
    left.set_ylabel("Order count")
    left.set_xticks(stations["station_id"][::2])
    left.grid(axis="y", alpha=0.22)
    left.legend(ncol=2, fontsize=9)

    # Keep Figure 2 comparable with the previous version: all natural HGB
    # checkpoints plus the unscaled Historical-average and RidgeAR forecasts.
    theta_rows = summary.loc[
        summary["method"].eq("RP-LAIPD") & summary["is_reference_variant"]
    ].copy()
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(THETAS)))
    for theta, color in zip(THETAS, colors):
        selected = theta_rows.loc[theta_rows["theta"].eq(theta)].sort_values("prediction_wape")
        right.plot(
            selected["prediction_wape"], selected["alg_over_opt"],
            marker="o", markersize=4, linewidth=1.8, color=color, label=f"theta={theta:g}",
        )
    style_axis(right, theta_rows["prediction_wape"], theta_rows["alg_over_opt"])
    right.set_title("Prediction-error sensitivity under different theta values")
    right.legend(ncol=2, fontsize=9)
    right.text(
        0.01, 0.01,
        "theta=0: fully prediction-driven   |   theta=1: fully robust/IPD",
        transform=right.transAxes, fontsize=9, color="#444444",
    )

    fig.savefig(output_dir / "figure_2_station_prediction_and_theta.png", dpi=220)
    fig.savefig(output_dir / "figure_2_station_prediction_and_theta.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--slots", type=int, help="Number of evenly-spaced Test slots.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    root = Path(config["data"]["output_dir"])
    predictions, catalog = load_model_predictions(root)
    context.predictions = predictions
    slot_count = args.slots or int(config["prediction"]["checkpoint_test_slots"])
    slot_ids = evenly_spaced_test_slots(context, slot_count)
    selected_theta = float(config["experiment"]["selected_theta"])

    catalog_lookup = catalog.set_index("model").to_dict("index")
    rows = []
    for slot_number, slot_id in enumerate(slot_ids, start=1):
        print(f"[{slot_number}/{len(slot_ids)}] Test slot {slot_id}", flush=True)
        orders = context.slot_orders(slot_id)
        opt_result = offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        for model in catalog["model"]:
            model_rows = run_chengdu_slot(
                context,
                slot_id,
                str(model),
                thetas=THETAS,
                seed=int(config["data"]["random_seed"]),
                opt_result=opt_result,
            )
            for row in model_rows:
                row["model_family"] = catalog_lookup[model]["model_family"]
                row["variant_label"] = catalog_lookup[model]["variant_label"]
                row["variant_order"] = catalog_lookup[model]["variant_order"]
                row["max_iter"] = catalog_lookup[model]["max_iter"]
                row["is_reference_variant"] = catalog_lookup[model]["is_reference_variant"]
            rows.extend(model_rows)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    validation = validate_theta_endpoints(raw)

    # Pick a representative slot by median total demand, not by performance.
    totals = (
        context.demand.loc[context.demand["slot_id"].isin(slot_ids)]
        .groupby("slot_id")["observed_count"].sum().sort_values()
    )
    representative_slot = str(totals.index[len(totals) // 2])
    station_models = ["HistoricalAverage", "RidgeAR", "HGB_iter_100"]
    station_labels = {
        "HistoricalAverage": "Historical avg.",
        "RidgeAR": "RidgeAR",
        "HGB_iter_100": "HGB-100",
    }
    stations = station_frame(context, representative_slot, station_models, station_labels)

    output_dir = root / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "final_model_theta_raw.csv", index=False)
    summary.to_csv(output_dir / "final_model_theta_summary.csv", index=False)
    validation.to_csv(output_dir / "theta_endpoint_validation.csv", index=False)
    stations.to_csv(output_dir / "station_prediction_source.csv", index=False)
    plot_model_accuracy(raw, summary, selected_theta, output_dir)
    plot_station_and_theta(stations, summary, representative_slot, output_dir)

    print(f"Theta endpoint checks: {len(validation)} / {len(validation)} passed")
    print(f"Figures and source CSV files: {output_dir}")


if __name__ == "__main__":
    main()
