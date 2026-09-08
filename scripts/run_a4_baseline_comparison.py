#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

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
from rood_dasfaa2019.learning.metrics import (
    advice_error,
    dispatch_row,
    opt_type_bus_allocation,
    prediction_error,
)


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


def timed(fn):
    from time import perf_counter

    started = perf_counter()
    result = fn()
    return result, (perf_counter() - started) * 1000.0


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["method", "theta"], dropna=False, as_index=False)
    summary = grouped.agg(
        runs=("slot_id", "count"),
        prediction_error_mean=("prediction_error", "mean"),
        prediction_error_std=("prediction_error", "std"),
        advice_error_mean=("advice_error", "mean"),
        advice_error_std=("advice_error", "std"),
        accepted_value_mean=("accepted_value", "mean"),
        accepted_value_std=("accepted_value", "std"),
        alg_over_opt_mean=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"),
        served_orders_mean=("served_orders", "mean"),
        served_orders_std=("served_orders", "std"),
        served_passengers_mean=("served_passengers", "mean"),
        served_passengers_std=("served_passengers", "std"),
        occupancy_mean=("occupancy", "mean"),
        occupancy_std=("occupancy", "std"),
        high_value_rejected_mean=("high_value_rejected", "mean"),
        high_value_rejected_std=("high_value_rejected", "std"),
        violations_mean=("violations", "mean"),
        violations_std=("violations", "std"),
        slot_runtime_ms_mean=("slot_runtime_ms", "mean"),
        slot_runtime_ms_std=("slot_runtime_ms", "std"),
        avg_request_latency_ms_mean=("avg_request_latency_ms", "mean"),
        avg_request_latency_ms_std=("avg_request_latency_ms", "std"),
    )
    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    for metric in [
        "prediction_error",
        "advice_error",
        "accepted_value",
        "alg_over_opt",
        "served_orders",
        "served_passengers",
        "occupancy",
        "high_value_rejected",
        "violations",
        "slot_runtime_ms",
        "avg_request_latency_ms",
    ]:
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / np.sqrt(summary["runs"].clip(lower=1))
    return summary.sort_values(["method", "theta"], na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 baseline comparison using the trained Chengdu predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--model", default="HistGradientBoosting")
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--theta", type=float, help="Override RP-LAIPD theta.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    theta = float(args.theta if args.theta is not None else config["experiment"]["selected_theta"])
    slot_ids = evenly_spaced_test_slots(context, args.slots)
    output_dir = args.output_dir or Path(config["data"]["output_dir"]) / "a4_baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for slot_number, slot_id in enumerate(slot_ids, start=1):
        print(f"[{slot_number}/{len(slot_ids)}] Test slot {slot_id}", flush=True)
        orders = context.slot_orders(slot_id)
        opt_result = offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        predicted = context.prediction_for_slot(slot_id, args.model)
        advice = build_predictive_lp_advice(
            orders,
            context.buses,
            len(context.stations),
            context.algorithm_cfg,
            predicted_counts=predicted,
            type_values=context.type_values,
        )
        observed = context.observed_counts(slot_id)
        pred_error = prediction_error(orders, len(context.stations), predicted)
        adv_error = advice_error(advice, opt_type_bus_allocation(orders, opt_result))

        methods = [
            ("Random", np.nan, lambda: random_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg, int(config["data"]["random_seed"]))),
            ("Greedy", np.nan, lambda: greedy_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
            ("IPD", np.nan, lambda: ipd_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
            ("Prediction-only", np.nan, lambda: prediction_only_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg, advice=advice)),
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
            (
                "RP-LAIPD",
                theta,
                lambda: rp_laipd_dispatch(
                    orders,
                    context.buses,
                    len(context.stations),
                    {**context.algorithm_cfg, "theta": theta},
                    advice=advice,
                ),
            ),
        ]

        for method, theta_value, run in methods:
            result, elapsed_ms = timed(run)
            violations = count_constraint_violations(
                result.accepted,
                orders,
                context.buses,
                len(context.stations),
                context.algorithm_cfg,
            )
            rows.append(
                dispatch_row(
                    slot_id=slot_id,
                    method=method,
                    theta=theta_value,
                    prediction_scale=1.0,
                    corruption="none",
                    corruption_strength=0.0,
                    pred_error=pred_error,
                    adv_error=adv_error,
                    orders=orders,
                    result=result,
                    opt_result=opt_result,
                    elapsed_ms=elapsed_ms,
                    total_capacity=sum(bus.capacity for bus in context.buses),
                    violations=violations,
                    seed=int(config["data"]["random_seed"]),
                    split="test",
                    prediction_model=args.model,
                    prediction_time_ms=0.0,
                    predictive_lp_time_ms=0.0,
                    slot_preparation_ms=0.0,
                )
            )

        rows.append(
            dispatch_row(
                slot_id=slot_id,
                method="Offline OPT",
                theta=np.nan,
                prediction_scale=1.0,
                corruption="none",
                corruption_strength=0.0,
                pred_error=pred_error,
                adv_error=adv_error,
                orders=orders,
                result=opt_result,
                opt_result=opt_result,
                elapsed_ms=0.0,
                total_capacity=sum(bus.capacity for bus in context.buses),
                violations=count_constraint_violations(
                    opt_result.accepted,
                    orders,
                    context.buses,
                    len(context.stations),
                    context.algorithm_cfg,
                ),
                seed=int(config["data"]["random_seed"]),
                split="test",
                prediction_model=args.model,
                prediction_time_ms=0.0,
                predictive_lp_time_ms=0.0,
                slot_preparation_ms=0.0,
            )
        )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw_path = output_dir / "a4_baseline_raw.csv"
    summary_path = output_dir / "a4_baseline_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
