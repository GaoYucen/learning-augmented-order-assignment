#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from rood_dasfaa2019.algorithms import (
    greedy_dispatch,
    ipd_dispatch,
    offline_opt,
    prediction_only_dispatch,
    random_dispatch,
    rp_laipd_dispatch,
)
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice
from rood_dasfaa2019.simulation.entities import Bus, Order, Station
from rood_dasfaa2019.simulation.generator import generate_instance
from rood_dasfaa2019.utils.config import load_experiment


def type_counts(orders, station_count):
    counts = {station: 0.0 for station in range(station_count)}
    for order in orders:
        counts[order.destination] += order.passengers
    return counts


def predicted_type_counts(orders, station_count, prediction_scale, corruption, corruption_strength):
    truth = type_counts(orders, station_count)
    predicted = {station: truth[station] * prediction_scale for station in truth}
    if corruption == "permutation" and corruption_strength > 0:
        selected = list(range(station_count))[: max(0, int(round(station_count * corruption_strength)))]
        if len(selected) >= 2:
            values = [predicted[selected[-1]], *[predicted[station] for station in selected[:-1]]]
            for station, value in zip(selected, values):
                predicted[station] = value
    elif corruption == "adversarial_concentration" and corruption_strength > 0:
        ordered = sorted(range(station_count), key=lambda station: predicted[station])
        source_count = max(1, int(round(station_count * min(max(corruption_strength, 0.0), 1.0) / 2)))
        sources = ordered[-source_count:]
        target = ordered[0]
        moved = sum(predicted[source] for source in sources)
        for source in sources:
            predicted[source] = 0.0
        predicted[target] += moved
    return predicted


def prediction_error(orders, station_count, prediction_scale, corruption, corruption_strength):
    truth = type_counts(orders, station_count)
    predicted = predicted_type_counts(orders, station_count, prediction_scale, corruption, corruption_strength)
    absolute_error = sum(abs(predicted[station] - truth[station]) for station in truth)
    total = max(sum(truth.values()), 1.0)
    return absolute_error / total


def opt_type_bus_allocation(orders, opt_result):
    order_by_id = {order.id: order for order in orders}
    allocation = {}
    for order_id, bus_id in opt_result.accepted.items():
        order = order_by_id[order_id]
        key = (order.destination, bus_id)
        allocation[key] = allocation.get(key, 0.0) + order.passengers
    return allocation


def advice_error(advice, opt_allocation):
    keys = set(advice) | set(opt_allocation)
    absolute_error = sum(abs(advice.get(key, 0.0) - opt_allocation.get(key, 0.0)) for key in keys)
    total = max(sum(opt_allocation.values()), 1.0)
    return absolute_error / total


def high_value_rejected_count(orders, result):
    if not orders:
        return 0
    threshold = pd.Series([order.priority for order in orders]).quantile(0.75)
    accepted = set(result.accepted)
    return sum(1 for order in orders if order.priority >= threshold and order.id not in accepted)


def dispatch_row(
    slot_id,
    method,
    theta,
    prediction_scale,
    corruption,
    corruption_strength,
    pred_error,
    adv_error,
    orders,
    result,
    opt_result,
    elapsed_ms,
):
    dispatch = result.dispatch
    opt = opt_result.dispatch
    return {
        "slot_id": slot_id,
        "method": method,
        "theta": theta,
        "prediction_scale": prediction_scale,
        "prediction_corruption": corruption,
        "corruption_strength": corruption_strength,
        "prediction_error": pred_error,
        "advice_error": adv_error,
        "accepted_value": dispatch.objective,
        "offline_value": opt.objective,
        "alg_over_opt": dispatch.objective / max(opt.objective, 1e-9),
        "served_orders": len(dispatch.accepted),
        "served_passengers": dispatch.passengers,
        "occupancy": dispatch.passengers / max(opt.passengers, 1),
        "high_value_rejected": high_value_rejected_count(orders, result),
        "violations": 0,
        "slot_runtime_ms": elapsed_ms,
        "avg_request_latency_ms": elapsed_ms / max(len(dispatch.accepted), 1),
    }


def timed(fn):
    start = perf_counter()
    result = fn()
    return result, (perf_counter() - start) * 1000


def run_one_slot(cfg, seed, prediction_scales, corruption_strengths, corruption, thetas, slot_id):
    stations, orders, buses = generate_instance(cfg, seed)
    return run_instance(stations, orders, buses, cfg, seed, prediction_scales, corruption_strengths, corruption, thetas, slot_id)


def run_instance(stations, orders, buses, cfg, seed, prediction_scales, corruption_strengths, corruption, thetas, slot_id):
    opt_result, _ = timed(lambda: offline_opt(orders, buses, len(stations), cfg))
    opt_allocation = opt_type_bus_allocation(orders, opt_result)
    rows = []

    random_result, random_ms = timed(lambda: random_dispatch(orders, buses, len(stations), cfg, seed))
    greedy_result, greedy_ms = timed(lambda: greedy_dispatch(orders, buses, len(stations), cfg))
    ipd_result, ipd_ms = timed(lambda: ipd_dispatch(orders, buses, len(stations), cfg))
    scenarios = [(scale, 0.0 if corruption == "scale" else strength) for scale in prediction_scales for strength in corruption_strengths]
    if corruption == "scale":
        scenarios = [(scale, abs(scale - 1.0)) for scale in prediction_scales]

    for scale, strength in scenarios:
        scaled_cfg = dict(cfg)
        scaled_cfg["prediction_scale"] = scale
        scaled_cfg["prediction_corruption"] = corruption
        scaled_cfg["corruption_strength"] = strength
        advice = build_predictive_lp_advice(orders, buses, len(stations), scaled_cfg)
        pred_error = prediction_error(orders, len(stations), scale, corruption, strength)
        adv_error = advice_error(advice, opt_allocation)
        rows.append(
            dispatch_row(slot_id, "Random", "", scale, corruption, strength, pred_error, adv_error, orders, random_result, opt_result, random_ms)
        )
        rows.append(
            dispatch_row(slot_id, "Greedy", "", scale, corruption, strength, pred_error, adv_error, orders, greedy_result, opt_result, greedy_ms)
        )
        rows.append(
            dispatch_row(slot_id, "IPD", "", scale, corruption, strength, pred_error, adv_error, orders, ipd_result, opt_result, ipd_ms)
        )

    for scale, strength in scenarios:
        scaled_cfg = dict(cfg)
        scaled_cfg["prediction_scale"] = scale
        scaled_cfg["prediction_corruption"] = corruption
        scaled_cfg["corruption_strength"] = strength
        advice = build_predictive_lp_advice(orders, buses, len(stations), scaled_cfg)
        pred_error = prediction_error(orders, len(stations), scale, corruption, strength)
        adv_error = advice_error(advice, opt_allocation)
        result, elapsed_ms = timed(lambda: prediction_only_dispatch(orders, buses, len(stations), scaled_cfg))
        rows.append(
            dispatch_row(
                slot_id,
                "Prediction-only",
                "",
                scale,
                corruption,
                strength,
                pred_error,
                adv_error,
                orders,
                result,
                opt_result,
                elapsed_ms,
            )
        )

        for theta in thetas:
            theta_cfg = dict(scaled_cfg)
            theta_cfg["theta"] = theta
            result, elapsed_ms = timed(lambda: rp_laipd_dispatch(orders, buses, len(stations), theta_cfg))
            rows.append(
                dispatch_row(
                    slot_id,
                    "RP-LAIPD",
                    theta,
                    scale,
                    corruption,
                    strength,
                    pred_error,
                    adv_error,
                    orders,
                    result,
                    opt_result,
                    elapsed_ms,
                )
            )

    return rows


def generate_bottleneck_instance(cfg, seed):
    """Synthetic setting where prediction should help reserve capacity."""
    station_count = int(cfg.get("num_stations", 8))
    bus_count = int(cfg.get("num_buses", 6))
    capacity = int(cfg.get("bus_capacity", 20))
    low_orders = int(cfg.get("low_value_orders", 90))
    high_orders = int(cfg.get("high_value_orders", 45))
    high_station_count = max(1, int(cfg.get("high_value_station_count", 2)))
    rng = __import__("numpy").random.default_rng(seed)

    stations = [Station(i, float(i), 0.0) for i in range(station_count)]
    high_stations = list(range(high_station_count))
    low_stations = list(range(high_station_count, station_count))
    if not low_stations:
        low_stations = high_stations

    buses = []
    for bus_id in range(bus_count):
        route = list(range(station_count))
        times = {station: 1.0 + station * 0.1 for station in route}
        buses.append(
            Bus(
                id=bus_id,
                capacity=capacity,
                available_from=0.0,
                depart_at=float(cfg.get("horizon_minutes", 60)),
                route=route,
                station_travel_time=times,
            )
        )

    orders = []
    order_id = 0
    for _ in range(low_orders):
        orders.append(
            Order(
                id=order_id,
                destination=int(rng.choice(low_stations)),
                passengers=1,
                arrival_time=float(rng.uniform(0, 25)),
                priority=float(rng.uniform(0.05, 0.25)),
            )
        )
        order_id += 1
    for _ in range(high_orders):
        orders.append(
            Order(
                id=order_id,
                destination=int(rng.choice(high_stations)),
                passengers=1,
                arrival_time=float(rng.uniform(30, 60)),
                priority=float(rng.uniform(0.8, 1.0)),
            )
        )
        order_id += 1
    orders.sort(key=lambda order: (order.arrival_time, order.id))
    return stations, orders, buses


def plot_sanity(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    plot_df = df.copy()
    plot_df["curve"] = plot_df.apply(
        lambda row: f"RP-LAIPD theta={row['theta']}" if row["method"] == "RP-LAIPD" else row["method"],
        axis=1,
    )
    grouped = plot_df.groupby(["curve", "prediction_error"], as_index=False)["alg_over_opt"].mean()
    for curve, sub in grouped.groupby("curve"):
        sub = sub.sort_values("prediction_error")
        ax.plot(sub["prediction_error"], sub["alg_over_opt"], marker="o", linewidth=1.8, label=curve)
    ax.set_title("Synthetic sanity-check: ALG/OPT vs prediction error")
    ax.set_xlabel("Prediction error |scale - 1|")
    ax.set_ylabel("ALG / OPT")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        [
            "method",
            "theta",
            "prediction_scale",
            "prediction_corruption",
            "corruption_strength",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        runs=("slot_id", "count"),
        prediction_error_mean=("prediction_error", "mean"),
        prediction_error_std=("prediction_error", "std"),
        advice_error_mean=("advice_error", "mean"),
        advice_error_std=("advice_error", "std"),
        alg_over_opt_mean=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"),
        accepted_value_mean=("accepted_value", "mean"),
        accepted_value_std=("accepted_value", "std"),
        served_orders_mean=("served_orders", "mean"),
        high_value_rejected_mean=("high_value_rejected", "mean"),
        slot_runtime_ms_mean=("slot_runtime_ms", "mean"),
    ).reset_index()
    summary["prediction_error_std"] = summary["prediction_error_std"].fillna(0.0)
    summary["advice_error_std"] = summary["advice_error_std"].fillna(0.0)
    summary["alg_over_opt_std"] = summary["alg_over_opt_std"].fillna(0.0)
    summary["accepted_value_std"] = summary["accepted_value_std"].fillna(0.0)
    summary["alg_over_opt_ci95"] = 1.96 * summary["alg_over_opt_std"] / summary["runs"].pow(0.5)
    summary["accepted_value_ci95"] = 1.96 * summary["accepted_value_std"] / summary["runs"].pow(0.5)
    return summary.sort_values(["prediction_error_mean", "method", "theta"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "synthetic_sanity")
    parser.add_argument("--setting", choices=["paper", "bottleneck"], default="paper")
    parser.add_argument(
        "--prediction-scales",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 1.25, 0.5, 1.5],
    )
    parser.add_argument(
        "--corruption",
        choices=["scale", "permutation", "adversarial_concentration"],
        default="scale",
    )
    parser.add_argument(
        "--corruption-strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
    )
    parser.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    args = parser.parse_args()

    cfg = load_experiment(args.experiment)
    if args.setting == "bottleneck":
        cfg.update(
            {
                "num_stations": 8,
                "num_buses": 6,
                "bus_capacity": 20,
                "num_orders": 135,
                "horizon_minutes": 60,
                "bus_wait_minutes": 60,
                "station_fairness_scale": 10.0,
                "bus_time_scale": 10.0,
                "station_time_scale": 10.0,
            }
        )
    rows = []
    for slot in range(args.slots):
        seed = args.seed + args.experiment * 100 + slot
        if args.setting == "bottleneck":
            stations, orders, buses = generate_bottleneck_instance(cfg, seed)
            rows.extend(
                run_instance(
                    stations,
                    orders,
                    buses,
                    cfg,
                    seed,
                    args.prediction_scales,
                    args.corruption_strengths,
                    args.corruption,
                    args.thetas,
                    slot,
                )
            )
        else:
            rows.extend(
                run_one_slot(
                    cfg=cfg,
                    seed=seed,
                    prediction_scales=args.prediction_scales,
                    corruption_strengths=args.corruption_strengths,
                    corruption=args.corruption,
                    thetas=args.thetas,
                    slot_id=slot,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    summary = summarize_results(df)
    csv_path = args.output_dir / "synthetic_sanity_results.csv"
    summary_path = args.output_dir / "synthetic_sanity_summary.csv"
    fig_path = args.output_dir / "alg_over_opt_vs_prediction_error.png"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_sanity(df, fig_path)
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
