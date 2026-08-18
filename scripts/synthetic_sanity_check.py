#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rood_dasfaa2019.learning.metrics import summarize_results
from rood_dasfaa2019.learning.runner import run_learning_augmented_instance, run_learning_augmented_slot
from rood_dasfaa2019.simulation.entities import Bus, Order, Station
from rood_dasfaa2019.utils.config import load_experiment


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
                run_learning_augmented_instance(
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
                run_learning_augmented_slot(
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
