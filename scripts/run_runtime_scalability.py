#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import ipd_dispatch, rp_laipd_dispatch
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice
from rood_dasfaa2019.simulation.generator import generate_instance


def timed(fn):
    start = perf_counter()
    value = fn()
    return value, (perf_counter() - start) * 1000.0


def benchmark(cfg: dict, seed: int, dimension: str, level: int) -> list[dict]:
    (stations, orders, buses), preparation_ms = timed(lambda: generate_instance(cfg, seed))
    predicted, prediction_ms = timed(lambda: {
        k: float(sum(order.destination == k for order in orders)) for k in range(len(stations))
    })
    advice, lp_ms = timed(lambda: build_predictive_lp_advice(
        orders, buses, len(stations), cfg, predicted_counts=predicted
    ))
    output = []
    for method, fn in [
        ("IPD", lambda: ipd_dispatch(orders, buses, len(stations), cfg)),
        ("RP-LAIPD", lambda: rp_laipd_dispatch(orders, buses, len(stations), cfg, advice=advice)),
    ]:
        result, runtime_ms = timed(fn)
        latencies = np.asarray(result.request_latencies_ms, dtype=float)
        output.append({
            "seed": seed, "dimension": dimension, "level": level, "N": cfg["num_orders"],
            "M": cfg["num_buses"], "K": cfg["num_stations"], "method": method,
            "prediction_time_ms": prediction_ms, "predictive_lp_time_ms": lp_ms,
            "slot_preparation_time_ms": preparation_ms, "slot_runtime_ms": runtime_ms,
            "avg_request_latency_ms": float(latencies.mean()),
            "p95_request_latency_ms": float(np.quantile(latencies, 0.95)),
            "served_orders": len(result.accepted),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 one-factor-at-a-time runtime benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = {
        "num_orders": 1000, "num_buses": 10, "num_stations": 50, "bus_capacity": 100,
        "max_passengers_per_order": 1, "horizon_minutes": 30, "bus_wait_minutes": 30,
        "route_min_stations": 5, "route_max_stations": 8, "route_overlap_factor": 3,
        "prediction_noise": 0.25, "epsilon": 0.2, "theta": 0.6,
    }
    grids = {
        "N": [100, 500, 1000, 5000],
        "M": [5, 10, 20, 50],
        "K": [10, 50, 100, 500],
    }
    if args.quick:
        grids = {"N": [100, 1000], "M": [5, 20], "K": [10, 100]}
        args.seeds = args.seeds[:1]
    rows = []
    for dimension, levels in grids.items():
        key = {"N": "num_orders", "M": "num_buses", "K": "num_stations"}[dimension]
        for level in levels:
            cfg = {**base, key: level}
            cfg["bus_capacity"] = max(30, int(np.ceil(cfg["num_orders"] / cfg["num_buses"])))
            for seed in args.seeds:
                rows.extend(benchmark(cfg, seed, dimension, level))
    raw = pd.DataFrame(rows)
    root = Path(config["data"]["output_dir"])
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    raw.to_csv(root / "raw" / "runtime_scalability.csv", index=False)
    measures = ["prediction_time_ms", "predictive_lp_time_ms", "slot_preparation_time_ms",
                "slot_runtime_ms", "avg_request_latency_ms", "p95_request_latency_ms"]
    summary = raw.groupby(["dimension", "level", "method"])[measures].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, item))) if isinstance(item, tuple) else item for item in summary.columns]
    summary.to_csv(root / "summary" / "runtime_scalability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
