from __future__ import annotations

import pandas as pd
import numpy as np

from rood_dasfaa2019.algorithms.common import SolverResult
from rood_dasfaa2019.simulation.entities import Order

from .prediction import SyntheticPredictionProvider
from .request_types import request_type, type_counts


def prediction_error(
    orders: list[Order],
    type_count: int,
    predicted_counts: dict[int, float],
) -> float:
    truth = type_counts(orders, type_count)
    absolute_error = sum(abs(predicted_counts.get(request_type_id, 0.0) - truth[request_type_id]) for request_type_id in truth)
    total = max(sum(truth.values()), 1.0)
    return absolute_error / total


def synthetic_prediction_error(orders: list[Order], type_count: int, cfg: dict) -> float:
    predicted = SyntheticPredictionProvider(cfg).predict(orders, type_count)
    return prediction_error(orders, type_count, predicted)


def opt_type_bus_allocation(orders: list[Order], opt_result: SolverResult) -> dict[tuple[int, int], float]:
    order_by_id = {order.id: order for order in orders}
    allocation: dict[tuple[int, int], float] = {}
    for order_id, bus_id in opt_result.accepted.items():
        order = order_by_id[order_id]
        key = (request_type(order), bus_id)
        allocation[key] = allocation.get(key, 0.0) + order.passengers
    return allocation


def advice_error(advice: dict[tuple[int, int], float], opt_allocation: dict[tuple[int, int], float]) -> float:
    keys = set(advice) | set(opt_allocation)
    absolute_error = sum(abs(advice.get(key, 0.0) - opt_allocation.get(key, 0.0)) for key in keys)
    total = max(sum(opt_allocation.values()), 1.0)
    return absolute_error / total


def high_value_rejected_count(orders: list[Order], result: SolverResult) -> int:
    if not orders:
        return 0
    threshold = pd.Series([order.priority for order in orders]).quantile(0.75)
    accepted = set(result.accepted)
    return sum(1 for order in orders if order.priority >= threshold and order.id not in accepted)


def dispatch_row(
    slot_id: int,
    method: str,
    theta,
    prediction_scale: float,
    corruption: str,
    corruption_strength: float,
    pred_error: float,
    adv_error: float,
    orders: list[Order],
    result: SolverResult,
    opt_result: SolverResult,
    elapsed_ms: float,
    total_capacity: float | None = None,
    violations: int = 0,
    seed: int | None = None,
    split: str | None = None,
    prediction_model: str | None = None,
    prediction_time_ms: float = 0.0,
    predictive_lp_time_ms: float = 0.0,
    slot_preparation_ms: float = 0.0,
) -> dict:
    dispatch = result.dispatch
    opt = opt_result.dispatch
    latencies = result.request_latencies_ms or []
    return {
        "seed": seed,
        "split": split,
        "slot_id": slot_id,
        "method": method,
        "theta": theta,
        "prediction_model": prediction_model,
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
        "occupancy": dispatch.passengers / max(total_capacity, 1.0) if total_capacity is not None else np.nan,
        "high_value_rejected": high_value_rejected_count(orders, result),
        "violations": violations,
        "prediction_time_ms": prediction_time_ms,
        "predictive_lp_time_ms": predictive_lp_time_ms,
        "slot_preparation_ms": slot_preparation_ms,
        "slot_runtime_ms": elapsed_ms,
        "avg_request_latency_ms": float(np.mean(latencies)) if latencies else elapsed_ms / max(len(orders), 1),
        "p95_request_latency_ms": float(np.quantile(latencies, 0.95)) if latencies else np.nan,
        "opt_status": opt_result.solver_status,
        "opt_success": opt_result.solver_success,
        "opt_time_limit_hit": opt_result.solver_time_limit_hit,
        "opt_dual_bound": opt_result.solver_dual_bound,
        "opt_gap": opt_result.solver_mip_gap,
    }


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

