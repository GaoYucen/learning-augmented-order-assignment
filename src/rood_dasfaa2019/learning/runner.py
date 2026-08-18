from __future__ import annotations

from time import perf_counter

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

from .metrics import advice_error, dispatch_row, opt_type_bus_allocation, synthetic_prediction_error


def timed(fn):
    start = perf_counter()
    result = fn()
    return result, (perf_counter() - start) * 1000


def prediction_scenarios(prediction_scales: list[float], corruption_strengths: list[float], corruption: str) -> list[tuple[float, float]]:
    if corruption == "scale":
        return [(scale, abs(scale - 1.0)) for scale in prediction_scales]
    return [(scale, strength) for scale in prediction_scales for strength in corruption_strengths]


def run_learning_augmented_slot(
    cfg: dict,
    seed: int,
    prediction_scales: list[float],
    corruption_strengths: list[float],
    corruption: str,
    thetas: list[float],
    slot_id: int,
) -> list[dict]:
    stations, orders, buses = generate_instance(cfg, seed)
    return run_learning_augmented_instance(
        stations,
        orders,
        buses,
        cfg,
        seed,
        prediction_scales,
        corruption_strengths,
        corruption,
        thetas,
        slot_id,
    )


def run_learning_augmented_instance(
    stations: list[Station],
    orders: list[Order],
    buses: list[Bus],
    cfg: dict,
    seed: int,
    prediction_scales: list[float],
    corruption_strengths: list[float],
    corruption: str,
    thetas: list[float],
    slot_id: int,
) -> list[dict]:
    station_count = len(stations)
    opt_result, _ = timed(lambda: offline_opt(orders, buses, station_count, cfg))
    opt_allocation = opt_type_bus_allocation(orders, opt_result)
    rows = []

    random_result, random_ms = timed(lambda: random_dispatch(orders, buses, station_count, cfg, seed))
    greedy_result, greedy_ms = timed(lambda: greedy_dispatch(orders, buses, station_count, cfg))
    ipd_result, ipd_ms = timed(lambda: ipd_dispatch(orders, buses, station_count, cfg))

    for scale, strength in prediction_scenarios(prediction_scales, corruption_strengths, corruption):
        scaled_cfg = dict(cfg)
        scaled_cfg["prediction_scale"] = scale
        scaled_cfg["prediction_corruption"] = corruption
        scaled_cfg["corruption_strength"] = strength
        advice = build_predictive_lp_advice(orders, buses, station_count, scaled_cfg)
        pred_error = synthetic_prediction_error(orders, station_count, scaled_cfg)
        adv_error = advice_error(advice, opt_allocation)
        rows.append(dispatch_row(slot_id, "Random", "", scale, corruption, strength, pred_error, adv_error, orders, random_result, opt_result, random_ms))
        rows.append(dispatch_row(slot_id, "Greedy", "", scale, corruption, strength, pred_error, adv_error, orders, greedy_result, opt_result, greedy_ms))
        rows.append(dispatch_row(slot_id, "IPD", "", scale, corruption, strength, pred_error, adv_error, orders, ipd_result, opt_result, ipd_ms))

        result, elapsed_ms = timed(lambda: prediction_only_dispatch(orders, buses, station_count, scaled_cfg))
        rows.append(dispatch_row(slot_id, "Prediction-only", "", scale, corruption, strength, pred_error, adv_error, orders, result, opt_result, elapsed_ms))

        for theta in thetas:
            theta_cfg = dict(scaled_cfg)
            theta_cfg["theta"] = theta
            result, elapsed_ms = timed(lambda: rp_laipd_dispatch(orders, buses, station_count, theta_cfg))
            rows.append(dispatch_row(slot_id, "RP-LAIPD", theta, scale, corruption, strength, pred_error, adv_error, orders, result, opt_result, elapsed_ms))

    return rows

