from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from rood_dasfaa2019.learning.prediction import SyntheticPredictionProvider

from .common import *


def _predicted_type_counts(orders, station_count, cfg, predicted_counts=None):
    """Return type-level demand advice n_hat_k."""
    if predicted_counts is not None:
        return {station: float(predicted_counts.get(station, 0.0)) for station in range(station_count)}
    advice_cfg = dict(cfg)
    advice_cfg.setdefault("prediction_floor", 1.0)
    return SyntheticPredictionProvider(advice_cfg).predict(orders, station_count)


def build_predictive_lp_advice(orders, buses, station_count, cfg, predicted_counts=None):
    """Solve the predictive LP and return type-to-bus quotas y_hat_kj."""
    predicted = _predicted_type_counts(orders, station_count, cfg, predicted_counts=predicted_counts)
    avg_priority = {v: 0.0 for v in range(station_count)}
    type_counts = {v: 0 for v in range(station_count)}
    for order in orders:
        avg_priority[order.destination] += order.priority
        type_counts[order.destination] += 1
    global_avg = sum(order.priority for order in orders) / max(len(orders), 1)
    for station in range(station_count):
        if type_counts[station]:
            avg_priority[station] /= type_counts[station]
        else:
            avg_priority[station] = global_avg

    pairs = [
        (station, bus)
        for station in range(station_count)
        for bus in buses
        if station in bus.route
    ]
    if not pairs:
        return defaultdict(float)

    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    rows = []
    upper_bounds = []

    for station in range(station_count):
        idx = [k for k, (s, _) in enumerate(pairs) if s == station]
        if idx:
            rows.append([(k, 1.0) for k in idx])
            upper_bounds.append(predicted[station])

    for bus in buses:
        idx = [k for k, (_, b) in enumerate(pairs) if b.id == bus.id]
        if idx:
            rows.append([(k, 1.0) for k in idx])
            upper_bounds.append(bus.capacity)
            rows.append([(k, pairs[k][1].station_travel_time[pairs[k][0]]) for k in idx])
            upper_bounds.append(bus_time[bus.id])

    for station in range(station_count):
        idx = [k for k, (s, _) in enumerate(pairs) if s == station]
        if idx:
            rows.append([(k, 1.0) for k in idx])
            upper_bounds.append(station_cap[station])
            rows.append([(k, pairs[k][1].station_travel_time[station]) for k in idx])
            upper_bounds.append(station_time[station])

    matrix = lil_matrix((len(rows), len(pairs)))
    for row_index, row in enumerate(rows):
        for variable_index, value in row:
            matrix[row_index, variable_index] = value

    objective = -np.asarray(
        [
            avg_priority[station] / max(bus.station_travel_time[station], 1e-6)
            for station, bus in pairs
        ],
        dtype=float,
    )
    result = linprog(
        c=objective,
        A_ub=matrix.tocsr(),
        b_ub=np.asarray(upper_bounds, dtype=float),
        bounds=[(0.0, None)] * len(pairs),
        method="highs",
    )
    values = np.zeros(len(pairs), dtype=float) if result.x is None or not result.success else result.x
    quota = defaultdict(float)
    for (station, bus), value in zip(pairs, values):
        quota[(station, bus.id)] = float(value)
    return quota


def _build_type_bus_advice(orders, buses, station_count, cfg, advice=None):
    if advice is not None:
        return advice
    return build_predictive_lp_advice(orders, buses, station_count, cfg)


def _quota_remaining(order, bus, used_quota, advice):
    return advice.get((order.destination, bus.id), 0.0) - used_quota[(order.destination, bus.id)]


def _scaled_buses(buses, fraction):
    scaled = clone_buses(buses)
    for bus in scaled:
        bus.capacity *= fraction
    return scaled


def _scaled_resource_cfg(orders, buses, station_count, cfg, fraction):
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    scaled = dict(cfg)
    scaled["station_capacity"] = {station: cap * fraction for station, cap in station_cap.items()}
    scaled["bus_time_capacity"] = {bus_id: cap * fraction for bus_id, cap in bus_time.items()}
    scaled["station_time_capacity"] = {station: cap * fraction for station, cap in station_time.items()}
    return scaled


def _scaled_advice(advice, fraction):
    return defaultdict(float, {key: value * fraction for key, value in advice.items()})


def prediction_only_dispatch(orders, buses, station_count, cfg, advice=None):
    """Dispatch by following predicted type-to-bus quota as much as possible."""
    buses = clone_buses(buses)
    sc, bt, st = limits(orders, buses, station_count, cfg)
    sp = {v: 0 for v in range(station_count)}
    sx = {v: 0.0 for v in range(station_count)}
    acc = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _build_type_bus_advice(orders, buses, station_count, cfg, advice=advice)
    cand_stats = candidate_stats(orders, buses)

    for order in orders:
        candidates = available_buses(order, buses)
        if not candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            continue
        candidates.sort(
            key=lambda bus: (
                -_quota_remaining(order, bus, used_quota, advice),
                bus.station_travel_time[order.destination],
                -bus.remaining_seats,
            )
        )
        accepted = False
        for bus in candidates:
            if _quota_remaining(order, bus, used_quota, advice) < order.passengers:
                continue
            reason = rejection_reason(order, bus, sp, sx, sc, bt, st)
            if reason == "accepted":
                commit(order, bus, sp, sx)
                used_quota[(order.destination, bus.id)] += order.passengers
                acc[order.id] = bus.id
                accepted = True
                break
        if not accepted:
            reasons["quota_or_feasibility"] = reasons.get("quota_or_feasibility", 0) + 1

    return SolverResult(summarize(acc, orders, buses), diagnostics=make_diagnostics(cand_stats, reasons))


def rp_laipd_dispatch(orders, buses, station_count, cfg, advice=None):
    """Resource-partitioned learning-augmented IPD.

    theta follows the robustness-oriented convention used in the
    learning-augmented online algorithms literature: theta is the resource
    fraction reserved for the robust IPD branch. Thus theta=1 ignores advice
    and recovers IPD, while theta=0 follows the predictive advice branch.
    """
    eps = float(cfg.get("epsilon", 0.2))
    theta = float(cfg.get("theta", 0.5))
    theta = min(max(theta, 0.0), 1.0)
    robust_fraction = theta
    advice_fraction = 1.0 - theta

    full_buses = clone_buses(buses)
    advice_buses = _scaled_buses(buses, advice_fraction)
    robust_buses = _scaled_buses(buses, robust_fraction)
    advice_cfg = _scaled_resource_cfg(orders, full_buses, station_count, cfg, advice_fraction)
    robust_cfg = _scaled_resource_cfg(orders, full_buses, station_count, cfg, robust_fraction)

    sc_p, bt_p, st_p = limits(orders, advice_buses, station_count, advice_cfg)
    sp_p = {v: 0 for v in range(station_count)}
    sx_p = {v: 0.0 for v in range(station_count)}

    sc_r, bt_r, st_r = limits(orders, robust_buses, station_count, robust_cfg)
    sp_r = {v: 0 for v in range(station_count)}
    sx_r = {v: 0.0 for v in range(station_count)}
    z = {bus.id: 0.0 for bus in robust_buses}
    u = {bus.id: 0.0 for bus in robust_buses}
    r = {v: 0.0 for v in range(station_count)}
    q = {v: 0.0 for v in range(station_count)}

    acc = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _scaled_advice(_build_type_bus_advice(orders, full_buses, station_count, cfg, advice=advice), advice_fraction)
    cand_stats = candidate_stats(orders, full_buses)

    for order in orders:
        full_candidates = available_buses(order, full_buses)
        if not full_candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            continue

        advice_candidates = available_buses(order, advice_buses)
        advice_candidates.sort(
            key=lambda bus: (
                -_quota_remaining(order, bus, used_quota, advice),
                bus.station_travel_time[order.destination],
                -bus.remaining_seats,
            )
        )
        accepted = False
        for bus in advice_candidates:
            if _quota_remaining(order, bus, used_quota, advice) < order.passengers:
                continue
            reason = rejection_reason(order, bus, sp_p, sx_p, sc_p, bt_p, st_p)
            if reason == "accepted":
                commit(order, bus, sp_p, sx_p)
                used_quota[(order.destination, bus.id)] += order.passengers
                acc[order.id] = bus.id
                accepted = True
                break
        if accepted:
            continue

        robust_candidates = available_buses(order, robust_buses)
        if not robust_candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            continue
        robust_candidates.sort(
            key=lambda bus: order.passengers * z[bus.id]
            + bus.station_travel_time[order.destination] * order.passengers * (u[bus.id] + q[order.destination])
        )
        bus = robust_candidates[0]
        travel = bus.station_travel_time[order.destination] * order.passengers
        dual = order.passengers * (z[bus.id] + r[order.destination]) + travel * (u[bus.id] + q[order.destination])
        reason = rejection_reason(order, bus, sp_r, sx_r, sc_r, bt_r, st_r, dual)
        if reason == "accepted":
            commit(order, bus, sp_r, sx_r)
            acc[order.id] = bus.id
            z[bus.id] = z[bus.id] * (1 + order.passengers / max(bus.capacity, 1e-9)) + order.priority * eps / (
                4 * max(bus.capacity, 1e-9)
            )
            r[order.destination] = r[order.destination] * (
                1 + order.passengers / max(sc_r[order.destination], 1e-9)
            ) + order.priority * eps / (4 * max(sc_r[order.destination], 1e-9))
            u[bus.id] = u[bus.id] * (1 + travel / max(bt_r[bus.id], 1e-9)) + order.priority * eps / (
                4 * max(bt_r[bus.id], 1e-9)
            )
            q[order.destination] = q[order.destination] * (
                1 + travel / max(st_r[order.destination], 1e-9)
            ) + order.priority * eps / (4 * max(st_r[order.destination], 1e-9))
            accepted = True
        if not accepted:
            reasons[reason] = reasons.get(reason, 0) + 1

    return SolverResult(summarize(acc, orders, full_buses), diagnostics=make_diagnostics(cand_stats, reasons))
