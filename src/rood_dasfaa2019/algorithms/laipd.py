from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import LinearConstraint, linprog
from scipy.sparse import lil_matrix

from .common import *


def _predicted_type_counts(orders, station_count, cfg):
    """Return type-level demand advice n_hat_k.

    The first implementation keeps this self-contained for synthetic sanity
    checks: it starts from realized type counts and corrupts them with a simple
    scaling factor. Later this can be replaced by an external predictor.
    """
    scale = float(cfg.get("prediction_scale", 1.0))
    corruption = str(cfg.get("prediction_corruption", "scale"))
    corruption_strength = float(cfg.get("corruption_strength", abs(scale - 1.0)))
    floor = float(cfg.get("prediction_floor", 1.0))
    counts = {v: 0.0 for v in range(station_count)}
    for order in orders:
        counts[order.destination] += order.passengers

    predicted = {v: counts[v] * scale for v in range(station_count)}
    if corruption == "permutation" and corruption_strength > 0:
        predicted = _permute_type_counts(predicted, corruption_strength)
    elif corruption == "adversarial_concentration" and corruption_strength > 0:
        predicted = _concentrate_type_counts(predicted, corruption_strength)
    return {v: max(floor, predicted[v]) for v in range(station_count)}


def _permute_type_counts(counts, strength):
    stations = sorted(counts)
    if len(stations) < 2:
        return counts
    swap_count = min(len(stations), max(0, int(round(len(stations) * strength))))
    if swap_count < 2:
        return counts
    selected = stations[:swap_count]
    rotated_values = [counts[selected[-1]], *[counts[s] for s in selected[:-1]]]
    corrupted = dict(counts)
    for station, value in zip(selected, rotated_values):
        corrupted[station] = value
    return corrupted


def _concentrate_type_counts(counts, strength):
    stations = sorted(counts)
    if len(stations) < 2:
        return counts
    corrupted = dict(counts)
    sorted_by_demand = sorted(stations, key=lambda station: counts[station])
    source_count = max(1, int(round(len(stations) * min(max(strength, 0.0), 1.0) / 2)))
    sources = sorted_by_demand[-source_count:]
    target = sorted_by_demand[0]
    moved = sum(corrupted[source] for source in sources)
    for source in sources:
        corrupted[source] = 0.0
    corrupted[target] += moved
    return corrupted


def build_predictive_lp_advice(orders, buses, station_count, cfg):
    """Solve the predictive LP and return type-to-bus quotas y_hat_kj."""
    predicted = _predicted_type_counts(orders, station_count, cfg)
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


def _build_type_bus_advice(orders, buses, station_count, cfg):
    return build_predictive_lp_advice(orders, buses, station_count, cfg)


def _quota_remaining(order, bus, used_quota, advice):
    return advice.get((order.destination, bus.id), 0.0) - used_quota[(order.destination, bus.id)]


def prediction_only_dispatch(orders, buses, station_count, cfg):
    """Dispatch by following predicted type-to-bus quota as much as possible."""
    buses = clone_buses(buses)
    sc, bt, st = limits(orders, buses, station_count, cfg)
    sp = {v: 0 for v in range(station_count)}
    sx = {v: 0.0 for v in range(station_count)}
    acc = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _build_type_bus_advice(orders, buses, station_count, cfg)
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


def rp_laipd_dispatch(orders, buses, station_count, cfg):
    """Learning-augmented IPD with a quota-aware prediction branch.

    theta controls how much the method trusts prediction advice:
    - theta=0 behaves close to IPD candidate scoring.
    - theta=1 strongly prioritizes remaining predictive quota.
    """
    eps = float(cfg.get("epsilon", 0.2))
    theta = float(cfg.get("theta", 0.5))
    theta = min(max(theta, 0.0), 1.0)

    buses = clone_buses(buses)
    sc, bt, st = limits(orders, buses, station_count, cfg)
    sp = {v: 0 for v in range(station_count)}
    sx = {v: 0.0 for v in range(station_count)}
    z = {bus.id: 0.0 for bus in buses}
    u = {bus.id: 0.0 for bus in buses}
    r = {v: 0.0 for v in range(station_count)}
    q = {v: 0.0 for v in range(station_count)}
    acc = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _build_type_bus_advice(orders, buses, station_count, cfg)
    cand_stats = candidate_stats(orders, buses)

    for order in orders:
        candidates = available_buses(order, buses)
        if not candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            continue

        def score(bus):
            travel = bus.station_travel_time[order.destination] * order.passengers
            ipd_score = order.passengers * z[bus.id] + travel * (u[bus.id] + q[order.destination])
            remaining = max(_quota_remaining(order, bus, used_quota, advice), 0.0)
            prediction_bonus = remaining / max(order.passengers, 1)
            return (1.0 - theta) * ipd_score - theta * prediction_bonus

        candidates.sort(key=score)
        accepted = False
        first_reason = None
        for bus in candidates:
            remaining_quota = _quota_remaining(order, bus, used_quota, advice)
            if theta >= 0.5 and remaining_quota < order.passengers and order.priority < cfg.get("prediction_release_priority", 0.5):
                first_reason = first_reason or "prediction_quota"
                continue
            travel = bus.station_travel_time[order.destination] * order.passengers
            dual = order.passengers * (z[bus.id] + r[order.destination]) + travel * (
                u[bus.id] + q[order.destination]
            )
            relaxed_dual = (1.0 - theta) * dual
            reason = rejection_reason(order, bus, sp, sx, sc, bt, st, relaxed_dual)
            if first_reason is None:
                first_reason = reason
            if reason == "accepted":
                commit(order, bus, sp, sx)
                used_quota[(order.destination, bus.id)] += order.passengers
                acc[order.id] = bus.id
                z[bus.id] = z[bus.id] * (1 + order.passengers / max(bus.capacity, 1)) + order.priority * eps / (
                    4 * max(bus.capacity, 1)
                )
                r[order.destination] = r[order.destination] * (
                    1 + order.passengers / max(sc[order.destination], 1e-9)
                ) + order.priority * eps / (4 * max(sc[order.destination], 1e-9))
                u[bus.id] = u[bus.id] * (1 + travel / max(bt[bus.id], 1e-9)) + order.priority * eps / (
                    4 * max(bt[bus.id], 1e-9)
                )
                q[order.destination] = q[order.destination] * (
                    1 + travel / max(st[order.destination], 1e-9)
                ) + order.priority * eps / (4 * max(st[order.destination], 1e-9))
                accepted = True
                break
        if not accepted:
            key = first_reason or "not_accepted"
            reasons[key] = reasons.get(key, 0) + 1

    return SolverResult(summarize(acc, orders, buses), diagnostics=make_diagnostics(cand_stats, reasons))
