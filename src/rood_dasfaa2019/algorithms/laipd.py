from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from rood_dasfaa2019.learning.prediction import SyntheticPredictionProvider

from .common import *


def _predicted_type_counts(orders, station_count, cfg, predicted_counts=None):
    """Return non-negative type-level demand advice n_hat_k."""
    if predicted_counts is not None:
        return {
            station: max(float(predicted_counts.get(station, 0.0)), 0.0)
            for station in range(station_count)
        }
    advice_cfg = dict(cfg)
    advice_cfg.setdefault("prediction_floor", 1.0)
    return SyntheticPredictionProvider(advice_cfg).predict(orders, station_count)


def build_predictive_lp_advice(
    orders,
    buses,
    station_count,
    cfg,
    predicted_counts=None,
    type_values=None,
):
    """Solve the predictive LP and return type-to-bus quotas y_hat_kj.

    ``predicted_counts`` and ``type_values`` let the real-data experiment use
    forecasts and values learned from Train rather than the realized Test slot.
    """
    predicted = _predicted_type_counts(
        orders, station_count, cfg, predicted_counts=predicted_counts
    )
    avg_priority = {station: 0.0 for station in range(station_count)}
    type_counts = {station: 0 for station in range(station_count)}
    if type_values is None:
        for order in orders:
            avg_priority[order.destination] += order.priority
            type_counts[order.destination] += 1
        global_avg = sum(order.priority for order in orders) / max(len(orders), 1)
        for station in range(station_count):
            if type_counts[station]:
                avg_priority[station] /= type_counts[station]
            else:
                avg_priority[station] = global_avg
    else:
        avg_priority = {
            station: float(type_values.get(station, 0.0))
            for station in range(station_count)
        }

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
        indices = [k for k, (candidate, _) in enumerate(pairs) if candidate == station]
        if indices:
            rows.append([(k, 1.0) for k in indices])
            upper_bounds.append(predicted[station])

    for bus in buses:
        indices = [k for k, (_, candidate) in enumerate(pairs) if candidate.id == bus.id]
        if indices:
            rows.append([(k, 1.0) for k in indices])
            upper_bounds.append(bus.capacity)
            rows.append(
                [(k, pairs[k][1].station_travel_time[pairs[k][0]]) for k in indices]
            )
            upper_bounds.append(bus_time[bus.id])

    for station in range(station_count):
        indices = [k for k, (candidate, _) in enumerate(pairs) if candidate == station]
        if indices:
            rows.append([(k, 1.0) for k in indices])
            upper_bounds.append(station_cap[station])
            rows.append(
                [(k, pairs[k][1].station_travel_time[station]) for k in indices]
            )
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
    values = (
        np.zeros(len(pairs), dtype=float)
        if result.x is None or not result.success
        else result.x
    )
    quota = defaultdict(float)
    for (station, bus), value in zip(pairs, values):
        quota[(station, bus.id)] = float(value)
    return quota


def _build_type_bus_advice(orders, buses, station_count, cfg, advice=None):
    if advice is not None:
        return advice
    return build_predictive_lp_advice(orders, buses, station_count, cfg)


def _quota_remaining(order, bus, used_quota, advice):
    return advice.get((order.destination, bus.id), 0.0) - used_quota[
        (order.destination, bus.id)
    ]


def _scaled_buses(buses, fraction):
    scaled = clone_buses(buses)
    for bus in scaled:
        bus.capacity *= fraction
    return scaled


def _scaled_resource_cfg(orders, buses, station_count, cfg, fraction):
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    scaled = dict(cfg)
    scaled["station_capacity"] = {
        station: cap * fraction for station, cap in station_cap.items()
    }
    scaled["bus_time_capacity"] = {
        bus_id: cap * fraction for bus_id, cap in bus_time.items()
    }
    scaled["station_time_capacity"] = {
        station: cap * fraction for station, cap in station_time.items()
    }
    return scaled


def _scaled_advice(advice, fraction):
    return defaultdict(float, {key: value * fraction for key, value in advice.items()})


def prediction_only_dispatch(orders, buses, station_count, cfg, advice=None):
    """Dispatch by following predicted type-to-bus quota as much as possible."""
    buses = clone_buses(buses)
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    station_passengers = {station: 0 for station in range(station_count)}
    station_used_time = {station: 0.0 for station in range(station_count)}
    accepted = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _build_type_bus_advice(orders, buses, station_count, cfg, advice=advice)
    stats = candidate_stats(orders, buses)
    latencies = []

    for order in orders:
        started = perf_counter()
        candidates = available_buses(order, buses)
        if not candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue
        candidates.sort(
            key=lambda bus: (
                -_quota_remaining(order, bus, used_quota, advice),
                bus.station_travel_time[order.destination],
                -bus.remaining_seats,
            )
        )
        was_accepted = False
        for bus in candidates:
            if _quota_remaining(order, bus, used_quota, advice) < order.passengers:
                continue
            reason = rejection_reason(
                order, bus, station_passengers, station_used_time,
                station_cap, bus_time, station_time,
            )
            if reason == "accepted":
                commit(order, bus, station_passengers, station_used_time)
                used_quota[(order.destination, bus.id)] += order.passengers
                accepted[order.id] = bus.id
                was_accepted = True
                break
        if not was_accepted:
            reasons["quota_or_feasibility"] = reasons.get("quota_or_feasibility", 0) + 1
        latencies.append((perf_counter() - started) * 1000.0)

    return SolverResult(
        summarize(accepted, orders, buses),
        diagnostics=make_diagnostics(stats, reasons),
        request_latencies_ms=latencies,
    )


def rp_laipd_dispatch(orders, buses, station_count, cfg, advice=None):
    """Resource-partitioned learning-augmented IPD.

    ``theta`` is the robust/IPD resource fraction. ``theta=0`` fully follows
    prediction advice; ``theta=1`` ignores prediction and exactly recovers IPD.
    """
    epsilon = float(cfg.get("epsilon", 0.2))
    theta = min(max(float(cfg.get("theta", 0.5)), 0.0), 1.0)
    robust_fraction = theta
    advice_fraction = 1.0 - theta

    full_buses = clone_buses(buses)
    advice_buses = _scaled_buses(buses, advice_fraction)
    robust_buses = _scaled_buses(buses, robust_fraction)
    advice_cfg = _scaled_resource_cfg(orders, full_buses, station_count, cfg, advice_fraction)
    robust_cfg = _scaled_resource_cfg(orders, full_buses, station_count, cfg, robust_fraction)

    station_cap_p, bus_time_p, station_time_p = limits(orders, advice_buses, station_count, advice_cfg)
    station_passengers_p = {station: 0 for station in range(station_count)}
    station_used_time_p = {station: 0.0 for station in range(station_count)}

    station_cap_r, bus_time_r, station_time_r = limits(orders, robust_buses, station_count, robust_cfg)
    station_passengers_r = {station: 0 for station in range(station_count)}
    station_used_time_r = {station: 0.0 for station in range(station_count)}
    z = {bus.id: 0.0 for bus in robust_buses}
    u = {bus.id: 0.0 for bus in robust_buses}
    r = {station: 0.0 for station in range(station_count)}
    q = {station: 0.0 for station in range(station_count)}

    accepted = {}
    reasons = {}
    used_quota = defaultdict(float)
    advice = _scaled_advice(
        _build_type_bus_advice(orders, full_buses, station_count, cfg, advice=advice),
        advice_fraction,
    )
    stats = candidate_stats(orders, full_buses)
    latencies = []

    for order in orders:
        started = perf_counter()
        if not available_buses(order, full_buses):
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue

        advice_candidates = available_buses(order, advice_buses)
        advice_candidates.sort(
            key=lambda bus: (
                -_quota_remaining(order, bus, used_quota, advice),
                bus.station_travel_time[order.destination],
                -bus.remaining_seats,
            )
        )
        was_accepted = False
        for bus in advice_candidates:
            if _quota_remaining(order, bus, used_quota, advice) < order.passengers:
                continue
            reason = rejection_reason(
                order, bus, station_passengers_p, station_used_time_p,
                station_cap_p, bus_time_p, station_time_p,
            )
            if reason == "accepted":
                commit(order, bus, station_passengers_p, station_used_time_p)
                used_quota[(order.destination, bus.id)] += order.passengers
                accepted[order.id] = bus.id
                was_accepted = True
                break

        if not was_accepted:
            robust_candidates = available_buses(order, robust_buses)
            robust_candidates.sort(
                key=lambda bus: order.passengers * z[bus.id]
                + bus.station_travel_time[order.destination] * order.passengers
                * (u[bus.id] + q[order.destination])
            )
            if not robust_candidates:
                reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            else:
                bus = robust_candidates[0]
                travel = bus.station_travel_time[order.destination] * order.passengers
                dual = order.passengers * (z[bus.id] + r[order.destination]) + travel * (
                    u[bus.id] + q[order.destination]
                )
                reason = rejection_reason(
                    order, bus, station_passengers_r, station_used_time_r,
                    station_cap_r, bus_time_r, station_time_r, dual,
                )
                if reason == "accepted":
                    commit(order, bus, station_passengers_r, station_used_time_r)
                    accepted[order.id] = bus.id
                    z[bus.id] = z[bus.id] * (
                        1 + order.passengers / max(bus.capacity, 1e-9)
                    ) + order.priority * epsilon / (4 * max(bus.capacity, 1e-9))
                    r[order.destination] = r[order.destination] * (
                        1 + order.passengers / max(station_cap_r[order.destination], 1e-9)
                    ) + order.priority * epsilon / (
                        4 * max(station_cap_r[order.destination], 1e-9)
                    )
                    u[bus.id] = u[bus.id] * (
                        1 + travel / max(bus_time_r[bus.id], 1e-9)
                    ) + order.priority * epsilon / (4 * max(bus_time_r[bus.id], 1e-9))
                    q[order.destination] = q[order.destination] * (
                        1 + travel / max(station_time_r[order.destination], 1e-9)
                    ) + order.priority * epsilon / (
                        4 * max(station_time_r[order.destination], 1e-9)
                    )
                    was_accepted = True
                else:
                    reasons[reason] = reasons.get(reason, 0) + 1
        latencies.append((perf_counter() - started) * 1000.0)

    return SolverResult(
        summarize(accepted, orders, full_buses),
        diagnostics=make_diagnostics(stats, reasons),
        request_latencies_ms=latencies,
    )
