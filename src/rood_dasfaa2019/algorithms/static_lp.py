from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from rood_dasfaa2019.learning.prediction import SyntheticPredictionProvider

from .common import *


def _mean_by_type(orders, station_count, attr: str, default: float = 1.0) -> dict[int, float]:
    totals = {station: 0.0 for station in range(station_count)}
    counts = {station: 0 for station in range(station_count)}
    for order in orders:
        totals[order.destination] += float(getattr(order, attr))
        counts[order.destination] += 1
    overall = sum(totals.values()) / max(sum(counts.values()), 1)
    if not np.isfinite(overall) or overall <= 0:
        overall = default
    return {
        station: (totals[station] / counts[station] if counts[station] else overall)
        for station in range(station_count)
    }


def _predicted_counts(orders, station_count, cfg, predicted_counts=None):
    if predicted_counts is not None:
        return {
            station: max(float(predicted_counts.get(station, 0.0)), 0.0)
            for station in range(station_count)
        }
    advice_cfg = dict(cfg)
    advice_cfg.setdefault("prediction_floor", 1.0)
    return SyntheticPredictionProvider(advice_cfg).predict(orders, station_count)


def _solve_static_prices(
    orders,
    buses,
    station_count,
    cfg,
    predicted_counts=None,
    type_values=None,
    avg_passengers=None,
):
    predicted = _predicted_counts(orders, station_count, cfg, predicted_counts=predicted_counts)
    if type_values is None:
        type_values = _mean_by_type(orders, station_count, "priority", default=0.0)
    if avg_passengers is None:
        avg_passengers = _mean_by_type(orders, station_count, "passengers", default=1.0)

    pairs = [
        (station, bus)
        for station in range(station_count)
        for bus in buses
        if station in bus.route
    ]
    if not pairs:
        zero_types = {station: 0.0 for station in range(station_count)}
        zero_buses = {bus.id: 0.0 for bus in buses}
        return zero_types, zero_buses, zero_buses, zero_types, zero_types

    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)

    rows: list[list[tuple[int, float]]] = []
    rhs: list[float] = []
    row_meta: list[tuple[str, int]] = []

    for station in range(station_count):
        indices = [idx for idx, (candidate, _) in enumerate(pairs) if candidate == station]
        if indices:
            rows.append([(idx, 1.0) for idx in indices])
            rhs.append(predicted[station])
            row_meta.append(("type", station))

    for bus in buses:
        indices = [idx for idx, (_, candidate) in enumerate(pairs) if candidate.id == bus.id]
        if indices:
            rows.append([(idx, avg_passengers[pairs[idx][0]]) for idx in indices])
            rhs.append(bus.capacity)
            row_meta.append(("bus_capacity", bus.id))
            rows.append(
                [
                    (idx, avg_passengers[pairs[idx][0]] * pairs[idx][1].station_travel_time[pairs[idx][0]])
                    for idx in indices
                ]
            )
            rhs.append(bus_time[bus.id])
            row_meta.append(("bus_time", bus.id))

    for station in range(station_count):
        indices = [idx for idx, (candidate, _) in enumerate(pairs) if candidate == station]
        if indices:
            rows.append([(idx, avg_passengers[pairs[idx][0]]) for idx in indices])
            rhs.append(station_cap[station])
            row_meta.append(("station_capacity", station))
            rows.append(
                [
                    (idx, avg_passengers[pairs[idx][0]] * pairs[idx][1].station_travel_time[station])
                    for idx in indices
                ]
            )
            rhs.append(station_time[station])
            row_meta.append(("station_time", station))

    matrix = lil_matrix((len(rows), len(pairs)))
    for row_index, row in enumerate(rows):
        for variable_index, value in row:
            matrix[row_index, variable_index] = value

    objective = -np.asarray([type_values[station] for station, _ in pairs], dtype=float)
    result = linprog(
        c=objective,
        A_ub=matrix.tocsr(),
        b_ub=np.asarray(rhs, dtype=float),
        bounds=[(0.0, None)] * len(pairs),
        method="highs",
    )
    if result.x is None or not result.success or getattr(result, "ineqlin", None) is None:
        zeros_types = {station: 0.0 for station in range(station_count)}
        zeros_buses = {bus.id: 0.0 for bus in buses}
        return zeros_types, zeros_buses, zeros_buses, zeros_types, zeros_types

    duals = np.maximum(-np.asarray(result.ineqlin.marginals, dtype=float), 0.0)
    type_price = {station: 0.0 for station in range(station_count)}
    bus_capacity_price = {bus.id: 0.0 for bus in buses}
    bus_time_price = {bus.id: 0.0 for bus in buses}
    station_capacity_price = {station: 0.0 for station in range(station_count)}
    station_time_price = {station: 0.0 for station in range(station_count)}
    for (kind, index), price in zip(row_meta, duals):
        if kind == "type":
            type_price[index] = float(price)
        elif kind == "bus_capacity":
            bus_capacity_price[index] = float(price)
        elif kind == "bus_time":
            bus_time_price[index] = float(price)
        elif kind == "station_capacity":
            station_capacity_price[index] = float(price)
        elif kind == "station_time":
            station_time_price[index] = float(price)
    return type_price, bus_capacity_price, bus_time_price, station_capacity_price, station_time_price


def static_bid_price_dispatch(
    orders,
    buses,
    station_count,
    cfg,
    predicted_counts=None,
    type_values=None,
    avg_passengers=None,
):
    """Static LP / bid-price baseline.

    We solve a one-shot LP from prediction, extract shadow prices, and use
    them as fixed bid prices during the online arrival sequence.
    """
    buses = clone_buses(buses)
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    station_passengers = {station: 0 for station in range(station_count)}
    station_used_time = {station: 0.0 for station in range(station_count)}
    accepted = {}
    reasons = {}
    stats = candidate_stats(orders, buses)
    latencies = []

    type_price, bus_cap_price, bus_time_price, station_cap_price, station_time_price = _solve_static_prices(
        orders,
        buses,
        station_count,
        cfg,
        predicted_counts=predicted_counts,
        type_values=type_values,
        avg_passengers=avg_passengers,
    )

    for order in orders:
        started = perf_counter()
        candidates = available_buses(order, buses)
        if not candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue

        scored = []
        for bus in candidates:
            travel = bus.station_travel_time[order.destination] * order.passengers
            reduced_cost = (
                order.priority
                - type_price.get(order.destination, 0.0)
                - order.passengers * (
                    bus_cap_price.get(bus.id, 0.0) + station_cap_price.get(order.destination, 0.0)
                )
                - travel * (
                    bus_time_price.get(bus.id, 0.0) + station_time_price.get(order.destination, 0.0)
                )
            )
            scored.append((reduced_cost, bus))
        scored.sort(key=lambda item: (-item[0], -item[1].remaining_seats, item[1].station_travel_time[order.destination]))
        reduced_cost, bus = scored[0]
        if reduced_cost <= 0:
            reasons["bid_price"] = reasons.get("bid_price", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue

        reason = rejection_reason(
            order, bus, station_passengers, station_used_time, station_cap, bus_time, station_time
        )
        if reason == "accepted":
            commit(order, bus, station_passengers, station_used_time)
            accepted[order.id] = bus.id
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
        latencies.append((perf_counter() - started) * 1000.0)

    return SolverResult(
        summarize(accepted, orders, buses),
        diagnostics=make_diagnostics(stats, reasons),
        request_latencies_ms=latencies,
    )
