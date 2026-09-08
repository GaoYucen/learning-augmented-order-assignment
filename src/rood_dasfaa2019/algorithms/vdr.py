from __future__ import annotations

from collections import defaultdict
from time import perf_counter
import math
from typing import Callable

from .common import (
    SolverResult,
    available_buses,
    candidate_stats,
    clone_buses,
    commit,
    limits,
    make_diagnostics,
    rejection_reason,
    summarize,
)
from .laipd import build_predictive_lp_advice


ArrivalCDF = Callable[[int, float], float]


def _cdf(arrival_cdf: ArrivalCDF, station: int, time: float) -> float:
    """Return a clipped Train-side cumulative arrival fraction for one type."""
    return min(max(float(arrival_cdf(station, time)), 0.0), 1.0)


def _expected_cumulative(predicted_counts, arrival_cdf: ArrivalCDF, time: float) -> float:
    return sum(
        float(count) * _cdf(arrival_cdf, station, time)
        for station, count in predicted_counts.items()
    )


def _projected_load(
    predicted_counts,
    arrival_cdf: ArrivalCDF,
    seen: int,
    time: float,
    total_capacity: float,
    cdf_floor: float,
) -> float:
    predicted_total = sum(predicted_counts.values())
    expected = _expected_cumulative(predicted_counts, arrival_cdf, time)
    fraction = expected / max(predicted_total, 1e-9)
    projected = 0.0 if fraction < cdf_floor else seen / max(fraction, cdf_floor)
    return max(predicted_total, projected) / max(total_capacity, 1e-9)


def _price_strength(load_ratio: float, cfg: dict) -> float:
    """Frozen congestion-price schedule used by the journal experiments."""
    if load_ratio <= 1.0:
        return 0.0
    slope = float(cfg.get("vdr_price_slope", 1.5))
    offset = float(cfg.get("vdr_price_offset", 0.7))
    lower = float(cfg.get("vdr_price_min", 0.35))
    upper = float(cfg.get("vdr_price_max", 1.8))
    return min(max(slope * (load_ratio - offset), lower), upper)


def _update_prices(
    order,
    bus,
    z,
    u,
    r,
    q,
    station_cap,
    bus_time,
    station_time,
    epsilon: float,
) -> None:
    if epsilon <= 0.0:
        return
    passengers = float(order.passengers)
    travel = passengers * float(bus.station_travel_time[order.destination])
    step = epsilon / 4.0

    bus_cap = max(float(bus.capacity), 1e-9)
    station_capacity = max(float(station_cap[order.destination]), 1e-9)
    bus_time_capacity = max(float(bus_time[bus.id]), 1e-9)
    station_time_capacity = max(float(station_time[order.destination]), 1e-9)

    z[bus.id] = z[bus.id] * (1.0 + passengers / bus_cap) + step * passengers / bus_cap
    r[order.destination] = (
        r[order.destination] * (1.0 + passengers / station_capacity)
        + step * passengers / station_capacity
    )
    u[bus.id] = u[bus.id] * (1.0 + travel / bus_time_capacity) + step * travel / bus_time_capacity
    q[order.destination] = (
        q[order.destination] * (1.0 + travel / station_time_capacity)
        + step * travel / station_time_capacity
    )


def _one_sided_calibrated_counts(
    predicted_counts,
    arrival_cdf: ArrivalCDF,
    seen_by_type,
    time: float,
    z_value: float,
    cdf_floor: float,
):
    """Release overpredicted quota without ever increasing the original forecast.

    A type forecast is reduced only when the observed cumulative count is below
    a one-sided z-sigma lower band of its model-implied cumulative count.  The
    replacement total is the corresponding observed-count upper estimate.
    """
    effective = {}
    for station, predicted in predicted_counts.items():
        predicted = max(float(predicted), 0.0)
        fraction = _cdf(arrival_cdf, station, time)
        if predicted <= 1e-12 or fraction < cdf_floor:
            effective[station] = predicted
            continue

        expected_seen = predicted * fraction
        observed = float(seen_by_type.get(station, 0.0))
        lower = expected_seen - z_value * math.sqrt(max(expected_seen, 1.0))
        if observed >= lower:
            effective[station] = predicted
            continue

        upper_total = (observed + z_value * math.sqrt(max(observed, 1.0))) / fraction
        effective[station] = min(predicted, max(observed, upper_total))
    return effective


def vdr_la_dispatch(
    orders,
    buses,
    station_count: int,
    cfg: dict,
    *,
    predicted_counts: dict[int, float],
    type_values: dict[int, float],
    arrival_cdf: ArrivalCDF,
    advice=None,
    calibrate: bool = True,
    calibration_z: float = 2.0,
    cdf_floor: float = 0.08,
):
    """Strict-feasible value-dominance reservation dispatch.

    This is the frozen journal algorithm after E8-A/E8-B/E8-C.  The caller must
    provide Train-side (or otherwise genuinely ex-ante) type counts, type values,
    and an arrival CDF.  The function deliberately does not infer these objects
    from the realized Test future.

    Core rules:
      1. follow the value-aware predictive LP quota when feasible;
      2. otherwise reserve only for predicted future types whose value exceeds
         the current order value (value-dominance reservation);
      3. optionally release only statistically supported overprediction via a
         one-sided online calibration rule;
      4. when overprediction is detected and calibrated total demand no longer
         exceeds physical capacity, revert to a strict work-conserving rule;
      5. every actual acceptance passes ``rejection_reason`` before ``commit``.

    ``calibrate=False`` recovers the frozen, fixed-forecast VDR ablation.
    """
    if arrival_cdf is None:
        raise ValueError("arrival_cdf must be a Train-side callable (station, time) -> cumulative fraction")

    predicted = {
        station: max(float(predicted_counts.get(station, 0.0)), 0.0)
        for station in range(station_count)
    }
    values = {
        station: float(type_values.get(station, 0.0))
        for station in range(station_count)
    }

    buses = clone_buses(buses)
    station_cap, bus_time, station_time = limits(orders, buses, station_count, cfg)
    total_capacity = float(sum(bus.capacity for bus in buses))
    advice = advice or build_predictive_lp_advice(
        orders,
        buses,
        station_count,
        cfg,
        predicted_counts=predicted,
        type_values=values,
    )

    station_passengers = {station: 0.0 for station in range(station_count)}
    station_used_time = {station: 0.0 for station in range(station_count)}
    z = {bus.id: 0.0 for bus in buses}
    u = {bus.id: 0.0 for bus in buses}
    r = {station: 0.0 for station in range(station_count)}
    q = {station: 0.0 for station in range(station_count)}
    used_quota = defaultdict(float)
    seen_by_type = {station: 0.0 for station in range(station_count)}
    accepted = {}
    reasons = {}
    stats = candidate_stats(orders, buses)
    latencies = []
    seen = 0

    for order in orders:
        started = perf_counter()
        seen += order.passengers
        seen_by_type[order.destination] += order.passengers

        effective_predicted = (
            _one_sided_calibrated_counts(
                predicted,
                arrival_cdf,
                seen_by_type,
                order.arrival_time,
                float(calibration_z),
                float(cdf_floor),
            )
            if calibrate
            else predicted
        )

        expected_seen = _expected_cumulative(predicted, arrival_cdf, order.arrival_time)
        global_lower = expected_seen - calibration_z * math.sqrt(max(expected_seen, 1.0))
        global_overprediction = bool(calibrate and seen < global_lower)
        effective_total = sum(effective_predicted.values())
        work_conserving = bool(
            global_overprediction and effective_total <= total_capacity + 1e-9
        )

        if global_overprediction:
            load_ratio = effective_total / max(total_capacity, 1e-9)
        else:
            load_ratio = _projected_load(
                predicted,
                arrival_cdf,
                seen,
                order.arrival_time,
                total_capacity,
                float(cdf_floor),
            )
        epsilon = _price_strength(load_ratio, cfg)

        candidates = available_buses(order, buses)
        if not candidates:
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue

        chosen = None
        via_advice = False

        def quota_scale(station: int) -> float:
            raw_prediction = predicted.get(station, 0.0)
            if not calibrate or raw_prediction <= 1e-12:
                return 1.0
            return min(
                1.0,
                max(float(effective_predicted.get(station, 0.0)), 0.0)
                / raw_prediction,
            )

        def effective_quota(station: int, bus_id: int) -> float:
            return float(advice.get((station, bus_id), 0.0)) * quota_scale(station)

        def quota_remaining(station: int, bus_id: int) -> float:
            return max(
                effective_quota(station, bus_id) - used_quota[(station, bus_id)],
                0.0,
            )

        if work_conserving:
            # Low effective load + statistically supported overprediction:
            # prediction-based reservation is unnecessary.
            for bus in sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.remaining_seats,
                    candidate.station_travel_time[order.destination],
                ),
            ):
                if rejection_reason(
                    order,
                    bus,
                    station_passengers,
                    station_used_time,
                    station_cap,
                    bus_time,
                    station_time,
                ) == "accepted":
                    chosen = bus
                    break
        else:
            # First try to consume the predictive LP quota for the current type.
            for bus in sorted(
                candidates,
                key=lambda candidate: (
                    -quota_remaining(order.destination, candidate.id),
                    -candidate.remaining_seats,
                ),
            ):
                if quota_remaining(order.destination, bus.id) + 1e-9 < order.passengers:
                    continue
                if rejection_reason(
                    order,
                    bus,
                    station_passengers,
                    station_used_time,
                    station_cap,
                    bus_time,
                    station_time,
                ) == "accepted":
                    chosen = bus
                    via_advice = True
                    break

            if chosen is None:
                lower = expected_seen - calibration_z * math.sqrt(max(expected_seen, 1.0))
                reservation_strength = (
                    1.0
                    if expected_seen < 5.0 or seen >= lower
                    else max(0.25, seen / max(lower, 1e-9))
                )

                if epsilon <= 0.0:
                    robust_candidates = sorted(
                        candidates,
                        key=lambda candidate: (
                            -candidate.remaining_seats,
                            candidate.station_travel_time[order.destination],
                        ),
                    )
                else:
                    robust_candidates = sorted(
                        candidates,
                        key=lambda candidate: (
                            order.passengers * (z[candidate.id] + r[order.destination])
                            + order.passengers
                            * candidate.station_travel_time[order.destination]
                            * (u[candidate.id] + q[order.destination])
                        ),
                    )

                for bus in robust_candidates:
                    if rejection_reason(
                        order,
                        bus,
                        station_passengers,
                        station_used_time,
                        station_cap,
                        bus_time,
                        station_time,
                    ) != "accepted":
                        continue

                    future = []
                    for station in range(station_count):
                        if station == order.destination:
                            continue
                        # Use the raw predictive quota for the arrival-CDF tail.
                        # Calibration is applied exactly once through quota_remaining().
                        future_quota = float(advice.get((station, bus.id), 0.0)) * (
                            1.0 - _cdf(arrival_cdf, station, order.arrival_time)
                        )
                        quantity = min(
                            quota_remaining(station, bus.id),
                            future_quota,
                        )
                        if quantity > 1e-9 and values[station] > order.priority + 1e-9:
                            future.append((station, quantity))

                    protected = reservation_strength * sum(quantity for _, quantity in future)
                    if bus.remaining_seats - order.passengers + 1e-9 < protected:
                        continue

                    if epsilon > 0.0:
                        travel = order.passengers * bus.station_travel_time[order.destination]
                        dual = order.passengers * (z[bus.id] + r[order.destination]) + travel * (
                            u[bus.id] + q[order.destination]
                        )
                        future_best = max(
                            [values[station] for station, quantity in future] + [-1.0]
                        )
                        value_dominates = order.priority >= future_best - 1e-9
                        if dual >= order.priority and not value_dominates:
                            continue

                    chosen = bus
                    break

        if chosen is None:
            reasons["reservation_or_price"] = reasons.get("reservation_or_price", 0) + 1
            latencies.append((perf_counter() - started) * 1000.0)
            continue

        # This check has already been performed on the exact current state above.
        # No prediction/calibration path can bypass it before commit.
        commit(order, chosen, station_passengers, station_used_time)
        accepted[order.id] = chosen.id
        if via_advice:
            used_quota[(order.destination, chosen.id)] += order.passengers
        if not work_conserving:
            _update_prices(
                order,
                chosen,
                z,
                u,
                r,
                q,
                station_cap,
                bus_time,
                station_time,
                epsilon,
            )
        latencies.append((perf_counter() - started) * 1000.0)

    return SolverResult(
        summarize(accepted, orders, buses),
        diagnostics=make_diagnostics(stats, reasons),
        request_latencies_ms=latencies,
    )
