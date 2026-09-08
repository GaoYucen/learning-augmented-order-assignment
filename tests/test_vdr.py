import pytest

from rood_dasfaa2019.algorithms import vdr_la_dispatch
from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.simulation.generator import generate_instance


def _cfg():
    return dict(
        num_stations=8,
        horizon_minutes=60,
        bus_wait_minutes=15,
        epsilon=0.2,
        max_passengers_per_order=3,
        station_fairness_scale=1,
        bus_time_scale=1,
        station_time_scale=1,
        num_buses=12,
        bus_capacity=15,
        num_orders=100,
    )


def _advice_objects(orders, station_count):
    counts = {station: 0.0 for station in range(station_count)}
    value_sum = {station: 0.0 for station in range(station_count)}
    value_n = {station: 0 for station in range(station_count)}
    for order in orders:
        counts[order.destination] += order.passengers
        value_sum[order.destination] += order.priority
        value_n[order.destination] += 1
    global_value = sum(order.priority for order in orders) / max(len(orders), 1)
    values = {
        station: (
            value_sum[station] / value_n[station]
            if value_n[station]
            else global_value
        )
        for station in range(station_count)
    }
    return counts, values


def test_vdr_la_is_strictly_feasible_under_overprediction():
    cfg = _cfg()
    stations, orders, buses = generate_instance(cfg, 23)
    counts, values = _advice_objects(orders, len(stations))
    overpredicted = {station: 2.0 * value for station, value in counts.items()}
    horizon = float(cfg["horizon_minutes"])

    result = vdr_la_dispatch(
        orders,
        buses,
        len(stations),
        cfg,
        predicted_counts=overpredicted,
        type_values=values,
        arrival_cdf=lambda station, time: min(max(time / horizon, 0.0), 1.0),
        calibrate=True,
    )

    assert result.objective >= 0
    assert count_constraint_violations(
        result.accepted, orders, buses, len(stations), cfg
    ) == 0


def test_vdr_la_fixed_forecast_ablation_is_feasible():
    cfg = _cfg()
    stations, orders, buses = generate_instance(cfg, 29)
    counts, values = _advice_objects(orders, len(stations))
    horizon = float(cfg["horizon_minutes"])

    result = vdr_la_dispatch(
        orders,
        buses,
        len(stations),
        cfg,
        predicted_counts=counts,
        type_values=values,
        arrival_cdf=lambda station, time: min(max(time / horizon, 0.0), 1.0),
        calibrate=False,
    )

    assert count_constraint_violations(
        result.accepted, orders, buses, len(stations), cfg
    ) == 0


def test_vdr_la_requires_ex_ante_arrival_cdf():
    cfg = _cfg()
    stations, orders, buses = generate_instance(cfg, 31)
    counts, values = _advice_objects(orders, len(stations))

    with pytest.raises(ValueError):
        vdr_la_dispatch(
            orders,
            buses,
            len(stations),
            cfg,
            predicted_counts=counts,
            type_values=values,
            arrival_cdf=None,
        )
