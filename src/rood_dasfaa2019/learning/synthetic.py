from __future__ import annotations

import numpy as np

from rood_dasfaa2019.simulation.entities import Bus, Order, Station


def apply_bottleneck_config(cfg: dict) -> dict:
    updated = dict(cfg)
    updated.update(
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
    return updated


def generate_bottleneck_instance(cfg: dict, seed: int):
    """Synthetic setting where prediction should help reserve capacity."""
    station_count = int(cfg.get("num_stations", 8))
    bus_count = int(cfg.get("num_buses", 6))
    capacity = int(cfg.get("bus_capacity", 20))
    low_orders = int(cfg.get("low_value_orders", 90))
    high_orders = int(cfg.get("high_value_orders", 45))
    high_station_count = max(1, int(cfg.get("high_value_station_count", 2)))
    rng = np.random.default_rng(seed)

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
