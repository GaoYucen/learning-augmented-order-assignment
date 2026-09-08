from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from rood_dasfaa2019.algorithms import (
    greedy_dispatch,
    ipd_dispatch,
    offline_opt,
    prediction_only_dispatch,
    random_dispatch,
    rp_laipd_dispatch,
)
from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice
from rood_dasfaa2019.learning.metrics import (
    advice_error,
    dispatch_row,
    opt_type_bus_allocation,
    prediction_error,
)
from rood_dasfaa2019.simulation.entities import Bus, Order, Station
from rood_dasfaa2019.routing.cvrp import build_bus_routes


def _timed(fn):
    start = perf_counter()
    result = fn()
    return result, (perf_counter() - start) * 1000.0


@dataclass
class ChengduExperimentContext:
    config: dict
    orders: pd.DataFrame
    demand: pd.DataFrame
    predictions: pd.DataFrame
    stations: list[Station]
    buses: list[Bus]
    algorithm_cfg: dict
    type_values: dict[int, float]
    type_passengers: dict[int, float]
    high_value_threshold: float

    @classmethod
    def load(cls, config: dict) -> "ChengduExperimentContext":
        data_cfg = config["data"]
        exp_cfg = config["experiment"]
        output_dir = Path(data_cfg["output_dir"])
        orders = pd.read_csv(output_dir / "processed" / "experiment_orders.csv.gz")
        demand = pd.read_csv(output_dir / "processed" / "demand_by_slot_type.csv")
        predictions = pd.read_csv(output_dir / "prediction" / "predictions.csv")
        mapping = pd.read_csv(output_dir / "summary" / "station_mapping.csv").sort_values("station_id")

        center_lon = float(mapping["center_lon"].mean())
        center_lat = float(mapping["center_lat"].mean())
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.deg2rad(center_lat))
        stations = [
            Station(
                id=int(row.station_id),
                x=float((row.center_lon - center_lon) * lon_scale),
                y=float((row.center_lat - center_lat) * lat_scale),
            )
            for row in mapping.itertuples()
        ]
        train_demand = (
            demand.loc[demand["split"] == "train"].groupby("type_id")["observed_count"].sum()
            .reindex(range(len(stations)), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        route_cfg = dict(exp_cfg)
        routes = build_bus_routes(
            stations,
            train_demand,
            int(exp_cfg["num_buses"]),
            int(exp_cfg["bus_capacity"]),
            cfg=route_cfg,
            seed=int(data_cfg["random_seed"]),
        )
        covered = {station for route in routes for station in route}
        missing = [station.id for station in stations if station.id not in covered]
        for offset, station_id in enumerate(missing):
            route = routes[offset % len(routes)]
            if len(route) >= int(exp_cfg["route_max_stations"]):
                route[-1] = station_id
            else:
                route.append(station_id)

        buses = []
        for bus_id, route in enumerate(routes):
            elapsed = 0.0
            times = {}
            previous = (0.0, 0.0)
            for station_id in route:
                station = stations[station_id]
                elapsed += float(np.hypot(station.x - previous[0], station.y - previous[1])) * 3.0
                times[station_id] = max(elapsed, 0.1)
                previous = (station.x, station.y)
            buses.append(
                Bus(
                    id=bus_id,
                    capacity=int(exp_cfg["bus_capacity"]),
                    available_from=0.0,
                    depart_at=float(exp_cfg["horizon_minutes"]),
                    route=list(dict.fromkeys(route)),
                    station_travel_time=times,
                )
            )

        total_capacity = sum(bus.capacity for bus in buses)
        shares = train_demand / max(train_demand.sum(), 1.0)
        station_capacity = {station: max(1.0, float(total_capacity * shares[station])) for station in range(len(stations))}
        average_time = np.mean([time for bus in buses for time in bus.station_travel_time.values()])
        bus_time_capacity = {
            bus.id: bus.capacity * np.mean(list(bus.station_travel_time.values())) * float(exp_cfg["bus_time_scale"])
            for bus in buses
        }
        station_time_capacity = {}
        for station in range(len(stations)):
            times = [bus.station_travel_time[station] for bus in buses if station in bus.station_travel_time]
            station_time_capacity[station] = station_capacity[station] * (np.mean(times) if times else average_time) * float(exp_cfg["station_time_scale"])
        algorithm_cfg = {
            **exp_cfg,
            "station_capacity": station_capacity,
            "bus_time_capacity": bus_time_capacity,
            "station_time_capacity": station_time_capacity,
        }
        train_orders = orders.loc[orders["split"] == "train"]
        type_values = (
            train_orders.groupby("station_id")["priority"].mean().reindex(range(len(stations)), fill_value=train_orders["priority"].mean()).to_dict()
        )
        type_passengers = (
            train_orders.groupby("station_id")["party_size"].mean().reindex(range(len(stations)), fill_value=train_orders["party_size"].mean()).to_dict()
        )
        high_value_threshold = float(train_orders["priority"].quantile(float(exp_cfg["high_value_quantile"])))
        return cls(
            config,
            orders,
            demand,
            predictions,
            stations,
            buses,
            algorithm_cfg,
            type_values,
            type_passengers,
            high_value_threshold,
        )

    def slot_orders(self, slot_id: str) -> list[Order]:
        selected = self.orders.loc[self.orders["slot_id"] == slot_id].sort_values(["arrival_minute", "request_id"])
        return [
            Order(
                id=index,
                destination=int(row.station_id),
                passengers=int(row.party_size),
                arrival_time=float(row.arrival_minute),
                priority=float(row.priority),
            )
            for index, row in enumerate(selected.itertuples())
        ]

    def prediction_for_slot(self, slot_id: str, model: str) -> dict[int, float]:
        selected = self.predictions.loc[
            (self.predictions["slot_id"] == slot_id) & (self.predictions["model"] == model)
        ]
        if selected.empty:
            raise KeyError(f"No prediction for slot={slot_id}, model={model}")
        return {int(row.type_id): float(row.predicted_count) for row in selected.itertuples()}

    def observed_counts(self, slot_id: str) -> dict[int, float]:
        selected = self.demand.loc[self.demand["slot_id"] == slot_id]
        return {int(row.type_id): float(row.observed_count) for row in selected.itertuples()}

    def corrupt_prediction(
        self,
        slot_id: str,
        model: str,
        corruption: str,
        level: float,
        seed: int,
    ) -> dict[int, float]:
        prediction = self.prediction_for_slot(slot_id, model)
        if corruption in {"none", "scale"}:
            scale = 1.0 if corruption == "none" else float(level)
            return {key: max(value * scale, 0.0) for key, value in prediction.items()}
        if corruption == "permutation":
            rng = np.random.default_rng(seed)
            count = max(2, int(round(len(prediction) * float(level))))
            selected = sorted(rng.choice(list(prediction), size=min(count, len(prediction)), replace=False).tolist())
            values = [prediction[key] for key in selected]
            rng.shuffle(values)
            result = dict(prediction)
            result.update(dict(zip(selected, values)))
            return result
        if corruption == "temporal_shift":
            shift_steps = int(round(level))
            slot_row = self.demand.loc[self.demand["slot_id"] == slot_id].iloc[0]
            ordered_slots = (
                self.demand[["slot_id", "service_date", "slot_in_day"]].drop_duplicates()
                .sort_values(["service_date", "slot_in_day"]).reset_index(drop=True)
            )
            position = int(ordered_slots.index[ordered_slots["slot_id"] == slot_id][0])
            source_position = min(max(position - shift_steps, 0), len(ordered_slots) - 1)
            source_slot = str(ordered_slots.iloc[source_position]["slot_id"])
            return self.prediction_for_slot(source_slot, model)
        if corruption == "scarce_resource":
            coverage = {station: sum(station in bus.route for bus in self.buses) for station in prediction}
            scarcity = {
                station: self.type_values[station] / max(coverage[station] * self.algorithm_cfg["station_capacity"][station], 1e-9)
                for station in prediction
            }
            ordered = sorted(prediction, key=scarcity.get, reverse=True)
            attacked = ordered[: max(1, int(round(len(ordered) * float(level))))]
            receivers = ordered[-len(attacked):]
            result = dict(prediction)
            moved = 0.0
            for station in attacked:
                removed = result[station] * min(max(float(level), 0.0), 1.0)
                result[station] -= removed
                moved += removed
            if receivers:
                for station in receivers:
                    result[station] += moved / len(receivers)
            return result
        raise ValueError(f"Unknown corruption: {corruption}")


def run_chengdu_slot(
    context: ChengduExperimentContext,
    slot_id: str,
    model: str,
    corruption: str = "none",
    level: float = 0.0,
    thetas: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    seed: int = 2026,
    include_baselines: bool = False,
    opt_result=None,
) -> list[dict]:
    prep_start = perf_counter()
    orders = context.slot_orders(slot_id)
    split = str(context.orders.loc[context.orders["slot_id"] == slot_id, "split"].iloc[0])
    prep_ms = (perf_counter() - prep_start) * 1000.0
    predicted, prediction_ms = _timed(lambda: context.corrupt_prediction(slot_id, model, corruption, level, seed))
    advice, lp_ms = _timed(
        lambda: build_predictive_lp_advice(
            orders,
            context.buses,
            len(context.stations),
            context.algorithm_cfg,
            predicted_counts=predicted,
            type_values=context.type_values,
        )
    )
    if opt_result is None:
        opt_result, _ = _timed(lambda: offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg))
    opt_allocation = opt_type_bus_allocation(orders, opt_result)
    observed = context.observed_counts(slot_id)
    pred_error = sum(abs(predicted.get(key, 0.0) - observed.get(key, 0.0)) for key in observed) / max(sum(observed.values()), 1.0)
    adv_error = advice_error(advice, opt_allocation)
    total_capacity = sum(bus.capacity for bus in context.buses)
    methods = []
    if include_baselines:
        methods.extend(
            [
                ("Random", "", lambda: random_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg, seed)),
                ("Greedy", "", lambda: greedy_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
            ]
        )
    methods.extend(
        [
            ("IPD", "", lambda: ipd_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)),
            ("Prediction-only", "", lambda: prediction_only_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg, advice=advice)),
        ]
    )
    for theta in thetas:
        theta_cfg = dict(context.algorithm_cfg)
        theta_cfg["theta"] = theta
        methods.append(
            ("RP-LAIPD", theta, lambda cfg=theta_cfg: rp_laipd_dispatch(orders, context.buses, len(context.stations), cfg, advice=advice))
        )
    rows = []
    for method, theta, fn in methods:
        result, runtime_ms = _timed(fn)
        violations = count_constraint_violations(
            result.accepted, orders, context.buses, len(context.stations), context.algorithm_cfg
        )
        row = dispatch_row(
            slot_id,
            method,
            theta,
            1.0 if corruption == "none" else level,
            corruption,
            level,
            pred_error,
            adv_error,
            orders,
            result,
            opt_result,
            runtime_ms,
            total_capacity=total_capacity,
            violations=violations,
            seed=seed,
            split=split,
            prediction_model=model,
            prediction_time_ms=prediction_ms,
            predictive_lp_time_ms=lp_ms,
            slot_preparation_ms=prep_ms,
        )
        row["high_value_rejected"] = sum(
            order.priority >= context.high_value_threshold and order.id not in result.accepted for order in orders
        )
        row["constraint_violations"] = row["violations"]
        rows.append(row)
    return rows
