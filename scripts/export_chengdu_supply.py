#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.experiments import ChengduExperimentContext


def main() -> None:
    with Path("configs/chengdu.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    root = Path(config["data"]["output_dir"]) / "summary"
    root.mkdir(parents=True, exist_ok=True)
    route_rows = []
    for bus in context.buses:
        for stop_order, station_id in enumerate(bus.route):
            route_rows.append({
                "bus_id": bus.id, "capacity": bus.capacity, "available_from": bus.available_from,
                "depart_at": bus.depart_at, "stop_order": stop_order, "station_id": station_id,
                "travel_time": bus.station_travel_time[station_id],
            })
    pd.DataFrame(route_rows).to_csv(root / "bus_routes.csv", index=False)
    stations = range(len(context.stations))
    pd.DataFrame({
        "station_id": stations,
        "station_capacity": [context.algorithm_cfg["station_capacity"][x] for x in stations],
        "station_time_capacity": [context.algorithm_cfg["station_time_capacity"][x] for x in stations],
        "train_type_value": [context.type_values[x] for x in stations],
    }).to_csv(root / "station_resources.csv", index=False)
    pd.DataFrame({
        "bus_id": [bus.id for bus in context.buses],
        "bus_time_capacity": [context.algorithm_cfg["bus_time_capacity"][bus.id] for bus in context.buses],
    }).to_csv(root / "bus_resources.csv", index=False)
    print(f"Frozen supply tables written to {root}")


if __name__ == "__main__":
    main()
