from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from rood_dasfaa2019.data.chengdu import assign_split, load_hex_centers, prepare_chengdu_dataset


def _hex_row(grid_id: str, lon: float, lat: float) -> list:
    return [
        grid_id,
        lon - 0.01,
        lat,
        lon - 0.005,
        lat + 0.01,
        lon + 0.005,
        lat + 0.01,
        lon + 0.01,
        lat,
        lon + 0.005,
        lat - 0.01,
        lon - 0.005,
        lat - 0.01,
    ]


def test_assign_split_is_chronological():
    cfg = {
        "train_start": "2016-11-01",
        "train_end": "2016-11-02",
        "validation_start": "2016-11-03",
        "validation_end": "2016-11-03",
        "test_start": "2016-11-04",
        "test_end": "2016-11-04",
        "shifted_test_start": "2016-11-05",
        "shifted_test_end": "2016-11-05",
    }
    dates = pd.Series(pd.to_datetime(["2016-11-01", "2016-11-03", "2016-11-04", "2016-11-05"]))
    assert assign_split(dates, cfg).tolist() == ["train", "validation", "test", "shifted_test"]


def test_load_hex_centers_skips_malformed_row(tmp_path: Path):
    path = tmp_path / "hexagon_grid_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_hex_row("a", 104.0, 30.0))
        writer.writerow(["bad", 0, 0])
    centers, malformed = load_hex_centers(path)
    assert centers["grid_id"].tolist() == ["a"]
    assert malformed == 1


def test_prepare_chengdu_dataset_deduplicates_and_caps_slots(tmp_path: Path):
    raw = tmp_path / "raw"
    orders_dir = raw / "total_ride_request"
    orders_dir.mkdir(parents=True)
    with (raw / "hexagon_grid_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_hex_row("a", 104.0, 30.0))
        writer.writerow(_hex_row("b", 104.2, 30.2))

    # 2016-11-01 00:00:00 and later in Asia/Shanghai.
    rows = [
        ["o1", 1477929600, 1477930200, 104.0, 30.0, 104.0, 30.0, 2.0],
        ["o1", 1477929600, 1477930200, 104.0, 30.0, 104.0, 30.0, 2.0],
        ["o2", 1477929660, 1477930260, 104.0, 30.0, 104.2, 30.2, 3.0],
        ["o3", 1477929720, 1477930320, 104.0, 30.0, 104.2, 30.2, 4.0],
    ]
    with (orders_dir / "order_20161101").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    cfg = {
        "data": {
            "raw_data_dir": str(raw),
            "output_dir": str(tmp_path / "out"),
            "timezone": "Asia/Shanghai",
            "slot_minutes": 30,
            "station_count": 2,
            "min_duration_seconds": 1,
            "max_duration_seconds": 14400,
            "max_orders_per_slot": 2,
            "random_seed": 7,
            "train_start": "2016-11-01",
            "train_end": "2016-11-01",
            "validation_start": "2016-11-02",
            "validation_end": "2016-11-02",
            "test_start": "2016-11-03",
            "test_end": "2016-11-03",
            "shifted_test_start": "2016-11-04",
            "shifted_test_end": "2016-11-04",
        }
    }
    outputs = prepare_chengdu_dataset(cfg)
    quality = pd.read_csv(outputs["quality"])
    assert int(quality.loc[quality["source_file"] == "TOTAL", "exact_duplicate_rows"].iloc[0]) == 1
    experiment = pd.read_csv(outputs["experiment_orders"])
    assert len(experiment) == 2
    demand = pd.read_csv(outputs["demand"])
    assert len(demand) == 4 * 48 * 2
