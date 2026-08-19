from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ORDER_COLUMNS = [
    "request_id",
    "start_timestamp",
    "end_timestamp",
    "origin_lon",
    "origin_lat",
    "destination_lon",
    "destination_lat",
    "raw_reward",
]


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.start, self.end, inclusive="both")


def _date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def split_ranges(cfg: dict) -> list[SplitRange]:
    return [
        SplitRange("train", _date(cfg["train_start"]), _date(cfg["train_end"])),
        SplitRange("validation", _date(cfg["validation_start"]), _date(cfg["validation_end"])),
        SplitRange("test", _date(cfg["test_start"]), _date(cfg["test_end"])),
        SplitRange("shifted_test", _date(cfg["shifted_test_start"]), _date(cfg["shifted_test_end"])),
    ]


def assign_split(dates: pd.Series, cfg: dict) -> pd.Series:
    result = pd.Series("excluded", index=dates.index, dtype="string")
    for split in split_ranges(cfg):
        result.loc[split.contains(dates)] = split.name
    return result


def load_hex_centers(path: Path) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split(",")
            if len(fields) != 13:
                malformed += 1
                continue
            try:
                coords = np.asarray([float(value) for value in fields[1:]], dtype=float).reshape(6, 2)
            except ValueError:
                malformed += 1
                continue
            rows.append(
                {
                    "grid_id": fields[0],
                    "center_lon": float(coords[:, 0].mean()),
                    "center_lat": float(coords[:, 1].mean()),
                    "source_line": line_number,
                }
            )
    centers = pd.DataFrame(rows)
    if centers.empty:
        raise ValueError(f"No valid hexagons found in {path}")
    if centers["grid_id"].duplicated().any():
        raise ValueError("hexagon_grid_table.csv contains duplicate grid IDs")
    return centers, malformed


def _read_day(path: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, header=None, names=ORDER_COLUMNS)
    input_rows = len(raw)
    duplicate_rows = int(raw.duplicated(keep="first").sum())
    frame = raw.drop_duplicates(keep="first").copy()
    frame["duration_seconds"] = frame["end_timestamp"] - frame["start_timestamp"]
    invalid_duration = ~frame["duration_seconds"].between(
        int(cfg["min_duration_seconds"]), int(cfg["max_duration_seconds"]), inclusive="both"
    )
    invalid_coordinates = ~(
        frame["origin_lon"].between(-180, 180)
        & frame["destination_lon"].between(-180, 180)
        & frame["origin_lat"].between(-90, 90)
        & frame["destination_lat"].between(-90, 90)
    )
    invalid_reward = ~np.isfinite(frame["raw_reward"]) | (frame["raw_reward"] <= 0)
    frame = frame.loc[~(invalid_duration | invalid_coordinates | invalid_reward)].copy()
    local_time = pd.to_datetime(frame["start_timestamp"], unit="s", utc=True).dt.tz_convert(cfg["timezone"])
    frame["service_date"] = local_time.dt.tz_localize(None).dt.normalize()
    frame["local_datetime"] = local_time.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    slot_minutes = int(cfg["slot_minutes"])
    minute_of_day = local_time.dt.hour * 60 + local_time.dt.minute
    frame["slot_in_day"] = (minute_of_day // slot_minutes).astype(int)
    frame["arrival_minute"] = (minute_of_day % slot_minutes + local_time.dt.second / 60.0).astype(float)
    frame["split"] = assign_split(frame["service_date"], cfg)
    frame = frame.loc[frame["split"] != "excluded"].copy()
    quality = {
        "source_file": path.name,
        "input_rows": input_rows,
        "exact_duplicate_rows": duplicate_rows,
        "invalid_duration_rows": int(invalid_duration.sum()),
        "invalid_coordinate_rows": int(invalid_coordinates.sum()),
        "invalid_reward_rows": int(invalid_reward.sum()),
        "clean_rows": len(frame),
    }
    return frame, quality


def _nearest_ids(points: np.ndarray, centers: pd.DataFrame) -> np.ndarray:
    tree = cKDTree(centers[["center_lon", "center_lat"]].to_numpy(dtype=float))
    _, indices = tree.query(points, k=1)
    return indices.astype(int)


def _order_files(raw_dir: Path) -> list[Path]:
    files = sorted((raw_dir / "total_ride_request").glob("order_20*"))
    if not files:
        raise FileNotFoundError(f"No order files found under {raw_dir / 'total_ride_request'}")
    return files


def discover_train_stations(raw_dir: Path, cfg: dict, hex_centers: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    counts = np.zeros(len(hex_centers), dtype=np.int64)
    train_rewards: list[np.ndarray] = []
    for path in _order_files(raw_dir):
        date_from_name = pd.Timestamp(path.name.removeprefix("order_"))
        if not _date(cfg["train_start"]) <= date_from_name <= _date(cfg["train_end"]):
            continue
        frame, _ = _read_day(path, cfg)
        frame = frame.loc[frame["split"] == "train"]
        indices = _nearest_ids(
            frame[["destination_lon", "destination_lat"]].to_numpy(dtype=float), hex_centers
        )
        np.add.at(counts, indices, 1)
        train_rewards.append(frame["raw_reward"].to_numpy(dtype=float))
    station_count = int(cfg["station_count"])
    if np.count_nonzero(counts) < station_count:
        raise ValueError(f"Only {np.count_nonzero(counts)} destination grids occur in Train, need {station_count}")
    selected = np.argsort(counts)[::-1][:station_count]
    stations = hex_centers.iloc[selected].copy().reset_index(drop=True)
    stations.insert(0, "station_id", np.arange(station_count, dtype=int))
    stations["train_destination_count"] = counts[selected]
    rewards = np.concatenate(train_rewards)
    reward_scale = float(np.quantile(rewards, 0.99))
    return stations, reward_scale


def _append_gzip_csv(frame: pd.DataFrame, path: Path, header: bool) -> None:
    mode = "wt" if header else "at"
    with gzip.open(path, mode, encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=header)


def _sample_slots(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cap = int(cfg["max_orders_per_slot"])
    seed = int(cfg["random_seed"])
    sampled: list[pd.DataFrame] = []
    for slot_id, group in frame.groupby("slot_id", sort=True):
        if len(group) <= cap:
            sampled.append(group)
        else:
            slot_seed = seed + int(pd.util.hash_pandas_object(pd.Index([slot_id]), index=False).iloc[0] % (2**31 - 1))
            sampled.append(group.sample(n=cap, replace=False, random_state=slot_seed))
    if not sampled:
        return frame.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True).sort_values(["slot_id", "arrival_minute", "request_id"])


def _full_demand_grid(experiment_orders: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ranges = split_ranges(cfg)
    first = min(item.start for item in ranges)
    last = max(item.end for item in ranges)
    slots_per_day = 24 * 60 // int(cfg["slot_minutes"])
    dates = pd.date_range(first, last, freq="D")
    slot_rows = []
    for date in dates:
        split = next((item.name for item in ranges if item.start <= date <= item.end), "excluded")
        for slot_in_day in range(slots_per_day):
            slot_rows.append((f"{date:%Y-%m-%d}-{slot_in_day:02d}", date, split, slot_in_day))
    slots = pd.DataFrame(slot_rows, columns=["slot_id", "service_date", "split", "slot_in_day"])
    types = pd.DataFrame({"type_id": np.arange(int(cfg["station_count"]), dtype=int)})
    slots["_key"] = 1
    types["_key"] = 1
    grid = slots.merge(types, on="_key").drop(columns="_key")
    observed = (
        experiment_orders.groupby(["slot_id", "station_id"], as_index=False)
        .agg(observed_count=("party_size", "sum"), mean_value=("priority", "mean"))
        .rename(columns={"station_id": "type_id"})
    )
    grid = grid.merge(observed, on=["slot_id", "type_id"], how="left")
    grid["observed_count"] = grid["observed_count"].fillna(0.0)
    grid["mean_value"] = grid["mean_value"].fillna(0.0)
    return grid.sort_values(["service_date", "slot_in_day", "type_id"]).reset_index(drop=True)


def prepare_chengdu_dataset(config: dict) -> dict[str, Path]:
    cfg = config["data"] if "data" in config else config
    raw_dir = Path(cfg["raw_data_dir"])
    output_dir = Path(cfg["output_dir"])
    processed_dir = output_dir / "processed"
    summary_dir = output_dir / "summary"
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    hex_centers, malformed_hex_rows = load_hex_centers(raw_dir / "hexagon_grid_table.csv")
    stations, reward_scale = discover_train_stations(raw_dir, cfg, hex_centers)
    station_tree = cKDTree(stations[["center_lon", "center_lat"]].to_numpy(dtype=float))

    all_grid_points = hex_centers[["center_lon", "center_lat"]].to_numpy(dtype=float)
    _, grid_station = station_tree.query(all_grid_points, k=1)
    grid_mapping = hex_centers.copy()
    grid_mapping.insert(0, "station_id", grid_station.astype(int))
    grid_mapping.to_csv(summary_dir / "grid_to_station_mapping.csv", index=False)
    stations.to_csv(summary_dir / "station_mapping.csv", index=False)

    cleaned_path = processed_dir / "cleaned_orders.csv.gz"
    experiment_path = processed_dir / "experiment_orders.csv.gz"
    for path in (cleaned_path, experiment_path):
        if path.exists():
            path.unlink()

    quality_rows: list[dict] = []
    experiment_days: list[pd.DataFrame] = []
    cleaned_header = True
    experiment_header = True
    for source_path in _order_files(raw_dir):
        frame, quality = _read_day(source_path, cfg)
        points = frame[["destination_lon", "destination_lat"]].to_numpy(dtype=float)
        _, station_ids = station_tree.query(points, k=1)
        frame["station_id"] = station_ids.astype(int)
        frame["party_size"] = 1
        frame["priority"] = np.clip(frame["raw_reward"] / max(reward_scale, 1e-9), 0.01, 1.0)
        frame["slot_id"] = frame["service_date"].dt.strftime("%Y-%m-%d") + "-" + frame["slot_in_day"].map(lambda x: f"{x:02d}")
        cleaned_columns = [
            "request_id", "service_date", "split", "slot_id", "slot_in_day", "local_datetime",
            "arrival_minute", "station_id", "party_size", "raw_reward", "priority", "duration_seconds",
            "origin_lon", "origin_lat", "destination_lon", "destination_lat",
        ]
        clean_out = frame[cleaned_columns].copy()
        clean_out["service_date"] = clean_out["service_date"].dt.strftime("%Y-%m-%d")
        _append_gzip_csv(clean_out, cleaned_path, cleaned_header)
        cleaned_header = False

        experiment = _sample_slots(frame, cfg)[cleaned_columns].copy()
        experiment["service_date"] = experiment["service_date"].dt.strftime("%Y-%m-%d")
        _append_gzip_csv(experiment, experiment_path, experiment_header)
        experiment_header = False
        experiment_days.append(experiment)
        quality["experiment_rows"] = len(experiment)
        quality_rows.append(quality)

    experiment_orders = pd.concat(experiment_days, ignore_index=True)
    demand = _full_demand_grid(experiment_orders, cfg)
    demand.to_csv(processed_dir / "demand_by_slot_type.csv", index=False)

    quality = pd.DataFrame(quality_rows)
    quality.loc[len(quality)] = {
        "source_file": "TOTAL",
        "input_rows": quality["input_rows"].sum(),
        "exact_duplicate_rows": quality["exact_duplicate_rows"].sum(),
        "invalid_duration_rows": quality["invalid_duration_rows"].sum(),
        "invalid_coordinate_rows": quality["invalid_coordinate_rows"].sum(),
        "invalid_reward_rows": quality["invalid_reward_rows"].sum(),
        "clean_rows": quality["clean_rows"].sum(),
        "experiment_rows": quality["experiment_rows"].sum(),
    }
    quality.to_csv(summary_dir / "data_quality_report.csv", index=False)

    split_manifest = pd.DataFrame(
        [{"split": item.name, "start_date": item.start.date(), "end_date": item.end.date()} for item in split_ranges(cfg)]
    )
    split_manifest.to_csv(summary_dir / "split_manifest.csv", index=False)

    stats_rows = []
    for split, group in experiment_orders.groupby("split", sort=False):
        slot_counts = group.groupby("slot_id").size()
        stats_rows.append(
            {
                "split": split,
                "start_date": group["service_date"].min(),
                "end_date": group["service_date"].max(),
                "requests": len(group),
                "nonempty_slots": group["slot_id"].nunique(),
                "request_types": group["station_id"].nunique(),
                "average_party_size": group["party_size"].mean(),
                "average_raw_reward": group["raw_reward"].mean(),
                "average_normalized_value": group["priority"].mean(),
                "peak_requests_per_slot": slot_counts.max(),
                "average_requests_per_nonempty_slot": slot_counts.mean(),
            }
        )
    stats = pd.DataFrame(stats_rows)
    stats["configured_slots"] = stats.apply(
        lambda row: ((pd.Timestamp(row["end_date"]) - pd.Timestamp(row["start_date"])).days + 1)
        * (24 * 60 // int(cfg["slot_minutes"])),
        axis=1,
    )
    stats["empty_slot_rate"] = 1.0 - stats["nonempty_slots"] / stats["configured_slots"]
    stats.to_csv(summary_dir / "dataset_statistics.csv", index=False)

    protocol = pd.DataFrame(
        [
            ("timezone", cfg["timezone"], "Unix timestamps converted from UTC to local time"),
            ("slot_minutes", cfg["slot_minutes"], "Fixed service-slot width"),
            ("station_count", cfg["station_count"], "Top Train destination centers; all hexes assigned to nearest center"),
            ("party_size", 1, "Dataset has no party size; semi-synthetic fixed value"),
            ("reward_scale", reward_scale, "Train-only 99th percentile; priority clipped to [0.01, 1]"),
            ("duration_filter_seconds", f"[{cfg['min_duration_seconds']}, {cfg['max_duration_seconds']}]", "Applied after exact deduplication"),
            ("max_orders_per_slot", cfg["max_orders_per_slot"], "Deterministic experiment cap for tractable offline MILP"),
            ("malformed_hex_rows", malformed_hex_rows, "Rows excluded because official schema requires 13 fields"),
        ],
        columns=["setting", "value", "rationale"],
    )
    protocol.to_csv(summary_dir / "b0_protocol.csv", index=False)

    return {
        "cleaned_orders": cleaned_path,
        "experiment_orders": experiment_path,
        "demand": processed_dir / "demand_by_slot_type.csv",
        "station_mapping": summary_dir / "station_mapping.csv",
        "grid_mapping": summary_dir / "grid_to_station_mapping.csv",
        "quality": summary_dir / "data_quality_report.csv",
        "statistics": summary_dir / "dataset_statistics.csv",
        "split_manifest": summary_dir / "split_manifest.csv",
        "protocol": summary_dir / "b0_protocol.csv",
    }
