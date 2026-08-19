from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


MODEL_HISTORICAL = "HistoricalAverage"
MODEL_RIDGE_AR = "RidgeAR"
MODEL_HGB = "HistGradientBoosting"


def _time_table(demand: pd.DataFrame) -> pd.DataFrame:
    slots = (
        demand[["slot_id", "service_date", "split", "slot_in_day"]]
        .drop_duplicates()
        .sort_values(["service_date", "slot_in_day"])
        .reset_index(drop=True)
    )
    slots["service_date"] = pd.to_datetime(slots["service_date"])
    slots["time_index"] = np.arange(len(slots), dtype=int)
    slots["day_of_week"] = slots["service_date"].dt.dayofweek.astype(int)
    return slots


def _matrix(demand: pd.DataFrame, slots: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    type_ids = sorted(demand["type_id"].unique().astype(int).tolist())
    pivot = demand.pivot(index="slot_id", columns="type_id", values="observed_count").reindex(
        index=slots["slot_id"], columns=type_ids, fill_value=0.0
    )
    return pivot.to_numpy(dtype=float), type_ids


def _base_features(slots: pd.DataFrame, seasonal_period: int) -> np.ndarray:
    slot = slots["slot_in_day"].to_numpy(dtype=float)
    dow = slots["day_of_week"].to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * slot / seasonal_period),
            np.cos(2 * np.pi * slot / seasonal_period),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ]
    )


def _lag_features(series: np.ndarray, lags: list[int]) -> np.ndarray:
    output = np.full((len(series), len(lags)), np.nan, dtype=float)
    for column, lag in enumerate(lags):
        if lag < len(series):
            output[lag:, column] = series[:-lag]
    return output


def _historical_average_predictions(demand: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    train = demand.loc[demand["split"] == "train"].copy()
    train["service_date"] = pd.to_datetime(train["service_date"])
    train["day_of_week"] = train["service_date"].dt.dayofweek
    target_dates = pd.to_datetime(target["service_date"])
    target_dow = target_dates.dt.dayofweek
    primary = train.groupby(["type_id", "day_of_week", "slot_in_day"])["observed_count"].mean()
    fallback = train.groupby(["type_id", "slot_in_day"])["observed_count"].mean()
    type_mean = train.groupby("type_id")["observed_count"].mean()
    predictions = []
    for type_id, dow, slot in zip(target["type_id"], target_dow, target["slot_in_day"]):
        value = primary.get((type_id, dow, slot), np.nan)
        if pd.isna(value):
            value = fallback.get((type_id, slot), np.nan)
        if pd.isna(value):
            value = type_mean.get(type_id, 0.0)
        predictions.append(max(float(value), 0.0))
    return np.asarray(predictions, dtype=float)


def generate_predictions(demand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    required = {"slot_id", "service_date", "split", "slot_in_day", "type_id", "observed_count"}
    missing = required - set(demand.columns)
    if missing:
        raise ValueError(f"Demand table is missing columns: {sorted(missing)}")
    demand = demand.copy()
    demand["service_date"] = pd.to_datetime(demand["service_date"])
    slots = _time_table(demand)
    values, type_ids = _matrix(demand, slots)
    seasonal_period = int(cfg.get("seasonal_period", 48))
    lags = sorted({int(value) for value in cfg.get("lag_steps", [1, 2, 48, 96, 336])})
    max_lag = max(lags)
    base = _base_features(slots, seasonal_period)
    train_time = slots["split"].eq("train").to_numpy()
    target_time = ~train_time

    target = demand.loc[demand["split"] != "train"].sort_values(["service_date", "slot_in_day", "type_id"]).copy()
    outputs = []

    historical = target.copy()
    historical["model"] = MODEL_HISTORICAL
    historical["predicted_count"] = _historical_average_predictions(demand, historical)
    outputs.append(historical)

    ridge_rows = []
    hgb_x_train = []
    hgb_y_train = []
    hgb_x_target = []
    hgb_target_keys = []
    train_type_means = values[train_time].mean(axis=0)

    for column, type_id in enumerate(type_ids):
        series = values[:, column]
        lagged = _lag_features(series, lags)
        features = np.column_stack([base, lagged])
        valid_train = train_time & (np.arange(len(series)) >= max_lag) & np.isfinite(features).all(axis=1)
        ridge = Ridge(alpha=float(cfg.get("ridge_alpha", 1.0)))
        ridge.fit(features[valid_train], series[valid_train])
        ridge_pred = np.maximum(ridge.predict(features[target_time]), 0.0)
        target_slots = slots.loc[target_time]
        ridge_rows.append(
            pd.DataFrame(
                {
                    "slot_id": target_slots["slot_id"].to_numpy(),
                    "service_date": target_slots["service_date"].to_numpy(),
                    "split": target_slots["split"].to_numpy(),
                    "slot_in_day": target_slots["slot_in_day"].to_numpy(),
                    "type_id": type_id,
                    "observed_count": series[target_time],
                    "model": MODEL_RIDGE_AR,
                    "predicted_count": ridge_pred,
                }
            )
        )

        type_feature = np.full((len(series), 1), float(type_id))
        mean_feature = np.full((len(series), 1), float(train_type_means[column]))
        hgb_features = np.column_stack([type_feature, mean_feature, features])
        hgb_x_train.append(hgb_features[valid_train])
        hgb_y_train.append(series[valid_train])
        hgb_x_target.append(hgb_features[target_time])
        hgb_target_keys.extend((row.slot_id, row.service_date, row.split, row.slot_in_day, type_id, observed)
                               for row, observed in zip(target_slots.itertuples(), series[target_time]))

    outputs.append(pd.concat(ridge_rows, ignore_index=True))

    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(cfg.get("gradient_boosting_max_iter", 100)),
        max_depth=int(cfg.get("gradient_boosting_max_depth", 6)),
        learning_rate=float(cfg.get("gradient_boosting_learning_rate", 0.05)),
        random_state=int(cfg.get("random_seed", 2026)),
    )
    hgb.fit(np.vstack(hgb_x_train), np.concatenate(hgb_y_train))
    hgb_pred = np.maximum(hgb.predict(np.vstack(hgb_x_target)), 0.0)
    hgb_rows = pd.DataFrame(
        hgb_target_keys,
        columns=["slot_id", "service_date", "split", "slot_in_day", "type_id", "observed_count"],
    )
    hgb_rows["model"] = MODEL_HGB
    hgb_rows["predicted_count"] = hgb_pred
    outputs.append(hgb_rows)

    result = pd.concat(outputs, ignore_index=True)
    result["predicted_count"] = result["predicted_count"].clip(lower=0.0)
    return result.sort_values(["split", "model", "service_date", "slot_in_day", "type_id"]).reset_index(drop=True)


def generate_hgb_checkpoint_predictions(
    demand: pd.DataFrame,
    cfg: dict,
    iterations: list[int],
) -> pd.DataFrame:
    """Train HGB checkpoints with identical features and different max_iter.

    Every checkpoint is fit only on Train rows. Lagged observations from the
    past are available to the rolling forecast, but no future target is used.
    """
    required = {"slot_id", "service_date", "split", "slot_in_day", "type_id", "observed_count"}
    missing = required - set(demand.columns)
    if missing:
        raise ValueError(f"Demand table is missing columns: {sorted(missing)}")
    iteration_values = sorted({int(value) for value in iterations})
    if not iteration_values or iteration_values[0] <= 0:
        raise ValueError("Checkpoint iterations must be positive integers")

    demand = demand.copy()
    demand["service_date"] = pd.to_datetime(demand["service_date"])
    slots = _time_table(demand)
    values, type_ids = _matrix(demand, slots)
    seasonal_period = int(cfg.get("seasonal_period", 48))
    lags = sorted({int(value) for value in cfg.get("lag_steps", [1, 2, 48, 96, 336])})
    max_lag = max(lags)
    base = _base_features(slots, seasonal_period)
    train_time = slots["split"].eq("train").to_numpy()
    target_time = ~train_time
    target_slots = slots.loc[target_time]
    train_type_means = values[train_time].mean(axis=0)

    x_train, y_train, x_target, target_keys = [], [], [], []
    for column, type_id in enumerate(type_ids):
        series = values[:, column]
        features = np.column_stack([base, _lag_features(series, lags)])
        valid_train = train_time & (np.arange(len(series)) >= max_lag) & np.isfinite(features).all(axis=1)
        hgb_features = np.column_stack(
            [np.full((len(series), 1), float(type_id)),
             np.full((len(series), 1), float(train_type_means[column])), features]
        )
        x_train.append(hgb_features[valid_train])
        y_train.append(series[valid_train])
        x_target.append(hgb_features[target_time])
        target_keys.extend(
            (row.slot_id, row.service_date, row.split, row.slot_in_day, type_id, observed)
            for row, observed in zip(target_slots.itertuples(), series[target_time])
        )

    train_x = np.vstack(x_train)
    train_y = np.concatenate(y_train)
    target_x = np.vstack(x_target)
    keys = pd.DataFrame(
        target_keys,
        columns=["slot_id", "service_date", "split", "slot_in_day", "type_id", "observed_count"],
    )
    outputs = []
    for max_iter in iteration_values:
        model_name = f"HGB_iter_{max_iter:03d}"
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            max_depth=int(cfg.get("gradient_boosting_max_depth", 6)),
            learning_rate=float(cfg.get("gradient_boosting_learning_rate", 0.05)),
            random_state=int(cfg.get("random_seed", 2026)),
        )
        model.fit(train_x, train_y)
        output = keys.copy()
        output["model"] = model_name
        output["max_iter"] = max_iter
        output["predicted_count"] = np.maximum(model.predict(target_x), 0.0)
        outputs.append(output)
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["split", "max_iter", "service_date", "slot_in_day", "type_id"]).reset_index(drop=True)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["split", "model"] + (["max_iter"] if "max_iter" in predictions.columns else [])
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(group_columns, keys))
        truth = group["observed_count"].to_numpy(dtype=float)
        predicted = group["predicted_count"].to_numpy(dtype=float)
        error = predicted - truth
        nonzero = truth > 0
        rows.append(
            {
                **identifiers,
                "observations": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(truth)), 1e-9)),
                "smape": float(np.mean(2 * np.abs(error) / np.maximum(np.abs(truth) + np.abs(predicted), 1e-9))),
                "mape_nonzero": float(np.mean(np.abs(error[nonzero]) / truth[nonzero])) if nonzero.any() else np.nan,
                "zero_demand_rate": float(np.mean(~nonzero)),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CSVPredictionProvider:
    predictions: pd.DataFrame
    model: str

    @classmethod
    def from_csv(cls, path: Path | str, model: str) -> "CSVPredictionProvider":
        return cls(pd.read_csv(path), model)

    def predict_slot(self, slot_id: str) -> dict[int, float]:
        selected = self.predictions.loc[
            (self.predictions["slot_id"] == slot_id) & (self.predictions["model"] == self.model)
        ]
        if selected.empty:
            raise KeyError(f"No predictions for slot={slot_id!r}, model={self.model!r}")
        return {int(row.type_id): float(row.predicted_count) for row in selected.itertuples()}
