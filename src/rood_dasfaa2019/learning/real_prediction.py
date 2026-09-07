from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_HISTORICAL = "HistoricalAverage"
MODEL_RIDGE_AR = "RidgeAR"
MODEL_HGB = "HGB"
MODEL_HISTORICAL_ORIGINAL = "HistoricalAverageOriginal"


@dataclass
class DirectForecastDataset:
    metadata: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray


@dataclass
class HorizonForecastResult:
    predictions: pd.DataFrame
    historical_original_predictions: pd.DataFrame
    selection: pd.DataFrame
    training_manifest: list[dict[str, Any]]
    common_test_slots: list[str]


def _validate_demand(demand: pd.DataFrame) -> None:
    required = {
        "slot_id",
        "service_date",
        "split",
        "slot_in_day",
        "type_id",
        "observed_count",
    }
    missing = required - set(demand.columns)
    if missing:
        raise ValueError(f"Demand table is missing columns: {sorted(missing)}")


def _time_table(demand: pd.DataFrame) -> pd.DataFrame:
    slots = (
        demand[["slot_id", "service_date", "split", "slot_in_day"]]
        .drop_duplicates()
        .sort_values(["service_date", "slot_in_day"])
        .reset_index(drop=True)
    )
    slots["slot_id"] = slots["slot_id"].astype(str)
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


def _calendar_features(slot_in_day: int, day_of_week: int, seasonal_period: int) -> dict[str, float]:
    return {
        "target_slot_sin": float(np.sin(2 * np.pi * slot_in_day / seasonal_period)),
        "target_slot_cos": float(np.cos(2 * np.pi * slot_in_day / seasonal_period)),
        "target_dow_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
        "target_dow_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
    }


def build_direct_forecast_dataset(
    demand: pd.DataFrame,
    horizon: int,
    origin_lags: list[int],
    seasonal_period: int,
) -> DirectForecastDataset:
    """Build direct-horizon features without reading demand after the origin.

    For target index ``t`` and horizon ``h``, the forecast origin is ``t-h``.
    Demand feature ``origin_lag_L`` reads only ``y[t-h-L]``.  Target calendar
    fields are allowed because they are known at forecast time.
    """
    _validate_demand(demand)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    lags = sorted({int(value) for value in origin_lags})
    if not lags or lags[0] < 0:
        raise ValueError("origin_lags must contain non-negative integers")

    demand = demand.copy()
    demand["slot_id"] = demand["slot_id"].astype(str)
    slots = _time_table(demand)
    values, type_ids = _matrix(demand, slots)
    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    targets: list[float] = []

    first_target = horizon + max(lags)
    for target_index in range(first_target, len(slots)):
        origin_index = target_index - horizon
        target_slot = slots.iloc[target_index]
        origin_slot = slots.iloc[origin_index]
        calendar = _calendar_features(
            int(target_slot["slot_in_day"]),
            int(target_slot["day_of_week"]),
            seasonal_period,
        )
        for column, type_id in enumerate(type_ids):
            metadata_rows.append(
                {
                    "horizon": horizon,
                    "forecast_origin": str(origin_slot["slot_id"]),
                    "slot_id": str(target_slot["slot_id"]),
                    "service_date": target_slot["service_date"].strftime("%Y-%m-%d"),
                    "slot_in_day": int(target_slot["slot_in_day"]),
                    "day_of_week": int(target_slot["day_of_week"]),
                    "type_id": int(type_id),
                    "split": str(target_slot["split"]),
                    "observed_count": float(values[target_index, column]),
                }
            )
            row = {
                "station_id": int(type_id),
                **calendar,
            }
            for lag in lags:
                row[f"origin_demand_lag_{lag}"] = float(values[origin_index - lag, column])
            feature_rows.append(row)
            targets.append(float(values[target_index, column]))

    return DirectForecastDataset(
        metadata=pd.DataFrame(metadata_rows),
        features=pd.DataFrame(feature_rows),
        target=np.asarray(targets, dtype=float),
    )


def _ridge_matrix(features: pd.DataFrame, station_count: int) -> np.ndarray:
    station_ids = features["station_id"].to_numpy(dtype=int)
    station_one_hot = np.eye(station_count, dtype=float)[station_ids]
    numeric = features.drop(columns="station_id").to_numpy(dtype=float)
    return np.column_stack([station_one_hot, numeric])


def _hgb_frame(features: pd.DataFrame, station_count: int) -> pd.DataFrame:
    result = features.copy()
    result["station_id"] = pd.Categorical(
        result["station_id"].astype(int), categories=list(range(station_count))
    )
    return result


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def _historical_tables(demand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, float]:
    train = demand.loc[demand["split"].eq("train")].copy()
    train["service_date"] = pd.to_datetime(train["service_date"])
    train["day_of_week"] = train["service_date"].dt.dayofweek.astype(int)
    fine = train.groupby(["type_id", "day_of_week", "slot_in_day"])["observed_count"].agg(
        fine_mean="mean", fine_count="size"
    ).reset_index()
    coarse = train.groupby(["type_id", "slot_in_day"])["observed_count"].mean().rename("coarse_mean").reset_index()
    station = train.groupby("type_id")["observed_count"].mean()
    return fine, coarse, station, float(train["observed_count"].mean())


def historical_average_prediction(
    metadata: pd.DataFrame,
    demand: pd.DataFrame,
    fine_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    fine, coarse, station_mean, global_mean = _historical_tables(demand)
    joined = metadata.merge(
        fine,
        on=["type_id", "day_of_week", "slot_in_day"],
        how="left",
    ).merge(coarse, on=["type_id", "slot_in_day"], how="left")
    station_fallback = joined["type_id"].map(station_mean).fillna(global_mean)
    coarse_value = joined["coarse_mean"].fillna(station_fallback)
    fine_value = joined["fine_mean"].fillna(coarse_value)
    prediction = fine_weight * fine_value + (1.0 - fine_weight) * coarse_value
    fine_count = joined["fine_count"].fillna(0.0).to_numpy(dtype=float)
    return np.maximum(prediction.to_numpy(dtype=float), 0.0), fine_count


def _prediction_frame(metadata: pd.DataFrame, model: str, predicted: np.ndarray) -> pd.DataFrame:
    output = metadata[
        [
            "horizon",
            "forecast_origin",
            "slot_id",
            "service_date",
            "slot_in_day",
            "type_id",
            "observed_count",
            "split",
        ]
    ].copy()
    output.insert(0, "model", model)
    output["predicted_count"] = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    return output


def train_horizon_predictors(
    demand: pd.DataFrame,
    cfg: dict,
    model_dir: Path | None = None,
) -> HorizonForecastResult:
    """Select predictor hyperparameters on Validation and evaluate later on Test."""
    _validate_demand(demand)
    horizons = [int(value) for value in cfg.get("horizons", [1, 2, 4, 8])]
    origin_lags = [int(value) for value in cfg.get("origin_lags", [0, 1, 47, 95, 335])]
    seasonal_period = int(cfg.get("seasonal_period", 48))
    station_count = int(demand["type_id"].nunique())
    alpha_grid = [float(value) for value in cfg.get("ridge_alpha_grid", [0.01, 0.1, 1, 10, 100])]
    hgb_iterations = [int(value) for value in cfg.get("hgb_max_iter_grid", [25, 50, 100, 200])]
    hgb_losses = [str(value) for value in cfg.get("hgb_losses", ["squared_error", "poisson"])]
    historical_weights = [float(value) for value in cfg.get("historical_fine_weight_grid", [0, 0.25, 0.5, 0.75, 1])]
    random_seed = int(cfg.get("random_seed", 2026))
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        horizon: build_direct_forecast_dataset(demand, horizon, origin_lags, seasonal_period)
        for horizon in horizons
    }
    common_test_slots = sorted(
        set.intersection(
            *[
                set(dataset.metadata.loc[dataset.metadata["split"].eq("test"), "slot_id"])
                for dataset in datasets.values()
            ]
        )
    )
    outputs = []
    original_outputs = []
    selection_rows = []
    manifest = []
    train_dates = pd.to_datetime(demand.loc[demand["split"].eq("train"), "service_date"])
    validation_dates = pd.to_datetime(demand.loc[demand["split"].eq("validation"), "service_date"])

    for horizon in horizons:
        dataset = datasets[horizon]
        train_mask = dataset.metadata["split"].eq("train").to_numpy()
        validation_mask = dataset.metadata["split"].eq("validation").to_numpy()
        target_mask = dataset.metadata["split"].isin(["validation", "test"]).to_numpy()
        target_metadata = dataset.metadata.loc[target_mask].reset_index(drop=True)
        target_features = dataset.features.loc[target_mask].reset_index(drop=True)

        historical_candidates = []
        for weight in historical_weights:
            predicted, fine_count = historical_average_prediction(
                dataset.metadata.loc[validation_mask].reset_index(drop=True), demand, weight
            )
            row = {
                "model": MODEL_HISTORICAL,
                "horizon": horizon,
                "parameter": "fine_weight",
                "parameter_value": weight,
                "loss": "calendar_mean",
                "validation_mae": _mae(dataset.target[validation_mask], predicted),
                "actual_iterations": np.nan,
                "fine_group_count_min": float(fine_count.min()),
                "fine_group_count_median": float(np.median(fine_count)),
                "fine_group_count_max": float(fine_count.max()),
            }
            historical_candidates.append(row)
        best_historical = min(
            historical_candidates,
            key=lambda row: (row["validation_mae"], row["parameter_value"]),
        )
        for row in historical_candidates:
            row["selected"] = row is best_historical
            selection_rows.append(row)
        historical_pred, fine_count = historical_average_prediction(
            target_metadata, demand, float(best_historical["parameter_value"])
        )
        original_pred, _ = historical_average_prediction(target_metadata, demand, 1.0)
        outputs.append(_prediction_frame(target_metadata, MODEL_HISTORICAL, historical_pred))
        original_outputs.append(
            _prediction_frame(target_metadata, MODEL_HISTORICAL_ORIGINAL, original_pred)
        )
        if model_dir is not None:
            fine_table, coarse_table, station_mean, global_mean = _historical_tables(demand)
            joblib.dump(
                {
                    "model": MODEL_HISTORICAL,
                    "horizon": horizon,
                    "selected_fine_weight": float(best_historical["parameter_value"]),
                    "fine_table": fine_table,
                    "coarse_table": coarse_table,
                    "station_mean": station_mean,
                    "global_mean": global_mean,
                    "fit_split": "Train only",
                },
                model_dir / f"historical_average_h{horizon}.joblib",
            )

        ridge_train_x = _ridge_matrix(dataset.features.loc[train_mask], station_count)
        ridge_validation_x = _ridge_matrix(dataset.features.loc[validation_mask], station_count)
        ridge_target_x = _ridge_matrix(target_features, station_count)
        ridge_candidates = []
        ridge_models = {}
        for alpha in alpha_grid:
            model = Pipeline(
                [
                    ("standard_scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
            model.fit(ridge_train_x, dataset.target[train_mask])
            predicted = np.maximum(model.predict(ridge_validation_x), 0.0)
            ridge_models[alpha] = model
            ridge_candidates.append(
                {
                    "model": MODEL_RIDGE_AR,
                    "horizon": horizon,
                    "parameter": "alpha",
                    "parameter_value": alpha,
                    "loss": "squared_error",
                    "validation_mae": _mae(dataset.target[validation_mask], predicted),
                    "actual_iterations": np.nan,
                }
            )
        best_ridge = min(ridge_candidates, key=lambda row: (row["validation_mae"], row["parameter_value"]))
        for row in ridge_candidates:
            row["selected"] = row is best_ridge
            selection_rows.append(row)
        ridge_model = ridge_models[float(best_ridge["parameter_value"])]
        ridge_pred = np.maximum(ridge_model.predict(ridge_target_x), 0.0)
        outputs.append(_prediction_frame(target_metadata, MODEL_RIDGE_AR, ridge_pred))
        if model_dir is not None:
            joblib.dump(
                {
                    "model": ridge_model,
                    "horizon": horizon,
                    "feature_columns": dataset.features.columns.tolist(),
                    "station_encoding": "fixed one-hot",
                },
                model_dir / f"ridge_ar_h{horizon}.joblib",
            )

        hgb_train_x = _hgb_frame(dataset.features.loc[train_mask], station_count)
        hgb_validation_x = _hgb_frame(dataset.features.loc[validation_mask], station_count)
        hgb_target_x = _hgb_frame(target_features, station_count)
        hgb_candidates = []
        hgb_models = {}
        for loss in hgb_losses:
            for max_iter in hgb_iterations:
                model = HistGradientBoostingRegressor(
                    loss=loss,
                    max_iter=max_iter,
                    max_depth=int(cfg.get("hgb_max_depth", 6)),
                    learning_rate=float(cfg.get("hgb_learning_rate", 0.05)),
                    categorical_features=["station_id"],
                    early_stopping=False,
                    random_state=random_seed,
                )
                model.fit(hgb_train_x, dataset.target[train_mask])
                predicted = np.maximum(model.predict(hgb_validation_x), 0.0)
                key = (loss, max_iter)
                hgb_models[key] = model
                hgb_candidates.append(
                    {
                        "model": MODEL_HGB,
                        "horizon": horizon,
                        "parameter": "loss/max_iter",
                        "parameter_value": max_iter,
                        "loss": loss,
                        "validation_mae": _mae(dataset.target[validation_mask], predicted),
                        "actual_iterations": int(model.n_iter_),
                    }
                )
        best_hgb = min(
            hgb_candidates,
            key=lambda row: (row["validation_mae"], row["parameter_value"], row["loss"]),
        )
        for row in hgb_candidates:
            row["selected"] = row is best_hgb
            selection_rows.append(row)
        hgb_key = (str(best_hgb["loss"]), int(best_hgb["parameter_value"]))
        hgb_model = hgb_models[hgb_key]
        hgb_pred = np.maximum(hgb_model.predict(hgb_target_x), 0.0)
        outputs.append(_prediction_frame(target_metadata, MODEL_HGB, hgb_pred))
        if model_dir is not None:
            joblib.dump(
                {
                    "model": hgb_model,
                    "horizon": horizon,
                    "feature_columns": dataset.features.columns.tolist(),
                    "station_id_categorical": True,
                    "early_stopping": False,
                    "actual_iterations": int(hgb_model.n_iter_),
                },
                model_dir / f"hgb_h{horizon}.joblib",
            )

        manifest.extend(
            [
                {
                    "model": MODEL_HISTORICAL,
                    "horizon": horizon,
                    "selected_fine_weight": float(best_historical["parameter_value"]),
                    "selection_basis": "minimum Validation MAE",
                    "fine_group_count_median": float(np.median(fine_count)),
                },
                {
                    "model": MODEL_RIDGE_AR,
                    "horizon": horizon,
                    "selected_alpha": float(best_ridge["parameter_value"]),
                    "selection_basis": "minimum Validation MAE",
                    "standard_scaler_fit": "Train only",
                },
                {
                    "model": MODEL_HGB,
                    "horizon": horizon,
                    "selected_loss": str(best_hgb["loss"]),
                    "selected_max_iter": int(best_hgb["parameter_value"]),
                    "actual_iterations": int(hgb_model.n_iter_),
                    "selection_basis": "minimum Validation MAE",
                    "early_stopping": False,
                    "station_id_categorical": True,
                    "max_depth": int(cfg.get("hgb_max_depth", 6)),
                    "learning_rate": float(cfg.get("hgb_learning_rate", 0.05)),
                    "random_state": random_seed,
                },
            ]
        )

    predictions = pd.concat(outputs, ignore_index=True)
    original = pd.concat(original_outputs, ignore_index=True)
    for frame in (predictions, original):
        frame.drop(
            frame.index[
                frame["split"].eq("test") & ~frame["slot_id"].isin(common_test_slots)
            ],
            inplace=True,
        )
        frame.sort_values(["split", "model", "horizon", "slot_id", "type_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    shared_manifest = {
        "training_start": train_dates.min().strftime("%Y-%m-%d"),
        "training_end": train_dates.max().strftime("%Y-%m-%d"),
        "validation_start": validation_dates.min().strftime("%Y-%m-%d"),
        "validation_end": validation_dates.max().strftime("%Y-%m-%d"),
        "target": "one 30-minute service-slot station demand",
        "origin_lags": origin_lags,
        "feature_definition": datasets[horizons[0]].features.columns.tolist(),
        "test_policy": "final evaluation only; common valid target slots across all horizons",
    }
    manifest = [
        {
            **shared_manifest,
            "training_target_rows": int(
                datasets[int(row["horizon"])].metadata["split"].eq("train").sum()
            ),
            **row,
        }
        for row in manifest
    ]
    return HorizonForecastResult(
        predictions=predictions,
        historical_original_predictions=original,
        selection=pd.DataFrame(selection_rows),
        training_manifest=manifest,
        common_test_slots=common_test_slots,
    )


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["split", "model"]
    for optional in ("horizon", "max_iter"):
        if optional in predictions.columns:
            group_columns.append(optional)
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
                "mse": float(np.mean(error**2)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(truth)), 1e-9)),
                "mape_nonzero": float(np.mean(np.abs(error[nonzero]) / truth[nonzero])) if nonzero.any() else np.nan,
                "zero_demand_rate_excluded_from_mape": float(np.mean(~nonzero)),
                "smape": float(
                    np.mean(2 * np.abs(error) / np.maximum(np.abs(truth) + np.abs(predicted), 1e-9))
                ),
            }
        )
    return pd.DataFrame(rows)


def generate_predictions(demand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Backward-compatible h=1 entry using the leakage-safe selected models."""
    local_cfg = {
        **cfg,
        "horizons": [1],
        "origin_lags": cfg.get("origin_lags", [0, 1, 3]),
        "ridge_alpha_grid": cfg.get("ridge_alpha_grid", [cfg.get("ridge_alpha", 1.0)]),
        "hgb_max_iter_grid": cfg.get("hgb_max_iter_grid", [cfg.get("gradient_boosting_max_iter", 100)]),
        "hgb_losses": cfg.get("hgb_losses", ["squared_error"]),
        "hgb_max_depth": cfg.get("hgb_max_depth", cfg.get("gradient_boosting_max_depth", 6)),
        "hgb_learning_rate": cfg.get("hgb_learning_rate", cfg.get("gradient_boosting_learning_rate", 0.05)),
        "historical_fine_weight_grid": cfg.get("historical_fine_weight_grid", [1.0]),
    }
    result = train_horizon_predictors(demand, local_cfg)
    return result.predictions


def generate_hgb_checkpoint_predictions(
    demand: pd.DataFrame,
    cfg: dict,
    iterations: list[int],
) -> pd.DataFrame:
    """Backward-compatible leakage-safe h=1 HGB checkpoint generator."""
    outputs = []
    for iteration in sorted({int(value) for value in iterations}):
        local_cfg = {
            **cfg,
            "horizons": [1],
            "origin_lags": cfg.get("origin_lags", [0, 1, 3]),
            "ridge_alpha_grid": [cfg.get("ridge_alpha", 1.0)],
            "hgb_max_iter_grid": [iteration],
            "hgb_losses": ["squared_error"],
            "historical_fine_weight_grid": [1.0],
        }
        result = train_horizon_predictors(demand, local_cfg)
        selected = result.predictions.loc[result.predictions["model"].eq(MODEL_HGB)].copy()
        selected["model"] = f"HGB_iter_{iteration:03d}"
        selected["max_iter"] = iteration
        outputs.append(selected)
    return pd.concat(outputs, ignore_index=True)


@dataclass
class CSVPredictionProvider:
    predictions: pd.DataFrame
    model: str

    @classmethod
    def from_csv(cls, path: Path | str, model: str) -> "CSVPredictionProvider":
        return cls(pd.read_csv(path), model)

    def predict_slot(self, slot_id: str, horizon: int | None = None) -> dict[int, float]:
        selected = self.predictions.loc[
            (self.predictions["slot_id"].astype(str) == str(slot_id))
            & (self.predictions["model"] == self.model)
        ]
        if horizon is not None and "horizon" in selected.columns:
            selected = selected.loc[selected["horizon"].eq(horizon)]
        if selected.empty:
            raise KeyError(f"No predictions for slot={slot_id!r}, model={self.model!r}")
        return {
            int(row.type_id): float(row.predicted_count)
            for row in selected.itertuples()
        }
