from __future__ import annotations

import numpy as np
import pandas as pd

from rood_dasfaa2019.learning.real_prediction import (
    MODEL_HGB,
    MODEL_HISTORICAL,
    MODEL_RIDGE_AR,
    CSVPredictionProvider,
    build_direct_forecast_dataset,
    generate_hgb_checkpoint_predictions,
    generate_predictions,
    prediction_metrics,
    train_horizon_predictors,
)


def _demand() -> pd.DataFrame:
    rows = []
    for time_index in range(24):
        date = pd.Timestamp("2016-11-01") + pd.Timedelta(days=time_index // 4)
        split = "train" if time_index < 16 else ("validation" if time_index < 20 else "test")
        for type_id in range(2):
            rows.append(
                {
                    "slot_id": f"s{time_index}",
                    "service_date": date,
                    "split": split,
                    "slot_in_day": time_index % 4,
                    "type_id": type_id,
                    "observed_count": float(5 + type_id + time_index % 4),
                }
            )
    return pd.DataFrame(rows)


def test_generate_predictions_has_all_models_and_nonnegative_values():
    result = generate_predictions(
        _demand(),
        {
            "seasonal_period": 4,
            "lag_steps": [1, 2, 4],
            "ridge_alpha": 1.0,
            "gradient_boosting_max_iter": 10,
            "gradient_boosting_max_depth": 2,
            "gradient_boosting_learning_rate": 0.1,
            "random_seed": 7,
        },
    )
    assert set(result["model"]) == {MODEL_HISTORICAL, MODEL_RIDGE_AR, MODEL_HGB}
    assert set(result["split"]) == {"validation", "test"}
    assert np.isfinite(result["predicted_count"]).all()
    assert (result["predicted_count"] >= 0).all()


def test_prediction_metrics_and_provider(tmp_path):
    result = generate_predictions(
        _demand(),
        {
            "seasonal_period": 4,
            "lag_steps": [1, 2, 4],
            "gradient_boosting_max_iter": 5,
            "random_seed": 7,
        },
    )
    metrics = prediction_metrics(result)
    assert {"mae", "rmse", "wape", "smape", "mape_nonzero"} <= set(metrics.columns)
    path = tmp_path / "predictions.csv"
    result.to_csv(path, index=False)
    provider = CSVPredictionProvider.from_csv(path, MODEL_HISTORICAL)
    prediction = provider.predict_slot("s16")
    assert set(prediction) == {0, 1}


def test_hgb_checkpoints_have_distinct_labels_and_metrics():
    predictions = generate_hgb_checkpoint_predictions(
        _demand(),
        {
            "seasonal_period": 4,
            "lag_steps": [1, 2, 4],
            "gradient_boosting_max_depth": 2,
            "gradient_boosting_learning_rate": 0.1,
            "random_seed": 7,
        },
        [1, 3],
    )
    assert set(predictions["model"]) == {"HGB_iter_001", "HGB_iter_003"}
    assert set(predictions["max_iter"]) == {1, 3}
    metrics = prediction_metrics(predictions)
    assert "max_iter" in metrics.columns
    assert len(metrics) == 4  # two checkpoints x validation/test


def test_direct_horizon_features_do_not_read_demand_after_forecast_origin():
    demand = _demand()
    original = build_direct_forecast_dataset(demand, horizon=4, origin_lags=[0, 1, 3], seasonal_period=4)
    changed = demand.copy()
    changed.loc[changed["slot_id"].isin(["s17", "s18", "s19", "s20", "s21", "s22", "s23"]), "observed_count"] += 1000
    mutated = build_direct_forecast_dataset(changed, horizon=4, origin_lags=[0, 1, 3], seasonal_period=4)

    original_rows = original.metadata["slot_id"].eq("s20")
    mutated_rows = mutated.metadata["slot_id"].eq("s20")
    pd.testing.assert_frame_equal(
        original.features.loc[original_rows].reset_index(drop=True),
        mutated.features.loc[mutated_rows].reset_index(drop=True),
    )
    assert original.metadata.loc[original_rows, "forecast_origin"].unique().tolist() == ["s16"]
    assert not np.array_equal(original.target[original_rows], mutated.target[mutated_rows])

    # The same boundary must also hold inside Train: no full-Train aggregate may
    # leak later training targets into an earlier training example.
    changed_train_future = demand.copy()
    changed_train_future.loc[
        changed_train_future["slot_id"].isin(["s9", "s10", "s11", "s12", "s13", "s14", "s15"]),
        "observed_count",
    ] += 500
    mutated_train = build_direct_forecast_dataset(
        changed_train_future, horizon=4, origin_lags=[0, 1, 3], seasonal_period=4
    )
    original_train_rows = original.metadata["slot_id"].eq("s12")
    mutated_train_rows = mutated_train.metadata["slot_id"].eq("s12")
    pd.testing.assert_frame_equal(
        original.features.loc[original_train_rows].reset_index(drop=True),
        mutated_train.features.loc[mutated_train_rows].reset_index(drop=True),
    )


def test_validation_selection_records_scaling_and_hgb_controls(tmp_path):
    result = train_horizon_predictors(
        _demand(),
        {
            "horizons": [1, 2],
            "origin_lags": [0, 1, 3],
            "seasonal_period": 4,
            "ridge_alpha_grid": [0.1, 1.0],
            "historical_fine_weight_grid": [0.0, 1.0],
            "hgb_max_iter_grid": [2, 4],
            "hgb_losses": ["squared_error", "poisson"],
            "hgb_max_depth": 2,
            "hgb_learning_rate": 0.1,
            "random_seed": 7,
        },
        model_dir=tmp_path,
    )
    assert {"model", "horizon", "forecast_origin", "slot_id", "type_id", "observed_count", "predicted_count", "split"} <= set(result.predictions.columns)
    assert set(result.predictions["model"]) == {MODEL_HISTORICAL, MODEL_RIDGE_AR, MODEL_HGB}
    assert result.selection.groupby(["model", "horizon"])["selected"].sum().eq(1).all()
    hgb_manifest = [row for row in result.training_manifest if row["model"] == MODEL_HGB]
    assert all(row["early_stopping"] is False for row in hgb_manifest)
    assert all(row["station_id_categorical"] is True for row in hgb_manifest)
    assert (tmp_path / "historical_average_h1.joblib").exists()
    assert (tmp_path / "ridge_ar_h1.joblib").exists()
    assert (tmp_path / "hgb_h2.joblib").exists()
