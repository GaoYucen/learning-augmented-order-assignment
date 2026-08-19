from __future__ import annotations

import numpy as np
import pandas as pd

from rood_dasfaa2019.learning.real_prediction import (
    MODEL_HGB,
    MODEL_HISTORICAL,
    MODEL_RIDGE_AR,
    CSVPredictionProvider,
    generate_hgb_checkpoint_predictions,
    generate_predictions,
    prediction_metrics,
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
