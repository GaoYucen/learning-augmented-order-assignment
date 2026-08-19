#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.learning.real_prediction import generate_predictions, prediction_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Chengdu demand prediction baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_dir = Path(config["data"]["output_dir"])
    demand_path = output_dir / "processed" / "demand_by_slot_type.csv"
    prediction_dir = output_dir / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    demand = pd.read_csv(demand_path)
    prediction_cfg = dict(config.get("prediction", {}))
    prediction_cfg["random_seed"] = config["data"]["random_seed"]
    predictions = generate_predictions(demand, prediction_cfg)
    metrics = prediction_metrics(predictions)
    predictions.to_csv(prediction_dir / "predictions.csv", index=False)
    metrics.to_csv(prediction_dir / "prediction_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"predictions: {prediction_dir / 'predictions.csv'}")
    print(f"metrics: {prediction_dir / 'prediction_metrics.csv'}")


if __name__ == "__main__":
    main()
