#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.learning.real_prediction import (
    generate_hgb_checkpoint_predictions,
    prediction_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HGB max_iter checkpoints on Chengdu Train data.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--iterations", type=int, nargs="+")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["data"]["output_dir"])
    demand = pd.read_csv(root / "processed" / "demand_by_slot_type.csv")
    prediction_cfg = {**config["prediction"], "random_seed": config["data"]["random_seed"]}
    iterations = args.iterations or prediction_cfg["checkpoint_iterations"]
    predictions = generate_hgb_checkpoint_predictions(demand, prediction_cfg, iterations)
    metrics = prediction_metrics(predictions)
    prediction_dir = root / "prediction"
    summary_dir = root / "summary"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_dir / "checkpoint_predictions.csv", index=False)
    metrics.to_csv(prediction_dir / "checkpoint_prediction_metrics.csv", index=False)

    validation = metrics[metrics["split"].eq("validation")].sort_values(["wape", "max_iter"]).copy()
    validation["validation_accuracy_rank"] = range(1, len(validation) + 1)
    validation["selection_basis"] = "Validation WAPE only; Test is not used for checkpoint selection"
    validation.to_csv(summary_dir / "checkpoint_selection.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"predictions: {prediction_dir / 'checkpoint_predictions.csv'}")


if __name__ == "__main__":
    main()
