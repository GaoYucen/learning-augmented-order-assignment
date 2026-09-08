#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="B4 ablation tables from paired B3 runs.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["data"]["output_dir"])
    raw = pd.read_csv(root / "raw" / "distribution_shift_results.csv")
    keep = raw[raw["method"].isin(["IPD", "Prediction-only", "RP-LAIPD"])].copy()
    best_theta = float(config["experiment"]["selected_theta"])
    keep = keep[(~keep["method"].eq("RP-LAIPD")) | keep["theta"].eq(best_theta)]
    keep["ablation_component"] = keep["method"].map({
        "RP-LAIPD": "full_model", "Prediction-only": "without_robust_branch", "IPD": "without_prediction_advice"
    })
    keep.to_csv(root / "raw" / "ablation_results.csv", index=False)
    summary = keep.groupby(
        ["prediction_corruption", "corruption_strength", "ablation_component", "method"], dropna=False
    ).agg(
        runs=("slot_id", "size"), mean=("alg_over_opt", "mean"), std=("alg_over_opt", "std"),
        high_value_rejected=("high_value_rejected", "mean"), violations=("violations", "sum"),
    ).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / summary["runs"].pow(0.5)
    summary["selected_theta"] = best_theta
    summary.to_csv(root / "summary" / "ablation_summary.csv", index=False)
    print(f"Pre-specified B0 theta={best_theta:g}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
