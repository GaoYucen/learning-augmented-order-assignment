#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import offline_opt
from rood_dasfaa2019.experiments import ChengduExperimentContext, run_chengdu_slot


def evenly_spaced_slots(context: ChengduExperimentContext, split: str, count: int) -> list[str]:
    slots = (
        context.demand.loc[context.demand["split"].eq(split), ["slot_id", "service_date", "slot_in_day"]]
        .drop_duplicates().sort_values(["service_date", "slot_in_day"])["slot_id"].tolist()
    )
    if count >= len(slots):
        return slots
    positions = np.linspace(0, len(slots) - 1, count, dtype=int)
    return [slots[position] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch every prediction checkpoint through A's algorithms.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--slots", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    root = Path(config["data"]["output_dir"])
    checkpoints = pd.read_csv(root / "prediction" / "checkpoint_predictions.csv")
    context.predictions = checkpoints
    selected_theta = float(config["experiment"]["selected_theta"])
    slot_count = args.slots or int(config["prediction"]["checkpoint_test_slots"])
    slot_ids = evenly_spaced_slots(context, "test", slot_count)
    seeds = args.seeds or config["experiment"]["seeds"]
    model_iterations = (
        checkpoints[["model", "max_iter"]].drop_duplicates().sort_values("max_iter").itertuples(index=False)
    )
    model_iterations = [(str(row.model), int(row.max_iter)) for row in model_iterations]

    rows = []
    for slot_id in slot_ids:
        orders = context.slot_orders(slot_id)
        opt_result = offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        for model, max_iter in model_iterations:
            for seed in seeds:
                result_rows = run_chengdu_slot(
                    context, slot_id, model, thetas=(selected_theta,), seed=int(seed), opt_result=opt_result
                )
                for row in result_rows:
                    row["max_iter"] = max_iter
                rows.extend(result_rows)
    raw = pd.DataFrame(rows)
    raw_dir = root / "raw"
    summary_dir = root / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / "checkpoint_dispatch_results.csv", index=False)

    # Deterministic repetitions are averaged first; the confidence interval's
    # effective unit is the paired Test slot, not duplicated seed rows.
    paired = raw.groupby(
        ["prediction_model", "max_iter", "slot_id", "method", "theta"], dropna=False, as_index=False
    ).agg(
        alg_over_opt=("alg_over_opt", "mean"), prediction_error=("prediction_error", "mean"),
        advice_error=("advice_error", "mean"), accepted_value=("accepted_value", "mean"),
        high_value_rejected=("high_value_rejected", "mean"), violations=("violations", "sum"),
    )
    summary = paired.groupby(
        ["prediction_model", "max_iter", "method", "theta"], dropna=False
    ).agg(
        effective_slots=("slot_id", "size"), alg_over_opt_mean=("alg_over_opt", "mean"),
        alg_over_opt_std=("alg_over_opt", "std"), prediction_error_mean=("prediction_error", "mean"),
        advice_error_mean=("advice_error", "mean"), accepted_value_mean=("accepted_value", "mean"),
        high_value_rejected_mean=("high_value_rejected", "mean"), violations=("violations", "sum"),
    ).reset_index()
    summary["alg_over_opt_ci95"] = 1.96 * summary["alg_over_opt_std"] / np.sqrt(summary["effective_slots"])
    summary.to_csv(summary_dir / "checkpoint_dispatch_summary.csv", index=False)

    checkpoint_means = paired.groupby(
        ["prediction_model", "max_iter", "method"], as_index=False
    ).agg(
        prediction_error=("prediction_error", "mean"), advice_error=("advice_error", "mean"),
        alg_over_opt=("alg_over_opt", "mean"),
    )

    def safe_corr(frame: pd.DataFrame, x: str, method: str) -> float:
        if frame[x].nunique() < 2 or frame["alg_over_opt"].nunique() < 2:
            return np.nan
        return float(frame[x].corr(frame["alg_over_opt"], method=method))

    correlations = []
    for method, group in checkpoint_means.groupby("method"):
        correlations.append({
            "method": method,
            "correlation_unit": "checkpoint mean across paired Test slots",
            "prediction_error_vs_performance_pearson": safe_corr(group, "prediction_error", "pearson"),
            "prediction_error_vs_performance_spearman": safe_corr(group, "prediction_error", "spearman"),
            "advice_error_vs_performance_pearson": safe_corr(group, "advice_error", "pearson"),
            "advice_error_vs_performance_spearman": safe_corr(group, "advice_error", "spearman"),
        })
    pd.DataFrame(correlations).to_csv(summary_dir / "checkpoint_correlations.csv", index=False)
    print(summary.to_string(index=False))
    print(f"raw: {raw_dir / 'checkpoint_dispatch_results.csv'}")


if __name__ == "__main__":
    main()
