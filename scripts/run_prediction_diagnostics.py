#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import offline_opt
from rood_dasfaa2019.experiments import ChengduExperimentContext, run_chengdu_slot


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 decision-aware prediction diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    selected_theta = float(config["experiment"]["selected_theta"])
    models = sorted(context.predictions["model"].unique())
    slot_ids = (
        context.demand.loc[context.demand["split"].eq("test"), ["slot_id", "service_date", "slot_in_day"]]
        .drop_duplicates().sort_values(["service_date", "slot_in_day"])["slot_id"].head(args.slots).tolist()
    )
    rows = []
    for slot_id in slot_ids:
        orders = context.slot_orders(slot_id)
        opt_result = offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        for model in models:
            for seed in args.seeds:
                rows.extend(run_chengdu_slot(context, slot_id, model, seed=seed, thetas=(selected_theta,), opt_result=opt_result))
    raw = pd.DataFrame(rows)
    out = Path(config["data"]["output_dir"])
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw" / "prediction_dispatch_diagnostics.csv", index=False)
    dispatch = raw.groupby(["prediction_model", "method", "theta"], dropna=False).agg(
        runs=("slot_id", "size"), demand_error=("prediction_error", "mean"),
        advice_error=("advice_error", "mean"), performance=("alg_over_opt", "mean"),
        performance_std=("alg_over_opt", "std"),
    ).reset_index()
    correlations = raw.groupby(["prediction_model", "method"], dropna=False).apply(
        lambda frame: pd.Series({
            "demand_error_vs_performance_spearman": frame["prediction_error"].corr(frame["alg_over_opt"], method="spearman"),
            "advice_error_vs_performance_spearman": frame["advice_error"].corr(frame["alg_over_opt"], method="spearman"),
        }), include_groups=False,
    ).reset_index()
    dispatch.to_csv(out / "summary" / "prediction_dispatch_summary.csv", index=False)
    correlations.to_csv(out / "summary" / "prediction_dispatch_correlations.csv", index=False)
    print(dispatch.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
