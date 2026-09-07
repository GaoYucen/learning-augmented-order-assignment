#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import offline_opt
from rood_dasfaa2019.experiments import ChengduExperimentContext, run_chengdu_slot
from rood_dasfaa2019.learning.metrics import summarize_results


SCENARIOS = {
    "scale": [0.5, 0.75, 1.0, 1.25, 1.5],
    "permutation": [0.25, 0.5, 0.75, 1.0],
    "temporal_shift": [-4, -2, 0, 2, 4],
    "scarce_resource": [0.25, 0.5, 0.75, 1.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="B3 controlled distribution-shift experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--model", default="HGB")
    parser.add_argument("--split", choices=["test", "shifted_test"], default="test")
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    context = ChengduExperimentContext.load(config)
    output_dir = args.output_dir or Path(config["data"]["output_dir"])
    slot_ids = (
        context.demand.loc[context.demand["split"].eq(args.split), ["slot_id", "service_date", "slot_in_day"]]
        .drop_duplicates().sort_values(["service_date", "slot_in_day"])["slot_id"].head(args.slots).tolist()
    )
    seeds = args.seeds or config["experiment"]["seeds"]
    rows = []
    for slot_id in slot_ids:
        orders = context.slot_orders(slot_id)
        opt_result = offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        for corruption, levels in SCENARIOS.items():
            for level in levels:
                actual_corruption = "none" if (corruption == "scale" and level == 1.0) else corruption
                actual_level = 0.0 if actual_corruption == "none" else level
                for seed in seeds:
                    rows.extend(
                        run_chengdu_slot(
                            context, slot_id, args.model, actual_corruption, actual_level,
                            seed=int(seed), opt_result=opt_result,
                        )
                    )
    raw = pd.DataFrame(rows).drop_duplicates(
        ["seed", "slot_id", "method", "theta", "prediction_corruption", "corruption_strength"]
    )
    raw_dir = output_dir / "raw"
    summary_dir = output_dir / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / "distribution_shift_results.csv", index=False)
    summary = summarize_results(raw)
    summary.to_csv(summary_dir / "distribution_shift_summary.csv", index=False)
    diagnostics = (
        raw.groupby(["prediction_corruption", "corruption_strength", "method", "theta"], dropna=False)
        .agg(runs=("slot_id", "size"), prediction_error=("prediction_error", "mean"),
             advice_error=("advice_error", "mean"), performance=("alg_over_opt", "mean"),
             performance_std=("alg_over_opt", "std"), violations=("violations", "sum"))
        .reset_index()
    )
    diagnostics.to_csv(summary_dir / "distribution_shift_diagnostics.csv", index=False)
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
