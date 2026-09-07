#!/usr/bin/env python
"""Run the final leakage-safe forecast and theta-sensitivity experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from rood_dasfaa2019.algorithms import ipd_dispatch, offline_opt, prediction_only_dispatch, rp_laipd_dispatch
from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice
from rood_dasfaa2019.experiments import ChengduExperimentContext
from rood_dasfaa2019.learning.metrics import advice_error, dispatch_row, opt_type_bus_allocation
from rood_dasfaa2019.learning.real_prediction import prediction_metrics, train_horizon_predictors


MODEL_COLORS = {
    "HistoricalAverage": "#F58518",
    "RidgeAR": "#4C78A8",
    "HGB": "#54A24B",
}


def timed(function):
    started = perf_counter()
    result = function()
    return result, (perf_counter() - started) * 1000.0


def daily_block_mae_ci(group: pd.DataFrame, repetitions: int, seed: int) -> tuple[float, float]:
    """Bootstrap complete service days so within-day temporal dependence remains intact."""
    days = sorted(group["service_date"].astype(str).unique())
    if len(days) <= 1:
        value = float(np.mean(np.abs(group["predicted_count"] - group["observed_count"])))
        return value, value
    errors = {
        day: np.abs(
            group.loc[group["service_date"].astype(str).eq(day), "predicted_count"].to_numpy(dtype=float)
            - group.loc[group["service_date"].astype(str).eq(day), "observed_count"].to_numpy(dtype=float)
        )
        for day in days
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(days, size=len(days), replace=True)
        estimates[index] = float(np.mean(np.concatenate([errors[day] for day in sampled])))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def forecast_metric_table(predictions: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    metrics = prediction_metrics(predictions)
    ci_rows = []
    for keys, group in predictions.groupby(["split", "model", "horizon"], sort=True):
        lower, upper = daily_block_mae_ci(group, repetitions, seed)
        ci_rows.append(
            {
                "split": keys[0],
                "model": keys[1],
                "horizon": keys[2],
                "mae_ci95_lower": lower,
                "mae_ci95_upper": upper,
                "ci_resampling_unit": "service_date block",
                "bootstrap_repetitions": repetitions,
            }
        )
    return metrics.merge(pd.DataFrame(ci_rows), on=["split", "model", "horizon"], how="left")


def choose_representative_slot(predictions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    h1 = predictions.loc[
        predictions["split"].eq("test") & predictions["horizon"].eq(1)
    ]
    totals = (
        h1.loc[h1["model"].eq("HistoricalAverage")]
        .groupby(["slot_id", "service_date", "slot_in_day"], as_index=False)["observed_count"]
        .sum()
        .rename(columns={"observed_count": "total_observed_demand"})
    )
    median = float(totals["total_observed_demand"].median())
    totals["absolute_distance_from_median"] = (
        totals["total_observed_demand"] - median
    ).abs()
    totals = totals.sort_values(
        ["absolute_distance_from_median", "service_date", "slot_in_day", "slot_id"]
    ).reset_index(drop=True)
    selected_slot = str(totals.iloc[0]["slot_id"])
    totals["selected"] = totals["slot_id"].astype(str).eq(selected_slot)
    totals["selection_rule"] = (
        "minimum absolute distance to median total demand; ties resolved by chronological order"
    )
    return selected_slot, totals


def evenly_spaced_slots(predictions: pd.DataFrame, count: int) -> list[str]:
    slots = (
        predictions.loc[
            predictions["split"].eq("test") & predictions["horizon"].eq(1),
            ["slot_id", "service_date", "slot_in_day"],
        ]
        .drop_duplicates()
        .sort_values(["service_date", "slot_in_day"])
    )
    values = slots["slot_id"].astype(str).tolist()
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[position] for position in positions]


def proven_optimal(result, tolerance: float) -> bool:
    if result.solver_status == "no_pairs":
        return True
    gap = result.solver_mip_gap
    return bool(
        result.solver_success
        and str(result.solver_status) == "0"
        and not result.solver_time_limit_hit
        and (gap is None or (np.isfinite(gap) and gap <= tolerance))
    )


def hierarchical_theta_ci(group: pd.DataFrame, repetitions: int, seed: int) -> tuple[float, float]:
    """Bootstrap perturbation seeds and whole service days as paired clusters."""
    block = group.groupby(["seed", "service_date"], as_index=False)["alg_over_opt"].mean()
    seeds = sorted(block["seed"].unique())
    days = sorted(block["service_date"].astype(str).unique())
    matrix = block.pivot(index="seed", columns="service_date", values="alg_over_opt").reindex(
        index=seeds, columns=days
    ).to_numpy(dtype=float)
    if len(seeds) == 1 and len(days) == 1:
        return float(matrix[0, 0]), float(matrix[0, 0])
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
        sampled_days = rng.integers(0, len(days), size=len(days))
        estimates[index] = float(np.nanmean(matrix[np.ix_(sampled_seeds, sampled_days)]))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def theta_summary_table(raw: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rp = raw.loc[raw["method"].eq("RP-LAIPD")].copy()
    rows = []
    for (lambda_value, theta), group in rp.groupby(["lambda", "theta"], sort=True):
        # lambda=0 is deterministic across perturbation seeds; retain one seed
        # for uncertainty so duplicated deterministic rows are not pseudo-replication.
        ci_group = group
        effective_seed_count = int(group["seed"].nunique())
        if np.isclose(lambda_value, 0.0):
            first_seed = int(group["seed"].min())
            ci_group = group.loc[group["seed"].eq(first_seed)]
            effective_seed_count = 1
        lower, upper = hierarchical_theta_ci(
            ci_group, repetitions, seed + int(round(lambda_value * 1000)) + int(round(theta * 100))
        )
        rows.append(
            {
                "lambda": float(lambda_value),
                "theta": float(theta),
                "actual_mae": float(group["actual_mae"].mean()),
                "actual_mae_std_across_seeds": float(group.groupby("seed")["actual_mae"].first().std(ddof=0)),
                "alg_over_opt": float(group["alg_over_opt"].mean()),
                "alg_over_opt_ci95_lower": lower,
                "alg_over_opt_ci95_upper": upper,
                "effective_perturbation_seeds": effective_seed_count,
                "effective_service_days": int(group["service_date"].nunique()),
                "effective_slots": int(group["slot_id"].nunique()),
                "ci_resampling_unit": "paired perturbation seed x service-date block",
            }
        )
    return pd.DataFrame(rows)


def run_theta_error_experiment(
    context: ChengduExperimentContext,
    predictions: pd.DataFrame,
    selected_model: str,
    config: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    error_cfg = config["theta_error_experiment"]
    horizon = int(error_cfg["predictor_horizon"])
    lambdas = [float(value) for value in error_cfg["lambdas"]]
    seeds = [int(value) for value in error_cfg["perturbation_seeds"]]
    thetas = [float(value) for value in error_cfg["thetas"]]
    slot_ids = evenly_spaced_slots(predictions, int(error_cfg["test_slot_count"]))
    tolerance = float(error_cfg.get("opt_gap_tolerance", 1e-9))

    opt_results = {}
    opt_rows = []
    for slot_id in slot_ids:
        orders = context.slot_orders(slot_id)
        result, elapsed_ms = timed(
            lambda: offline_opt(orders, context.buses, len(context.stations), context.algorithm_cfg)
        )
        exact = proven_optimal(result, tolerance)
        service_date = str(context.demand.loc[context.demand["slot_id"].eq(slot_id), "service_date"].iloc[0])
        opt_rows.append(
            {
                "slot_id": slot_id,
                "service_date": service_date,
                "orders": len(orders),
                "objective": result.objective,
                "solver_status": result.solver_status,
                "solver_success": result.solver_success,
                "time_limit_hit": result.solver_time_limit_hit,
                "mip_gap": result.solver_mip_gap,
                "dual_bound": result.solver_dual_bound,
                "solver_message": result.solver_message,
                "runtime_ms": elapsed_ms,
                "proven_optimal": exact,
            }
        )
        if exact:
            opt_results[slot_id] = result
    opt_status = pd.DataFrame(opt_rows)
    valid_slots = [slot for slot in slot_ids if slot in opt_results]
    if not valid_slots:
        raise RuntimeError("No dispatch slot has a proven-optimal offline solution")

    base = predictions.loc[
        predictions["model"].eq(selected_model)
        & predictions["horizon"].eq(horizon)
        & predictions["split"].eq("test")
    ].sort_values(["slot_id", "type_id"]).reset_index(drop=True)
    train_scale = (
        context.demand.loc[context.demand["split"].eq("train")]
        .groupby("type_id")["observed_count"]
        .std(ddof=0)
        .reindex(range(len(context.stations)), fill_value=0.0)
        .clip(lower=1.0)
    )

    dispatch_rows = []
    perturbation_rows = []
    endpoint_rows = []
    total_capacity = sum(bus.capacity for bus in context.buses)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        noise_direction = rng.standard_normal(len(base))
        for lambda_value in lambdas:
            perturbed = base.copy()
            perturbed["station_scale"] = perturbed["type_id"].map(train_scale).astype(float)
            perturbed["noise_z"] = noise_direction
            perturbed["lambda"] = lambda_value
            perturbed["seed"] = seed
            perturbed["base_predicted_count"] = perturbed["predicted_count"]
            perturbed["predicted_count"] = np.maximum(
                0.0,
                perturbed["base_predicted_count"]
                + lambda_value * perturbed["station_scale"] * perturbed["noise_z"],
            )
            actual_mae = float(np.mean(np.abs(perturbed["predicted_count"] - perturbed["observed_count"])))
            perturbed["actual_mae"] = actual_mae
            perturbation_rows.append(perturbed)

            for slot_id in valid_slots:
                slot_prediction = perturbed.loc[perturbed["slot_id"].eq(slot_id)]
                predicted_counts = {
                    int(row.type_id): float(row.predicted_count)
                    for row in slot_prediction.itertuples()
                }
                orders = context.slot_orders(slot_id)
                service_date = str(slot_prediction["service_date"].iloc[0])
                slot_mae = float(np.mean(np.abs(slot_prediction["predicted_count"] - slot_prediction["observed_count"])))
                slot_wape = float(
                    np.abs(slot_prediction["predicted_count"] - slot_prediction["observed_count"]).sum()
                    / max(float(slot_prediction["observed_count"].sum()), 1.0)
                )
                advice, lp_ms = timed(
                    lambda: build_predictive_lp_advice(
                        orders,
                        context.buses,
                        len(context.stations),
                        context.algorithm_cfg,
                        predicted_counts=predicted_counts,
                        type_values=context.type_values,
                    )
                )
                opt_result = opt_results[slot_id]
                opt_allocation = opt_type_bus_allocation(orders, opt_result)
                current_advice_error = advice_error(advice, opt_allocation)
                prediction_result, prediction_runtime = timed(
                    lambda: prediction_only_dispatch(
                        orders, context.buses, len(context.stations), context.algorithm_cfg, advice=advice
                    )
                )
                ipd_result, ipd_runtime = timed(
                    lambda: ipd_dispatch(orders, context.buses, len(context.stations), context.algorithm_cfg)
                )

                method_results = [
                    ("Prediction-only", 0.0, prediction_result, prediction_runtime),
                    ("IPD", 1.0, ipd_result, ipd_runtime),
                ]
                rp_by_theta = {}
                for theta in thetas:
                    theta_cfg = {**context.algorithm_cfg, "theta": theta}
                    result, runtime_ms = timed(
                        lambda cfg=theta_cfg: rp_laipd_dispatch(
                            orders, context.buses, len(context.stations), cfg, advice=advice
                        )
                    )
                    rp_by_theta[theta] = result
                    method_results.append(("RP-LAIPD", theta, result, runtime_ms))

                endpoint_rows.append(
                    {
                        "slot_id": slot_id,
                        "seed": seed,
                        "lambda": lambda_value,
                        "theta_0_assignment_matches_prediction_only": rp_by_theta[0.0].accepted == prediction_result.accepted,
                        "theta_0_value_matches_prediction_only": np.isclose(rp_by_theta[0.0].objective, prediction_result.objective),
                        "theta_1_assignment_matches_ipd": rp_by_theta[1.0].accepted == ipd_result.accepted,
                        "theta_1_value_matches_ipd": np.isclose(rp_by_theta[1.0].objective, ipd_result.objective),
                    }
                )
                for method, theta, result, runtime_ms in method_results:
                    violations = count_constraint_violations(
                        result.accepted, orders, context.buses, len(context.stations), context.algorithm_cfg
                    )
                    row = dispatch_row(
                        slot_id,
                        method,
                        theta,
                        1.0,
                        "station_scaled_gaussian",
                        lambda_value,
                        slot_wape,
                        current_advice_error,
                        orders,
                        result,
                        opt_result,
                        runtime_ms,
                        total_capacity=total_capacity,
                        violations=violations,
                        seed=seed,
                        split="test",
                        prediction_model=selected_model,
                        predictive_lp_time_ms=lp_ms,
                    )
                    row.update(
                        {
                            "lambda": lambda_value,
                            "actual_mae": actual_mae,
                            "slot_mae": slot_mae,
                            "service_date": service_date,
                            "sampled_demand_label": "sampled experimental demand; max 300 orders/slot; one passenger/order",
                        }
                    )
                    dispatch_rows.append(row)

    dispatch = pd.DataFrame(dispatch_rows)
    perturbations = pd.concat(perturbation_rows, ignore_index=True)
    endpoints = pd.DataFrame(endpoint_rows)
    endpoint_columns = [column for column in endpoints if "matches" in column]
    if not endpoints[endpoint_columns].all().all():
        raise AssertionError("Theta endpoint allocation/value equivalence failed")
    if int(dispatch["violations"].sum()) != 0:
        raise AssertionError("Resource-constraint violations detected")
    return dispatch, perturbations, endpoints, opt_status


def dependency_table() -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML", "joblib"]
    rows = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"package": package, "version": version})
    rows.append({"package": "python", "version": sys.version.split()[0]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chengdu.yaml"))
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["data"]["output_dir"])
    output_dir = root / "final_experiment"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    demand = pd.read_csv(root / "processed" / "demand_by_slot_type.csv")
    forecast_cfg = {**config["prediction"], "random_seed": config["data"]["random_seed"]}
    forecast = train_horizon_predictors(demand, forecast_cfg, model_dir=model_dir)

    predictions = forecast.predictions
    predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    forecast.historical_original_predictions.to_csv(
        output_dir / "historical_original_predictions.csv", index=False
    )
    repetitions = int(forecast_cfg["bootstrap_repetitions"])
    metrics = forecast_metric_table(predictions, repetitions, int(config["data"]["random_seed"]))
    original_metrics = forecast_metric_table(
        forecast.historical_original_predictions,
        repetitions,
        int(config["data"]["random_seed"]) + 1,
    )
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    original_metrics.to_csv(output_dir / "historical_original_metrics.csv", index=False)
    forecast.selection.to_csv(output_dir / "validation_model_selection.csv", index=False)
    pd.DataFrame(forecast.training_manifest).to_csv(output_dir / "training_manifest.csv", index=False)
    feature_definition = {
        "target": "one destination-station demand count in one 30-minute target slot",
        "forecast_origin": "target slot time index minus horizon",
        "demand_features": [
            column
            for column in forecast.training_manifest[0]["feature_definition"]
            if column.startswith("origin_demand_lag_")
        ],
        "demand_feature_boundary": (
            "origin_demand_lag_L reads the completed slot at forecast_origin-L; "
            "no demand after forecast_origin is used"
        ),
        "known_target_calendar_features": [
            column
            for column in forecast.training_manifest[0]["feature_definition"]
            if column.startswith("target_")
        ],
        "station_feature": "station_id; one-hot for RidgeAR and categorical for HGB",
        "fit_boundary": "model and scaler fit on Train targets only; choices use Validation only",
    }
    with (output_dir / "feature_definition.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_definition, handle, ensure_ascii=False, indent=2)
    pd.DataFrame({"slot_id": forecast.common_test_slots}).to_csv(
        output_dir / "common_test_slots.csv", index=False
    )

    selected_slot, representative_selection = choose_representative_slot(predictions)
    representative_selection.to_csv(output_dir / "representative_slot_selection.csv", index=False)
    representative_source = predictions.loc[
        predictions["split"].eq("test")
        & predictions["horizon"].eq(1)
        & predictions["slot_id"].eq(selected_slot)
    ].copy()
    representative_source.to_csv(output_dir / "representative_station_predictions.csv", index=False)

    validation_h1 = metrics.loc[
        metrics["split"].eq("validation") & metrics["horizon"].eq(1)
    ].sort_values(["mae", "model"])
    selected_predictor = str(validation_h1.iloc[0]["model"])
    validation_h1.assign(selected=lambda frame: frame["model"].eq(selected_predictor)).to_csv(
        output_dir / "theta_predictor_selection.csv", index=False
    )

    context = ChengduExperimentContext.load(config)
    context.predictions = predictions.loc[predictions["horizon"].eq(1)].copy()
    dispatch, perturbations, endpoints, opt_status = run_theta_error_experiment(
        context, predictions, selected_predictor, config, output_dir
    )
    dispatch.to_csv(output_dir / "theta_dispatch_results.csv", index=False)
    perturbations.to_csv(output_dir / "perturbed_predictions.csv", index=False)
    endpoints.to_csv(output_dir / "theta_endpoint_validation.csv", index=False)
    opt_status.to_csv(output_dir / "opt_status.csv", index=False)
    theta_summary = theta_summary_table(
        dispatch, repetitions, int(config["data"]["random_seed"]) + 100
    )
    theta_summary.to_csv(output_dir / "theta_error_summary.csv", index=False)

    noise_settings = pd.DataFrame(
        [
            {
                "lambda": lambda_value,
                "seed": seed,
                "noise_definition": "max(0, predicted_count + lambda * max(train_station_std, 1) * z)",
                "shared_direction_rule": "same seed reuses identical z across all lambda and theta values",
            }
            for seed in config["theta_error_experiment"]["perturbation_seeds"]
            for lambda_value in config["theta_error_experiment"]["lambdas"]
        ]
    )
    noise_settings.to_csv(output_dir / "perturbation_settings.csv", index=False)
    dependency_table().to_csv(output_dir / "dependency_versions.csv", index=False)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    manifest = {
        "demand_definition": "sampled experimental demand; max 300 orders per 30-minute slot; one passenger per order",
        "prediction_target": "station demand in one target 30-minute service slot",
        "common_test_target_slots": len(forecast.common_test_slots),
        "dispatch_test_slots_requested": int(config["theta_error_experiment"]["test_slot_count"]),
        "dispatch_slots_with_proven_opt": int(opt_status["proven_optimal"].sum()),
        "selected_theta_predictor": selected_predictor,
        "selected_representative_slot": selected_slot,
        "theta_definition": "robust IPD resource fraction; theta=0 Prediction-only; theta=1 IPD",
        "confidence_interval": "service-date block bootstrap; perturbation experiment also resamples paired seeds",
        "formal_claim_scope": "empirical sensitivity under configured Gaussian perturbations; not a theoretical worst-case guarantee",
    }
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if not args.skip_plots:
        subprocess.run(
            [sys.executable, "scripts/plot_final_prediction_figures.py", "--input-dir", str(output_dir)],
            check=True,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
