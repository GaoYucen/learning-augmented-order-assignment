from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from rood_dasfaa2019.simulation.entities import Order

from .request_types import type_counts


class TypeDemandPredictor(Protocol):
    def predict(self, orders: list[Order], type_count: int) -> dict[int, float]: ...


class SlotTypeDemandProvider(Protocol):
    def predict_slot(self, slot_id: str) -> dict[int, float]: ...


class StaticPredictionProvider:
    """Fixed prediction provider for tests and adapter handoff.

    The expected shape is the same as the B-side predictor output:
    {request_type_id: predicted_count}.
    """

    def __init__(self, predicted_counts: dict[int, float]):
        self.predicted_counts = {int(key): max(float(value), 0.0) for key, value in predicted_counts.items()}

    def predict(self, orders: list[Order], type_count: int) -> dict[int, float]:
        return {request_type_id: self.predicted_counts.get(request_type_id, 0.0) for request_type_id in range(type_count)}


class SlotPredictionAdapter:
    """Bind a slot-level prediction provider to the runner predictor interface."""

    def __init__(self, provider: SlotTypeDemandProvider, slot_id: str):
        self.provider = provider
        self.slot_id = slot_id

    def predict(self, orders: list[Order], type_count: int) -> dict[int, float]:
        raw = self.provider.predict_slot(self.slot_id)
        return {request_type_id: max(float(raw.get(request_type_id, 0.0)), 0.0) for request_type_id in range(type_count)}


class PredictionFrameProvider:
    """Read B-side prediction rows and expose predict_slot(slot_id).

    Required columns: slot_id, type_id, predicted_count.
    Optional column: model. If model is set on the provider, rows are filtered
    to that prediction model.
    """

    def __init__(self, predictions: pd.DataFrame, model: str | None = None):
        required = {"slot_id", "type_id", "predicted_count"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
        if model is not None and "model" not in predictions.columns:
            raise ValueError("Prediction table must contain a model column when model is specified")
        self.predictions = predictions.copy()
        self.predictions["slot_id"] = self.predictions["slot_id"].astype(str)
        self.model = model

    @classmethod
    def from_csv(cls, path: Path | str, model: str | None = None) -> "PredictionFrameProvider":
        return cls(pd.read_csv(path), model=model)

    def predict_slot(self, slot_id: str) -> dict[int, float]:
        selected = self.predictions.loc[self.predictions["slot_id"] == str(slot_id)]
        if self.model is not None:
            selected = selected.loc[selected["model"] == self.model]
        if selected.empty:
            raise KeyError(f"No predictions for slot={slot_id!r}, model={self.model!r}")
        return {int(row.type_id): max(float(row.predicted_count), 0.0) for row in selected.itertuples()}


class SyntheticPredictionProvider:
    """Synthetic request-type demand predictor for controlled A1/A2 tests.

    It starts from realized demand and corrupts it in a controlled way. This is
    not the final predictor; it is a placeholder with the same output shape that
    a real predictor should later provide.
    """

    def __init__(self, cfg: dict):
        self.scale = float(cfg.get("prediction_scale", 1.0))
        self.corruption = str(cfg.get("prediction_corruption", "scale"))
        self.corruption_strength = float(cfg.get("corruption_strength", abs(self.scale - 1.0)))
        self.floor = float(cfg.get("prediction_floor", 0.0))

    def predict(self, orders: list[Order], type_count: int) -> dict[int, float]:
        truth = type_counts(orders, type_count)
        predicted = {request_type_id: truth[request_type_id] * self.scale for request_type_id in truth}
        if self.corruption == "permutation" and self.corruption_strength > 0:
            predicted = self._permute_type_counts(predicted)
        elif self.corruption == "adversarial_concentration" and self.corruption_strength > 0:
            predicted = self._concentrate_type_counts(predicted)
        return {request_type_id: max(self.floor, predicted[request_type_id]) for request_type_id in predicted}

    def _permute_type_counts(self, counts: dict[int, float]) -> dict[int, float]:
        request_types = sorted(counts)
        if len(request_types) < 2:
            return counts
        swap_count = min(len(request_types), max(0, int(round(len(request_types) * self.corruption_strength))))
        if swap_count < 2:
            return counts
        selected = request_types[:swap_count]
        rotated_values = [counts[selected[-1]], *[counts[request_type_id] for request_type_id in selected[:-1]]]
        corrupted = dict(counts)
        for request_type_id, value in zip(selected, rotated_values):
            corrupted[request_type_id] = value
        return corrupted

    def _concentrate_type_counts(self, counts: dict[int, float]) -> dict[int, float]:
        request_types = sorted(counts)
        if len(request_types) < 2:
            return counts
        corrupted = dict(counts)
        sorted_by_demand = sorted(request_types, key=lambda request_type_id: counts[request_type_id])
        source_count = max(1, int(round(len(request_types) * min(max(self.corruption_strength, 0.0), 1.0) / 2)))
        sources = sorted_by_demand[-source_count:]
        target = sorted_by_demand[0]
        moved = sum(corrupted[source] for source in sources)
        for source in sources:
            corrupted[source] = 0.0
        corrupted[target] += moved
        return corrupted
