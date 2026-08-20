from __future__ import annotations

from typing import Protocol

from rood_dasfaa2019.simulation.entities import Order

from .request_types import type_counts


class TypeDemandPredictor(Protocol):
    def predict(self, orders: list[Order], type_count: int) -> dict[int, float]: ...


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
