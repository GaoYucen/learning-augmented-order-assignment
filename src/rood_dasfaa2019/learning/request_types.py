from __future__ import annotations

from rood_dasfaa2019.simulation.entities import Order


def request_type(order: Order) -> int:
    """Map an order to the current request type.

    The DASFAA airport benchmark naturally groups requests by destination
    station. If later experiments need origin-destination or time-aware types,
    this function is the single place to extend the definition.
    """
    return order.destination


def type_counts(orders: list[Order], type_count: int) -> dict[int, float]:
    counts = {request_type_id: 0.0 for request_type_id in range(type_count)}
    for order in orders:
        counts[request_type(order)] += order.passengers
    return counts

