from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.algorithms.laipd import build_predictive_lp_advice, rp_laipd_dispatch
from rood_dasfaa2019.simulation.entities import Bus, Order


def test_external_prediction_and_precomputed_advice_are_accepted():
    orders = [
        Order(0, 0, 1, 0.0, 1.0),
        Order(1, 1, 1, 1.0, 0.8),
    ]
    buses = [Bus(0, 2, 0.0, 30.0, [0, 1], {0: 1.0, 1: 2.0})]
    cfg = {"theta": 0.4, "epsilon": 0.2}
    advice = build_predictive_lp_advice(
        orders, buses, 2, cfg, predicted_counts={0: 1.0, 1: 1.0}, type_values={0: 1.0, 1: 0.8}
    )
    assert advice[(0, 0)] >= 0
    result = rp_laipd_dispatch(orders, buses, 2, cfg, advice=advice)
    assert len(result.request_latencies_ms) == len(orders)
    assert count_constraint_violations(result.accepted, orders, buses, 2, cfg) == 0
