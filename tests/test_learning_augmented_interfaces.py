from rood_dasfaa2019.algorithms.common import count_constraint_violations
from rood_dasfaa2019.algorithms import ipd_dispatch, prediction_only_dispatch
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


def test_external_advice_respects_new_theta_endpoint_semantics():
    orders = [
        Order(0, 0, 1, 0.0, 1.0),
        Order(1, 1, 1, 1.0, 0.8),
    ]
    buses = [Bus(0, 2, 0.0, 30.0, [0, 1], {0: 1.0, 1: 2.0})]
    cfg = {"epsilon": 0.2}
    advice = build_predictive_lp_advice(
        orders, buses, 2, cfg,
        predicted_counts={0: 1.0, 1: 1.0},
        type_values={0: 1.0, 1: 0.8},
    )

    prediction_only = prediction_only_dispatch(orders, buses, 2, cfg, advice=advice)
    theta_zero = rp_laipd_dispatch(orders, buses, 2, {**cfg, "theta": 0.0}, advice=advice)
    ipd = ipd_dispatch(orders, buses, 2, cfg)
    theta_one = rp_laipd_dispatch(orders, buses, 2, {**cfg, "theta": 1.0}, advice=advice)

    assert theta_zero.accepted == prediction_only.accepted
    assert theta_zero.objective == prediction_only.objective
    assert theta_one.accepted == ipd.accepted
    assert theta_one.objective == ipd.objective
