from rood_dasfaa2019.learning.metrics import prediction_error
from rood_dasfaa2019.learning.prediction import SyntheticPredictionProvider
from rood_dasfaa2019.learning.request_types import request_type, type_counts
from rood_dasfaa2019.learning.runner import run_learning_augmented_slot
from rood_dasfaa2019.simulation.entities import Order


def test_request_type_uses_destination_station():
    order = Order(id=1, destination=3, passengers=2, arrival_time=10.0, priority=0.5)

    assert request_type(order) == 3


def test_synthetic_prediction_scale_controls_type_counts():
    orders = [
        Order(id=1, destination=0, passengers=2, arrival_time=1.0, priority=0.2),
        Order(id=2, destination=1, passengers=3, arrival_time=2.0, priority=0.4),
    ]

    truth = type_counts(orders, 3)
    predicted = SyntheticPredictionProvider({"prediction_scale": 0.5}).predict(orders, 3)

    assert truth == {0: 2.0, 1: 3.0, 2: 0.0}
    assert predicted == {0: 1.0, 1: 1.5, 2: 0.0}
    assert prediction_error(orders, 3, predicted) == 0.5


def test_learning_augmented_runner_returns_standard_rows():
    cfg = dict(
        num_stations=5,
        horizon_minutes=60,
        bus_wait_minutes=15,
        epsilon=0.2,
        max_passengers_per_order=2,
        station_fairness_scale=1,
        bus_time_scale=1,
        station_time_scale=1,
        num_buses=5,
        bus_capacity=8,
        num_orders=20,
    )

    rows = run_learning_augmented_slot(
        cfg=cfg,
        seed=7,
        prediction_scales=[1.0],
        corruption_strengths=[0.0],
        corruption="scale",
        thetas=[0.5],
        slot_id=0,
    )
    methods = {row["method"] for row in rows}

    assert {"Random", "Greedy", "IPD", "Prediction-only", "RP-LAIPD"} <= methods
    assert all("prediction_error" in row for row in rows)
    assert all("advice_error" in row for row in rows)
    assert all("alg_over_opt" in row for row in rows)
