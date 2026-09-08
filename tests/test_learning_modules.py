import pandas as pd

from rood_dasfaa2019.learning.metrics import prediction_error
from rood_dasfaa2019.learning.prediction import (
    PredictionFrameProvider,
    SlotPredictionAdapter,
    StaticPredictionProvider,
    SyntheticPredictionProvider,
)
from rood_dasfaa2019.learning.request_types import request_type, type_counts
from rood_dasfaa2019.learning.runner import run_learning_augmented_instance, run_learning_augmented_slot
from rood_dasfaa2019.simulation.entities import Bus, Order, Station
from rood_dasfaa2019.algorithms import static_bid_price_dispatch


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


def test_prediction_frame_provider_matches_b_side_prediction_shape():
    predictions = pd.DataFrame(
        [
            {"slot_id": "slot-a", "type_id": 0, "model": "BModel", "predicted_count": 4.0},
            {"slot_id": "slot-a", "type_id": 1, "model": "BModel", "predicted_count": 6.0},
            {"slot_id": "slot-b", "type_id": 0, "model": "BModel", "predicted_count": 9.0},
        ]
    )
    provider = PredictionFrameProvider(predictions, model="BModel")
    adapter = SlotPredictionAdapter(provider, "slot-a")

    assert provider.predict_slot("slot-a") == {0: 4.0, 1: 6.0}
    assert adapter.predict([], 3) == {0: 4.0, 1: 6.0, 2: 0.0}


def test_learning_augmented_runner_accepts_external_predictor():
    cfg = dict(
        num_stations=2,
        horizon_minutes=60,
        bus_wait_minutes=15,
        epsilon=0.2,
        station_fairness_scale=1,
        bus_time_scale=1,
        station_time_scale=1,
    )
    stations = [Station(0, 0.0, 0.0), Station(1, 1.0, 0.0)]
    buses = [
        Bus(0, capacity=3, available_from=0.0, depart_at=60.0, route=[0, 1], station_travel_time={0: 1.0, 1: 2.0}),
        Bus(1, capacity=3, available_from=0.0, depart_at=60.0, route=[0, 1], station_travel_time={0: 1.5, 1: 1.0}),
    ]
    orders = [
        Order(0, destination=0, passengers=1, arrival_time=1.0, priority=0.5),
        Order(1, destination=1, passengers=1, arrival_time=2.0, priority=0.8),
        Order(2, destination=1, passengers=1, arrival_time=3.0, priority=0.7),
    ]
    predictor = StaticPredictionProvider({0: 1.0, 1: 2.0})

    rows = run_learning_augmented_instance(
        stations=stations,
        orders=orders,
        buses=buses,
        cfg=cfg,
        seed=11,
        prediction_scales=[1.0],
        corruption_strengths=[0.0],
        corruption="scale",
        thetas=[0.5],
        slot_id=0,
        predictor=predictor,
    )

    assert {row["method"] for row in rows} >= {"Prediction-only", "RP-LAIPD", "IPD"}
    assert all(row["prediction_scale"] == 1.0 for row in rows)


def test_static_bid_price_dispatch_runs_with_prediction():
    cfg = dict(
        num_stations=2,
        horizon_minutes=60,
        bus_wait_minutes=15,
        epsilon=0.2,
        station_fairness_scale=1,
        bus_time_scale=1,
        station_time_scale=1,
    )
    stations = [Station(0, 0.0, 0.0), Station(1, 1.0, 0.0)]
    buses = [
        Bus(0, capacity=3, available_from=0.0, depart_at=60.0, route=[0, 1], station_travel_time={0: 1.0, 1: 2.0}),
        Bus(1, capacity=3, available_from=0.0, depart_at=60.0, route=[0, 1], station_travel_time={0: 1.5, 1: 1.0}),
    ]
    orders = [
        Order(0, destination=0, passengers=1, arrival_time=1.0, priority=0.5),
        Order(1, destination=1, passengers=1, arrival_time=2.0, priority=0.8),
        Order(2, destination=1, passengers=1, arrival_time=3.0, priority=0.7),
    ]

    result = static_bid_price_dispatch(
        orders,
        buses,
        2,
        cfg,
        predicted_counts={0: 1.0, 1: 2.0},
        type_values={0: 1.0, 1: 0.8},
        avg_passengers={0: 1.0, 1: 1.0},
    )

    assert result.objective >= 0.0
    assert len(result.request_latencies_ms) == len(orders)
