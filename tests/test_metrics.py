import numpy as np

from src.metrics import calculate_metrics, haversine_distance


def test_identical_coordinates_have_zero_distance():
    distance = haversine_distance(
        true_lat=51.5074,
        true_lng=-0.1278,
        pred_lat=51.5074,
        pred_lng=-0.1278,
    )

    assert np.isclose(distance, 0.0)


def test_london_to_paris_distance():
    distance = haversine_distance(
        true_lat=51.5074,
        true_lng=-0.1278,
        pred_lat=48.8566,
        pred_lng=2.3522,
    )

    # London and Paris are approximately 344 km apart.
    assert 340 < distance < 350


def test_metrics_with_perfect_predictions():
    latitudes = [40.0, 50.0]
    longitudes = [10.0, 20.0]

    metrics = calculate_metrics(
        latitudes,
        longitudes,
        latitudes,
        longitudes,
    )

    assert np.isclose(metrics["median_distance_km"], 0.0)
    assert np.isclose(metrics["mean_distance_km"], 0.0)
    assert metrics["below_200_km"] == 100.0
    assert metrics["below_750_km"] == 100.0