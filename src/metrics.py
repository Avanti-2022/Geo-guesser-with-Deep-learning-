import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_distance(
    true_lat,
    true_lng,
    pred_lat,
    pred_lng,
):
    """Calculate geographic distance in kilometres."""

    true_lat = np.radians(np.asarray(true_lat, dtype=float))
    true_lng = np.radians(np.asarray(true_lng, dtype=float))
    pred_lat = np.radians(np.asarray(pred_lat, dtype=float))
    pred_lng = np.radians(np.asarray(pred_lng, dtype=float))

    lat_difference = pred_lat - true_lat
    lng_difference = pred_lng - true_lng

    a = (
        np.sin(lat_difference / 2) ** 2
        + np.cos(true_lat)
        * np.cos(pred_lat)
        * np.sin(lng_difference / 2) ** 2
    )

    # Protect against tiny floating-point errors.
    a = np.clip(a, 0.0, 1.0)

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def calculate_metrics(true_lat, true_lng, pred_lat, pred_lng):
    """Calculate all assignment evaluation metrics."""

    distances = haversine_distance(
        true_lat,
        true_lng,
        pred_lat,
        pred_lng,
    )

    return {
        "median_distance_km": float(np.median(distances)),
        "mean_distance_km": float(np.mean(distances)),
        "below_200_km": float(np.mean(distances < 200) * 100),
        "below_750_km": float(np.mean(distances < 750) * 100),
    }