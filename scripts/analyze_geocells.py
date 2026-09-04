import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.metrics import calculate_metrics


TRAIN_CSV = "data/splits/train.csv"
VALIDATION_CSV = "data/splits/validation.csv"

CELL_COUNTS = [50, 100, 200, 300, 500]
SEED = 42


def coordinates_to_xyz(latitudes, longitudes):
    """Convert latitude/longitude to points on a unit sphere."""

    latitudes = np.radians(
        np.asarray(latitudes, dtype=float)
    )
    longitudes = np.radians(
        np.asarray(longitudes, dtype=float)
    )

    x = np.cos(latitudes) * np.cos(longitudes)
    y = np.cos(latitudes) * np.sin(longitudes)
    z = np.sin(latitudes)

    return np.column_stack((x, y, z))


def xyz_to_coordinates(points):
    """Convert unit-sphere points back to latitude/longitude."""

    points = np.asarray(points, dtype=float)

    norms = np.linalg.norm(
        points,
        axis=1,
        keepdims=True,
    )

    points = points / np.clip(
        norms,
        1e-12,
        None,
    )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    latitudes = np.degrees(
        np.arctan2(z, np.sqrt(x**2 + y**2))
    )

    longitudes = np.degrees(
        np.arctan2(y, x)
    )

    return np.column_stack(
        (latitudes, longitudes)
    )


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    validation_df = pd.read_csv(VALIDATION_CSV)

    train_xyz = coordinates_to_xyz(
        train_df["lat"],
        train_df["lng"],
    )

    validation_xyz = coordinates_to_xyz(
        validation_df["lat"],
        validation_df["lng"],
    )

    print("Geocell oracle analysis")
    print("-----------------------")
    print(f"Training examples: {len(train_df)}")
    print(
        f"Validation examples: "
        f"{len(validation_df)}"
    )

    results = []

    for number_of_cells in CELL_COUNTS:
        print(
            f"\nTesting {number_of_cells} cells..."
        )

        clustering = KMeans(
            n_clusters=number_of_cells,
            random_state=SEED,
            n_init=10,
        )

        clustering.fit(train_xyz)

        validation_cells = clustering.predict(
            validation_xyz
        )

        cell_centres = xyz_to_coordinates(
            clustering.cluster_centers_
        )

        predicted_coordinates = cell_centres[
            validation_cells
        ]

        metrics = calculate_metrics(
            true_lat=validation_df["lat"],
            true_lng=validation_df["lng"],
            pred_lat=predicted_coordinates[:, 0],
            pred_lng=predicted_coordinates[:, 1],
        )

        smallest_cell = np.bincount(
            clustering.labels_,
            minlength=number_of_cells,
        ).min()

        largest_cell = np.bincount(
            clustering.labels_,
            minlength=number_of_cells,
        ).max()

        results.append(
            {
                "cells": number_of_cells,
                "oracle_median_km": (
                    metrics["median_distance_km"]
                ),
                "oracle_mean_km": (
                    metrics["mean_distance_km"]
                ),
                "below_200_percent": (
                    metrics["below_200_km"]
                ),
                "smallest_training_cell": (
                    smallest_cell
                ),
                "largest_training_cell": (
                    largest_cell
                ),
            }
        )

        print(
            "Oracle median: "
            f"{metrics['median_distance_km']:.2f} km"
        )
        print(
            "Oracle mean: "
            f"{metrics['mean_distance_km']:.2f} km"
        )
        print(
            "Below 200 km: "
            f"{metrics['below_200_km']:.2f}%"
        )
        print(
            "Training images per cell: "
            f"{smallest_cell} to {largest_cell}"
        )

    results_df = pd.DataFrame(results)

    print("\nSummary")
    print("-------")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


if __name__ == "__main__":
    main()