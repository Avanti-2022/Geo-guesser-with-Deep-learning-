from pathlib import Path

import pandas as pd

from src.metrics import calculate_metrics


TRAIN_SPLIT = Path("data/splits/train.csv")
VALIDATION_SPLIT = Path("data/splits/validation.csv")


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(
        f"Median distance: {metrics['median_distance_km']:.2f} km"
    )
    print(
        f"Mean distance:   {metrics['mean_distance_km']:.2f} km"
    )
    print(
        f"Below 200 km:    {metrics['below_200_km']:.2f}%"
    )
    print(
        f"Below 750 km:    {metrics['below_750_km']:.2f}%"
    )


def main():
    train_df = pd.read_csv(TRAIN_SPLIT)
    validation_df = pd.read_csv(VALIDATION_SPLIT)

    # Learn one central coordinate from training data only.
    median_latitude = train_df["lat"].median()
    median_longitude = train_df["lng"].median()

    print("Global median prediction")
    print("------------------------")
    print(f"Predicted latitude:  {median_latitude:.6f}")
    print(f"Predicted longitude: {median_longitude:.6f}")

    median_metrics = calculate_metrics(
        true_lat=validation_df["lat"],
        true_lng=validation_df["lng"],
        pred_lat=median_latitude,
        pred_lng=median_longitude,
    )

    print_metrics("Validation results", median_metrics)


if __name__ == "__main__":
    main()