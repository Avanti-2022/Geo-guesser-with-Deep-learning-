import pandas as pd

from src.metrics import calculate_metrics


TRAIN_CSV = "data/splits/train.csv"
VALIDATION_CSV = "data/splits/validation.csv"


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    validation_df = pd.read_csv(VALIDATION_CSV)

    country_centres = (
        train_df.groupby("country")[["lat", "lng"]]
        .median()
        .rename(
            columns={
                "lat": "pred_lat",
                "lng": "pred_lng",
            }
        )
    )

    predictions = validation_df.merge(
        country_centres,
        left_on="country",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    metrics = calculate_metrics(
        true_lat=predictions["lat"],
        true_lng=predictions["lng"],
        pred_lat=predictions["pred_lat"],
        pred_lng=predictions["pred_lng"],
    )

    print("True-country centroid oracle")
    print("----------------------------")
    print(
        "Median distance: "
        f"{metrics['median_distance_km']:.2f} km"
    )
    print(
        "Mean distance: "
        f"{metrics['mean_distance_km']:.2f} km"
    )
    print(
        "Below 200 km: "
        f"{metrics['below_200_km']:.2f}%"
    )
    print(
        "Below 750 km: "
        f"{metrics['below_750_km']:.2f}%"
    )

    print("\nMedian distance by country")
    print("--------------------------")

    for country, group in predictions.groupby("country"):
        country_metrics = calculate_metrics(
            true_lat=group["lat"],
            true_lng=group["lng"],
            pred_lat=group["pred_lat"],
            pred_lng=group["pred_lng"],
        )

        print(
            f"{country:15s} "
            f"{country_metrics['median_distance_km']:.2f} km"
        )


if __name__ == "__main__":
    main()