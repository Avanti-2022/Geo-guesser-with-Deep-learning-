from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


TRAIN_CSV = Path("data/splits/train.csv")
VALIDATION_CSV = Path("data/splits/validation.csv")

TRAIN_OUTPUT = Path(
    "data/splits/train_geocells.csv"
)
VALIDATION_OUTPUT = Path(
    "data/splits/validation_geocells.csv"
)
CENTRES_OUTPUT = Path(
    "data/splits/geocell_centres.csv"
)

NUMBER_OF_CELLS = 100
SEED = 42
KM_PER_LATITUDE_DEGREE = 111.32


def coordinates_to_xyz(latitudes, longitudes):
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
    points = np.asarray(points, dtype=float)

    norms = np.linalg.norm(
        points,
        axis=1,
        keepdims=True,
    )
    points = points / np.clip(norms, 1e-12, None)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    latitudes = np.degrees(
        np.arctan2(z, np.sqrt(x**2 + y**2))
    )
    longitudes = np.degrees(np.arctan2(y, x))

    return np.column_stack(
        (latitudes, longitudes)
    )


def add_cell_information(
    dataframe,
    cell_ids,
    cell_centres,
):
    result = dataframe.copy()
    result["cell_id"] = cell_ids

    centre_coordinates = cell_centres[cell_ids]

    result["cell_lat"] = centre_coordinates[:, 0]
    result["cell_lng"] = centre_coordinates[:, 1]

    result["offset_north_km"] = (
        result["lat"] - result["cell_lat"]
    ) * KM_PER_LATITUDE_DEGREE

    longitude_scale = (
        KM_PER_LATITUDE_DEGREE
        * np.cos(np.radians(result["cell_lat"]))
    )

    result["offset_east_km"] = (
        result["lng"] - result["cell_lng"]
    ) * longitude_scale

    return result


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

    print(
        f"Creating {NUMBER_OF_CELLS} geocells..."
    )

    clustering = KMeans(
        n_clusters=NUMBER_OF_CELLS,
        random_state=SEED,
        n_init=10,
    )

    train_cell_ids = clustering.fit_predict(
        train_xyz
    )

    validation_cell_ids = clustering.predict(
        validation_xyz
    )

    cell_centres = xyz_to_coordinates(
        clustering.cluster_centers_
    )

    train_output = add_cell_information(
        train_df,
        train_cell_ids,
        cell_centres,
    )

    validation_output = add_cell_information(
        validation_df,
        validation_cell_ids,
        cell_centres,
    )

    centres_output = pd.DataFrame(
        {
            "cell_id": np.arange(NUMBER_OF_CELLS),
            "cell_lat": cell_centres[:, 0],
            "cell_lng": cell_centres[:, 1],
        }
    )

    train_output.to_csv(
        TRAIN_OUTPUT,
        index=False,
    )
    validation_output.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )
    centres_output.to_csv(
        CENTRES_OUTPUT,
        index=False,
    )

    cell_counts = train_output[
        "cell_id"
    ].value_counts()

    print(f"Training rows: {len(train_output)}")
    print(
        f"Validation rows: "
        f"{len(validation_output)}"
    )
    print(f"Cells: {len(centres_output)}")
    print(
        "Smallest training cell: "
        f"{cell_counts.min()} images"
    )
    print(
        "Largest training cell: "
        f"{cell_counts.max()} images"
    )
    print(
        "Missing values: "
        f"{train_output.isna().sum().sum()}"
    )

    assert len(train_output) == len(train_df)
    assert len(validation_output) == len(validation_df)
    assert len(centres_output) == NUMBER_OF_CELLS
    assert train_output["cell_id"].nunique() == NUMBER_OF_CELLS
    assert train_output.isna().sum().sum() == 0
    assert validation_output.isna().sum().sum() == 0

    print("\nSaved:")
    print(TRAIN_OUTPUT)
    print(VALIDATION_OUTPUT)
    print(CENTRES_OUTPUT)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()