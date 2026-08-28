from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
HOLDOUT_DIR = DATA_DIR / "holdout_public"
LABELS_FILE = DATA_DIR / "train_labels.csv"


def main():
    labels = pd.read_csv(LABELS_FILE)

    print("Dataset summary")
    print("----------------")
    print(f"CSV rows: {len(labels)}")
    print(f"Training images: {len(list(TRAIN_DIR.glob('*')))}")
    print(f"Holdout images: {len(list(HOLDOUT_DIR.glob('*')))}")

    print("\nColumns:")
    print(labels.columns.tolist())

    print("\nFirst five rows:")
    print(labels.head())

    print("\nMissing values:")
    print(labels.isna().sum())

    print("\nCoordinate ranges:")
    print(f"Latitude:  {labels['lat'].min()} to {labels['lat'].max()}")
    print(f"Longitude: {labels['lng'].min()} to {labels['lng'].max()}")

    print("\nImages per country:")
    print(labels["country"].value_counts())

    image_filenames = {
        path.name for path in TRAIN_DIR.iterdir() if path.is_file()
    }
    csv_filenames = set(labels["filename"])

    missing_images = csv_filenames - image_filenames
    images_without_labels = image_filenames - csv_filenames

    print("\nFile checks:")
    print(f"CSV entries without an image: {len(missing_images)}")
    print(f"Training images without a CSV entry: {len(images_without_labels)}")


if __name__ == "__main__":
    main()