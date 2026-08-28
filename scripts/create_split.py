from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS_FILE = Path("data/train_labels.csv")
SPLITS_DIR = Path("data/splits")
SEED = 42
VALIDATION_FRACTION = 0.20


def main():
    labels = pd.read_csv(LABELS_FILE)

    train_df, validation_df = train_test_split(
        labels,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=labels["country"],
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_file = SPLITS_DIR / "train.csv"
    validation_file = SPLITS_DIR / "validation.csv"

    train_df.to_csv(train_file, index=False)
    validation_df.to_csv(validation_file, index=False)

    print("Split created")
    print("-------------")
    print(f"Original rows:   {len(labels)}")
    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")

    print("\nTraining images per country:")
    print(train_df["country"].value_counts().sort_index())

    print("\nValidation images per country:")
    print(validation_df["country"].value_counts().sort_index())

    overlap = set(train_df["filename"]) & set(validation_df["filename"])
    print(f"\nFiles present in both splits: {len(overlap)}")

    assert len(train_df) + len(validation_df) == len(labels)
    assert len(overlap) == 0
    assert not train_df.isna().any().any()
    assert not validation_df.isna().any().any()

    print("\nAll checks passed.")
    print(f"Saved: {train_file}")
    print(f"Saved: {validation_file}")


if __name__ == "__main__":
    main()