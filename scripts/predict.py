from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import GeoCNN


HOLDOUT_DIR = Path("data/holdout_public")
CHECKPOINT_PATH = Path(
    "outputs/checkpoints/best_model.pt"
)
OUTPUT_FILE = Path("predictions.csv")

BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class HoldoutDataset(Dataset):
    def __init__(self, image_dir, image_size):
        self.image_dir = Path(image_dir)

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "filename": image_path.name,
        }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    image_size = checkpoint["image_size"]

    dataset = HoldoutDataset(
        image_dir=HOLDOUT_DIR,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = GeoCNN().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    coordinate_mean = checkpoint[
        "coordinate_mean"
    ].to(device)

    coordinate_std = checkpoint[
        "coordinate_std"
    ].to(device)

    filenames = []
    predicted_latitudes = []
    predicted_longitudes = []

    print(f"Holdout images: {len(dataset)}")
    print(
        "Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        "Checkpoint median distance: "
        f"{checkpoint['validation_metrics']['median_distance_km']:.2f} km"
    )

    with torch.no_grad():
        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):
            images = batch["image"].to(device)

            normalized_predictions = model(images)

            predictions = (
                normalized_predictions * coordinate_std
                + coordinate_mean
            )

            predictions = predictions.cpu()

            filenames.extend(batch["filename"])
            predicted_latitudes.extend(
                predictions[:, 0].tolist()
            )
            predicted_longitudes.extend(
                predictions[:, 1].tolist()
            )

            if batch_number % 20 == 0:
                print(
                    f"Processed {batch_number}/"
                    f"{len(loader)} batches"
                )

    predictions_df = pd.DataFrame(
        {
            "filename": filenames,
            "pred_lat": predicted_latitudes,
            "pred_lng": predicted_longitudes,
        }
    )

    # Validate the result before saving.
    if len(predictions_df) != len(dataset):
        raise ValueError(
            "Prediction row count does not match "
            "the holdout image count."
        )

    if predictions_df["filename"].duplicated().any():
        raise ValueError(
            "Duplicate filenames found."
        )

    if predictions_df.isna().any().any():
        raise ValueError(
            "Missing prediction values found."
        )

    if not predictions_df["pred_lat"].between(
        -90,
        90,
    ).all():
        raise ValueError(
            "Invalid latitude prediction found."
        )

    if not predictions_df["pred_lng"].between(
        -180,
        180,
    ).all():
        raise ValueError(
            "Invalid longitude prediction found."
        )

    predictions_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(predictions_df)}")
    print(
        "Columns: "
        f"{predictions_df.columns.tolist()}"
    )
    print("\nFirst five predictions:")
    print(predictions_df.head())


if __name__ == "__main__":
    main()