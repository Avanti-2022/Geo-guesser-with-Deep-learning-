import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import GeoCNN


HOLDOUT_DIR = Path("data/holdout_public")
BATCH_SIZE = 16
KM_PER_LATITUDE_DEGREE = 111.32
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


def all_country_coordinates(
    normalized_offsets,
    country_centres,
    offset_means,
    offset_stds,
):
    normalized_offsets = torch.clamp(
        normalized_offsets,
        min=-4.0,
        max=4.0,
    )

    offsets = (
        normalized_offsets
        * offset_stds.unsqueeze(0)
        + offset_means.unsqueeze(0)
    )

    centre_latitudes = country_centres[
        :, 0
    ].unsqueeze(0)

    centre_longitudes = country_centres[
        :, 1
    ].unsqueeze(0)

    predicted_latitudes = (
        centre_latitudes
        + offsets[:, :, 0]
        / KM_PER_LATITUDE_DEGREE
    )

    longitude_scale = (
        KM_PER_LATITUDE_DEGREE
        * torch.cos(
            torch.deg2rad(centre_latitudes)
        )
    )

    longitude_scale = torch.clamp(
        longitude_scale,
        min=1e-6,
    )

    predicted_longitudes = (
        centre_longitudes
        + offsets[:, :, 1]
        / longitude_scale
    )

    return torch.stack(
        (
            predicted_latitudes,
            predicted_longitudes,
        ),
        dim=2,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate holdout predictions."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("final_model_392km.pt"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {arguments.checkpoint}")

    if not arguments.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{arguments.checkpoint}"
        )

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )

    countries = checkpoint["countries"]
    image_size = checkpoint["image_size"]
    prediction_mode = checkpoint["prediction_mode"]

    model = GeoCNN(
        number_of_countries=len(countries)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    country_centres = checkpoint[
        "country_centres"
    ].to(device)

    offset_means = checkpoint[
        "offset_means"
    ].to(device)

    offset_stds = checkpoint[
        "offset_stds"
    ].to(device)

    dataset = HoldoutDataset(
        image_dir=HOLDOUT_DIR,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    filenames = []
    predicted_latitudes = []
    predicted_longitudes = []

    print(f"Holdout images: {len(dataset)}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Validation median distance: "
        f"{checkpoint['validation_metrics']['median_distance_km']:.2f} km"
    )
    print(f"Prediction mode: {prediction_mode}")
    print(
        "Parameter count: "
        f"{checkpoint['parameter_count']:,}"
    )

    with torch.no_grad():
        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):
            images = batch["image"].to(device)

            output = model(images)

            country_probabilities = torch.softmax(
                output["country_logits"],
                dim=1,
            )

            country_coordinates = (
                all_country_coordinates(
                    normalized_offsets=output[
                        "country_offsets"
                    ],
                    country_centres=country_centres,
                    offset_means=offset_means,
                    offset_stds=offset_stds,
                )
            )

            if prediction_mode == "hard":
                predicted_countries = (
                    country_probabilities.argmax(dim=1)
                )

                batch_indexes = torch.arange(
                    images.size(0),
                    device=device,
                )

                predictions = country_coordinates[
                    batch_indexes,
                    predicted_countries,
                ]

            elif prediction_mode == "weighted":
                predictions = (
                    country_coordinates
                    * country_probabilities.unsqueeze(2)
                ).sum(dim=1)

            else:
                raise ValueError(
                    "Unknown prediction mode: "
                    f"{prediction_mode}"
                )

            predictions = predictions.cpu()

            filenames.extend(batch["filename"])
            predicted_latitudes.extend(
                predictions[:, 0].tolist()
            )
            predicted_longitudes.extend(
                predictions[:, 1].tolist()
            )

            if batch_number % 25 == 0:
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

    if len(predictions_df) != len(dataset):
        raise ValueError(
            "Prediction row count does not match "
            "the holdout image count."
        )

    if predictions_df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found.")

    if predictions_df.isna().any().any():
        raise ValueError("Missing values found.")

    if not predictions_df["pred_lat"].between(
        -90,
        90,
    ).all():
        raise ValueError("Invalid latitude found.")

    if not predictions_df["pred_lng"].between(
        -180,
        180,
    ).all():
        raise ValueError("Invalid longitude found.")

    predictions_df.to_csv(
        arguments.output,
        index=False,
    )

    print(f"\nSaved: {arguments.output}")
    print(f"Rows: {len(predictions_df)}")
    print(
        "Columns: "
        f"{predictions_df.columns.tolist()}"
    )
    print("\nFirst five predictions:")
    print(predictions_df.head())


if __name__ == "__main__":
    main()