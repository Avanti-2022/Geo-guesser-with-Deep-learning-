import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import GeoDataset
from src.metrics import calculate_metrics
from src.model import GeoCNN, count_parameters


SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_PARAMETERS = 5_000_000

TRAIN_CSV = "data/splits/train.csv"
VALIDATION_CSV = "data/splits/validation.csv"
IMAGE_DIR = "data/train"
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")


def train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    coordinate_mean,
    coordinate_std,
    device,
):
    model.train()
    total_loss = 0.0

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        coordinates = batch["coordinates"].to(device)

        normalized_coordinates = (
            coordinates - coordinate_mean
        ) / coordinate_std

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(
            predictions,
            normalized_coordinates,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        if batch_number % 50 == 0:
            print(
                f"  Batch {batch_number}/{len(loader)} "
                f"- loss: {loss.item():.4f}"
            )

    return total_loss / len(loader.dataset)


def validate(
    model,
    loader,
    coordinate_mean,
    coordinate_std,
    device,
):
    model.eval()

    actual_coordinates = []
    predicted_coordinates = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            coordinates = batch["coordinates"]

            normalized_predictions = model(images)

            predictions = (
                normalized_predictions * coordinate_std
                + coordinate_mean
            )

            actual_coordinates.append(coordinates.numpy())
            predicted_coordinates.append(
                predictions.cpu().numpy()
            )

    actual_coordinates = np.concatenate(
        actual_coordinates,
        axis=0,
    )
    predicted_coordinates = np.concatenate(
        predicted_coordinates,
        axis=0,
    )

    return calculate_metrics(
        true_lat=actual_coordinates[:, 0],
        true_lng=actual_coordinates[:, 1],
        pred_lat=predicted_coordinates[:, 0],
        pred_lng=predicted_coordinates[:, 1],
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    train_dataset = GeoDataset(
        csv_file=TRAIN_CSV,
        image_dir=IMAGE_DIR,
        image_size=IMAGE_SIZE,
    )

    validation_dataset = GeoDataset(
        csv_file=VALIDATION_CSV,
        image_dir=IMAGE_DIR,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    coordinate_mean = torch.tensor(
        train_dataset.labels[["lat", "lng"]]
        .mean()
        .to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    coordinate_std = torch.tensor(
        train_dataset.labels[["lat", "lng"]]
        .std()
        .to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    model = GeoCNN().to(device)
    parameter_count = count_parameters(model)

    assert parameter_count <= MAX_PARAMETERS

    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Batches per epoch: {len(train_loader)}")

    best_median_distance = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")

        training_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            coordinate_mean=coordinate_mean,
            coordinate_std=coordinate_std,
            device=device,
        )

        metrics = validate(
            model=model,
            loader=validation_loader,
            coordinate_mean=coordinate_mean,
            coordinate_std=coordinate_std,
            device=device,
        )

        elapsed_minutes = (time.time() - start_time) / 60

        print(f"Training loss: {training_loss:.4f}")
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
        print(f"Epoch time: {elapsed_minutes:.2f} minutes")

        if (
            metrics["median_distance_km"]
            < best_median_distance
        ):
            best_median_distance = (
                metrics["median_distance_km"]
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "coordinate_mean": (
                        coordinate_mean.cpu()
                    ),
                    "coordinate_std": coordinate_std.cpu(),
                    "image_size": IMAGE_SIZE,
                    "parameter_count": parameter_count,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                },
                CHECKPOINT_PATH,
            )

            print(f"Saved best model: {CHECKPOINT_PATH}")

    print(
        "\nBest median validation distance: "
        f"{best_median_distance:.2f} km"
    )


if __name__ == "__main__":
    main()