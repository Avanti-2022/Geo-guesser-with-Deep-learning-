import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import COUNTRIES, GeoDataset
from src.metrics import calculate_metrics
from src.model import GeoCNN, count_parameters


SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

INITIAL_LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0002
FINE_TUNING_START_EPOCH = 11

COUNTRY_LOSS_WEIGHT = 0.25
MAX_PARAMETERS = 5_000_000

TRAIN_CSV = "data/splits/train.csv"
VALIDATION_CSV = "data/splits/validation.csv"
IMAGE_DIR = "data/train"

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/country_aware_best_model.pt"
)


def train_one_epoch(
    model,
    loader,
    coordinate_loss_function,
    country_loss_function,
    optimizer,
    coordinate_mean,
    coordinate_std,
    device,
):
    model.train()

    total_loss = 0.0
    total_coordinate_loss = 0.0
    total_country_loss = 0.0
    correct_countries = 0

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        coordinates = batch["coordinates"].to(device)
        country_indexes = batch["country_index"].to(device)

        normalized_coordinates = (
            coordinates - coordinate_mean
        ) / coordinate_std

        optimizer.zero_grad()

        output = model(images)

        coordinate_loss = coordinate_loss_function(
            output["coordinates"],
            normalized_coordinates,
        )

        country_loss = country_loss_function(
            output["country_logits"],
            country_indexes,
        )

        loss = (
            coordinate_loss
            + COUNTRY_LOSS_WEIGHT * country_loss
        )

        loss.backward()
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item() * current_batch_size
        )

        total_coordinate_loss += (
            coordinate_loss.item()
            * current_batch_size
        )

        total_country_loss += (
            country_loss.item()
            * current_batch_size
        )

        predicted_countries = output[
            "country_logits"
        ].argmax(dim=1)

        correct_countries += (
            predicted_countries == country_indexes
        ).sum().item()

        if batch_number % 50 == 0:
            print(
                f"  Batch {batch_number}/{len(loader)} "
                f"- total loss: {loss.item():.4f}"
            )

    dataset_size = len(loader.dataset)

    return {
        "total_loss": total_loss / dataset_size,
        "coordinate_loss": (
            total_coordinate_loss / dataset_size
        ),
        "country_loss": (
            total_country_loss / dataset_size
        ),
        "country_accuracy": (
            100 * correct_countries / dataset_size
        ),
    }


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

    correct_countries = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            coordinates = batch["coordinates"]

            country_indexes = batch[
                "country_index"
            ].to(device)

            output = model(images)

            predictions = (
                output["coordinates"]
                * coordinate_std
                + coordinate_mean
            )

            predicted_countries = output[
                "country_logits"
            ].argmax(dim=1)

            correct_countries += (
                predicted_countries
                == country_indexes
            ).sum().item()

            total_examples += images.size(0)

            actual_coordinates.append(
                coordinates.numpy()
            )

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

    metrics = calculate_metrics(
        true_lat=actual_coordinates[:, 0],
        true_lng=actual_coordinates[:, 1],
        pred_lat=predicted_coordinates[:, 0],
        pred_lng=predicted_coordinates[:, 1],
    )

    metrics["country_accuracy"] = (
        100 * correct_countries / total_examples
    )

    return metrics


def create_optimizer(model, learning_rate):
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Countries: {len(COUNTRIES)}")
    print(f"Country loss weight: {COUNTRY_LOSS_WEIGHT}")
    print(
        f"Initial learning rate: "
        f"{INITIAL_LEARNING_RATE}"
    )
    print(
        f"Fine-tuning learning rate: "
        f"{FINE_TUNING_LEARNING_RATE}"
    )

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

    model = GeoCNN(
        number_of_countries=len(COUNTRIES)
    ).to(device)

    parameter_count = count_parameters(model)

    assert parameter_count <= MAX_PARAMETERS

    coordinate_loss_function = nn.SmoothL1Loss()
    country_loss_function = nn.CrossEntropyLoss()

    current_learning_rate = INITIAL_LEARNING_RATE

    optimizer = create_optimizer(
        model,
        current_learning_rate,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Training images: {len(train_dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print(f"Parameter count: {parameter_count:,}")
    print(f"Batches per epoch: {len(train_loader)}")

    best_median_distance = float("inf")

    for epoch in range(1, EPOCHS + 1):
        if epoch == FINE_TUNING_START_EPOCH:
            current_learning_rate = (
                FINE_TUNING_LEARNING_RATE
            )

            # A fresh optimizer matches the experiment
            # used to produce the final model.
            optimizer = create_optimizer(
                model,
                current_learning_rate,
            )

            print(
                "\nReduced learning rate to "
                f"{current_learning_rate}"
            )

        start_time = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(
            f"Learning rate: "
            f"{current_learning_rate}"
        )

        training_results = train_one_epoch(
            model=model,
            loader=train_loader,
            coordinate_loss_function=(
                coordinate_loss_function
            ),
            country_loss_function=(
                country_loss_function
            ),
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

        elapsed_minutes = (
            time.time() - start_time
        ) / 60

        print(
            "Training total loss: "
            f"{training_results['total_loss']:.4f}"
        )
        print(
            "Training coordinate loss: "
            f"{training_results['coordinate_loss']:.4f}"
        )
        print(
            "Training country loss: "
            f"{training_results['country_loss']:.4f}"
        )
        print(
            "Training country accuracy: "
            f"{training_results['country_accuracy']:.2f}%"
        )
        print(
            "Validation country accuracy: "
            f"{metrics['country_accuracy']:.2f}%"
        )
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
        print(
            f"Epoch time: "
            f"{elapsed_minutes:.2f} minutes"
        )

        if (
            metrics["median_distance_km"]
            < best_median_distance
        ):
            best_median_distance = (
                metrics["median_distance_km"]
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "coordinate_mean": (
                        coordinate_mean.cpu()
                    ),
                    "coordinate_std": (
                        coordinate_std.cpu()
                    ),
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "validation_metrics": metrics,
                    "countries": COUNTRIES,
                    "country_loss_weight": (
                        COUNTRY_LOSS_WEIGHT
                    ),
                    "initial_learning_rate": (
                        INITIAL_LEARNING_RATE
                    ),
                    "fine_tuning_learning_rate": (
                        FINE_TUNING_LEARNING_RATE
                    ),
                    "fine_tuning_start_epoch": (
                        FINE_TUNING_START_EPOCH
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                f"Saved improved model: "
                f"{CHECKPOINT_PATH}"
            )

    print(
        "\nBest median validation distance: "
        f"{best_median_distance:.2f} km"
    )


if __name__ == "__main__":
    main()