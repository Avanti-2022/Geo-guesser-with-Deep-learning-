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
BATCH_SIZE = 16
EPOCHS = 30

INITIAL_LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0002
FINE_TUNING_START_EPOCH = 16

OFFSET_LOSS_WEIGHT = 1.0
MAX_PARAMETERS = 5_000_000
KM_PER_LATITUDE_DEGREE = 111.32

TRAIN_CSV = "data/splits/train.csv"
VALIDATION_CSV = "data/splits/validation.csv"
IMAGE_DIR = "data/train"

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/country_heads_best_model.pt"
)


def calculate_country_statistics(dataset, device):
    centres = []
    offset_means = []
    offset_stds = []

    labels = dataset.labels

    for country in COUNTRIES:
        country_rows = labels[
            labels["country"] == country
        ]

        centre_lat = country_rows["lat"].median()
        centre_lng = country_rows["lng"].median()

        north_offsets = (
            country_rows["lat"] - centre_lat
        ) * KM_PER_LATITUDE_DEGREE

        longitude_scale = (
            KM_PER_LATITUDE_DEGREE
            * np.cos(np.radians(centre_lat))
        )

        east_offsets = (
            country_rows["lng"] - centre_lng
        ) * longitude_scale

        offsets = np.column_stack(
            (north_offsets, east_offsets)
        )

        centres.append([centre_lat, centre_lng])
        offset_means.append(offsets.mean(axis=0))
        offset_stds.append(offsets.std(axis=0))

    centres = torch.tensor(
        np.asarray(centres),
        dtype=torch.float32,
        device=device,
    )

    offset_means = torch.tensor(
        np.asarray(offset_means),
        dtype=torch.float32,
        device=device,
    )

    offset_stds = torch.tensor(
        np.asarray(offset_stds),
        dtype=torch.float32,
        device=device,
    )

    offset_stds = torch.clamp(
        offset_stds,
        min=1.0,
    )

    return centres, offset_means, offset_stds


def calculate_target_offsets(
    coordinates,
    country_ids,
    country_centres,
):
    selected_centres = country_centres[country_ids]

    centre_latitudes = selected_centres[:, 0]
    centre_longitudes = selected_centres[:, 1]

    north_offsets = (
        coordinates[:, 0] - centre_latitudes
    ) * KM_PER_LATITUDE_DEGREE

    longitude_scale = (
        KM_PER_LATITUDE_DEGREE
        * torch.cos(
            torch.deg2rad(centre_latitudes)
        )
    )

    east_offsets = (
        coordinates[:, 1] - centre_longitudes
    ) * longitude_scale

    return torch.stack(
        (north_offsets, east_offsets),
        dim=1,
    )


def all_country_coordinates(
    normalized_offsets,
    country_centres,
    offset_means,
    offset_stds,
):
    # Prevent extreme coordinate predictions.
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


def train_one_epoch(
    model,
    loader,
    country_loss_function,
    offset_loss_function,
    optimizer,
    country_centres,
    offset_means,
    offset_stds,
    device,
):
    model.train()

    total_loss = 0.0
    total_country_loss = 0.0
    total_offset_loss = 0.0
    correct_countries = 0

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch["image"].to(device)
        coordinates = batch["coordinates"].to(device)
        country_ids = batch["country_index"].to(device)

        output = model(images)

        country_loss = country_loss_function(
            output["country_logits"],
            country_ids,
        )

        batch_indexes = torch.arange(
            images.size(0),
            device=device,
        )

        selected_offset_predictions = output[
            "country_offsets"
        ][batch_indexes, country_ids]

        target_offsets = calculate_target_offsets(
            coordinates=coordinates,
            country_ids=country_ids,
            country_centres=country_centres,
        )

        selected_offset_means = offset_means[
            country_ids
        ]
        selected_offset_stds = offset_stds[
            country_ids
        ]

        normalized_targets = (
            target_offsets - selected_offset_means
        ) / selected_offset_stds

        offset_loss = offset_loss_function(
            selected_offset_predictions,
            normalized_targets,
        )

        loss = (
            country_loss
            + OFFSET_LOSS_WEIGHT * offset_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += loss.item() * current_batch_size
        total_country_loss += (
            country_loss.item() * current_batch_size
        )
        total_offset_loss += (
            offset_loss.item() * current_batch_size
        )

        predicted_countries = output[
            "country_logits"
        ].argmax(dim=1)

        correct_countries += (
            predicted_countries == country_ids
        ).sum().item()

        if batch_number % 100 == 0:
            print(
                f"  Batch {batch_number}/{len(loader)} "
                f"- loss: {loss.item():.4f}"
            )

    dataset_size = len(loader.dataset)

    return {
        "total_loss": total_loss / dataset_size,
        "country_loss": (
            total_country_loss / dataset_size
        ),
        "offset_loss": (
            total_offset_loss / dataset_size
        ),
        "country_accuracy": (
            100 * correct_countries / dataset_size
        ),
    }


def validate(
    model,
    loader,
    country_centres,
    offset_means,
    offset_stds,
    device,
):
    model.eval()

    actual_coordinates = []
    hard_predictions = []
    weighted_predictions = []

    correct_countries = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            country_ids = batch["country_index"].to(device)

            output = model(images)

            country_probabilities = torch.softmax(
                output["country_logits"],
                dim=1,
            )

            predicted_countries = (
                country_probabilities.argmax(dim=1)
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

            batch_indexes = torch.arange(
                images.size(0),
                device=device,
            )

            hard_coordinates = country_coordinates[
                batch_indexes,
                predicted_countries,
            ]

            weighted_coordinates = (
                country_coordinates
                * country_probabilities.unsqueeze(2)
            ).sum(dim=1)

            correct_countries += (
                predicted_countries == country_ids
            ).sum().item()

            total_examples += images.size(0)

            actual_coordinates.append(
                batch["coordinates"].numpy()
            )

            hard_predictions.append(
                hard_coordinates.cpu().numpy()
            )

            weighted_predictions.append(
                weighted_coordinates.cpu().numpy()
            )

    actual_coordinates = np.concatenate(
        actual_coordinates,
        axis=0,
    )

    hard_predictions = np.concatenate(
        hard_predictions,
        axis=0,
    )

    weighted_predictions = np.concatenate(
        weighted_predictions,
        axis=0,
    )

    hard_metrics = calculate_metrics(
        true_lat=actual_coordinates[:, 0],
        true_lng=actual_coordinates[:, 1],
        pred_lat=hard_predictions[:, 0],
        pred_lng=hard_predictions[:, 1],
    )

    weighted_metrics = calculate_metrics(
        true_lat=actual_coordinates[:, 0],
        true_lng=actual_coordinates[:, 1],
        pred_lat=weighted_predictions[:, 0],
        pred_lng=weighted_predictions[:, 1],
    )

    country_accuracy = (
        100 * correct_countries / total_examples
    )

    if (
        weighted_metrics["median_distance_km"]
        < hard_metrics["median_distance_km"]
    ):
        selected_mode = "weighted"
        selected_metrics = weighted_metrics
    else:
        selected_mode = "hard"
        selected_metrics = hard_metrics

    return {
        "country_accuracy": country_accuracy,
        "hard_metrics": hard_metrics,
        "weighted_metrics": weighted_metrics,
        "selected_mode": selected_mode,
        "selected_metrics": selected_metrics,
    }


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
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    country_centres, offset_means, offset_stds = (
        calculate_country_statistics(
            train_dataset,
            device,
        )
    )

    model = GeoCNN(
        number_of_countries=len(COUNTRIES)
    ).to(device)

    parameter_count = count_parameters(model)
    assert parameter_count <= MAX_PARAMETERS

    country_loss_function = nn.CrossEntropyLoss()
    offset_loss_function = nn.SmoothL1Loss()

    current_learning_rate = (
        INITIAL_LEARNING_RATE
    )

    optimizer = create_optimizer(
        model,
        current_learning_rate,
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Using device: {device}")
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
            country_loss_function=(
                country_loss_function
            ),
            offset_loss_function=(
                offset_loss_function
            ),
            optimizer=optimizer,
            country_centres=country_centres,
            offset_means=offset_means,
            offset_stds=offset_stds,
            device=device,
        )

        validation_results = validate(
            model=model,
            loader=validation_loader,
            country_centres=country_centres,
            offset_means=offset_means,
            offset_stds=offset_stds,
            device=device,
        )

        metrics = validation_results[
            "selected_metrics"
        ]

        elapsed_minutes = (
            time.time() - start_time
        ) / 60

        print(
            "Training loss: "
            f"{training_results['total_loss']:.4f}"
        )
        print(
            "Training country accuracy: "
            f"{training_results['country_accuracy']:.2f}%"
        )
        print(
            "Validation country accuracy: "
            f"{validation_results['country_accuracy']:.2f}%"
        )
        print(
            "Hard median distance: "
            f"{validation_results['hard_metrics']['median_distance_km']:.2f} km"
        )
        print(
            "Weighted median distance: "
            f"{validation_results['weighted_metrics']['median_distance_km']:.2f} km"
        )
        print(
            "Selected mode: "
            f"{validation_results['selected_mode']}"
        )
        print(
            "Selected median distance: "
            f"{metrics['median_distance_km']:.2f} km"
        )
        print(
            "Selected mean distance: "
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
                    "country_centres": (
                        country_centres.cpu()
                    ),
                    "offset_means": (
                        offset_means.cpu()
                    ),
                    "offset_stds": (
                        offset_stds.cpu()
                    ),
                    "countries": COUNTRIES,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "prediction_mode": (
                        validation_results[
                            "selected_mode"
                        ]
                    ),
                    "validation_metrics": metrics,
                    "country_accuracy": (
                        validation_results[
                            "country_accuracy"
                        ]
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