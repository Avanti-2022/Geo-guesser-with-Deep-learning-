import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import GeoDataset
from src.model import GeoCNN, count_parameters


SEED = 42
BATCH_SIZE = 8
TINY_DATASET_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MAX_PARAMETERS = 5_000_000


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    full_dataset = GeoDataset(
        csv_file="data/splits/train.csv",
        image_dir="data/train",
        image_size=224,
    )

    # Use only 32 images for this test.
    tiny_dataset = Subset(
        full_dataset,
        range(TINY_DATASET_SIZE),
    )

    loader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Normalize coordinates to make training easier.
    coordinate_mean = torch.tensor(
        full_dataset.labels[["lat", "lng"]].mean().to_numpy(),
        dtype=torch.float32,
        device=device,
    )

    coordinate_std = torch.tensor(
        full_dataset.labels[["lat", "lng"]].std().to_numpy(),
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

    print(f"Training images: {len(tiny_dataset)}")
    print(f"Parameter count: {parameter_count:,}")
    print("Starting tiny overfitting test...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
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

        average_loss = total_loss / len(tiny_dataset)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} "
                f"- loss: {average_loss:.4f}"
            )

    print("\nTiny overfitting test finished.")

    # Show one prediction converted back to latitude/longitude.
    model.eval()

    with torch.no_grad():
        sample = full_dataset[0]
        image = sample["image"].unsqueeze(0).to(device)

        normalized_prediction = model(image)[0]
        prediction = (
            normalized_prediction * coordinate_std
            + coordinate_mean
        )

    print(f"Actual coordinates:    {sample['coordinates']}")
    print(f"Predicted coordinates: {prediction.cpu()}")


if __name__ == "__main__":
    main()