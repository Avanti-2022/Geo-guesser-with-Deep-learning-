import torch
from torch.utils.data import DataLoader

from src.dataset import GeoDataset
from src.model import GeoCNN, count_parameters


MAX_PARAMETERS = 5_000_000


def main():
    dataset = GeoDataset(
        csv_file="data/splits/train.csv",
        image_dir="data/train",
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    images = batch["image"]

    model = GeoCNN()
    parameter_count = count_parameters(model)

    with torch.no_grad():
        predictions = model(images)

    print(f"Dataset size: {len(dataset)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Parameter limit: {MAX_PARAMETERS:,}")

    assert predictions.shape == (8, 2)
    assert parameter_count <= MAX_PARAMETERS

    print("Model checks passed.")


if __name__ == "__main__":
    main()