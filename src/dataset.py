from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


COUNTRIES = (
    "Belarus",
    "Finland",
    "France",
    "Germany",
    "Iceland",
    "Italy",
    "Norway",
    "Poland",
    "Spain",
    "Sweden",
    "Turkey",
    "United_Kingdom",
)

COUNTRY_TO_INDEX = {
    country: index
    for index, country in enumerate(COUNTRIES)
}


class GeoDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        image_size=128,
    ):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)

        required_columns = {
            "filename",
            "country",
            "lat",
            "lng",
            "cell_id",
            "offset_north_km",
            "offset_east_km",
        }

        missing_columns = (
            required_columns - set(self.labels.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing CSV columns: "
                f"{sorted(missing_columns)}"
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
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        image_path = self.image_dir / row["filename"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        coordinates = torch.tensor(
            [row["lat"], row["lng"]],
            dtype=torch.float32,
        )

        country_index = torch.tensor(
            COUNTRY_TO_INDEX[row["country"]],
            dtype=torch.long,
        )

        cell_id = torch.tensor(
            int(row["cell_id"]),
            dtype=torch.long,
        )

        offsets = torch.tensor(
            [
                row["offset_north_km"],
                row["offset_east_km"],
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "coordinates": coordinates,
            "country_index": country_index,
            "cell_id": cell_id,
            "offsets": offsets,
            "filename": row["filename"],
            "country": row["country"],
        }