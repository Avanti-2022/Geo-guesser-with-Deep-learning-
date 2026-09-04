from torch import nn


class GeoCNN(nn.Module):
    def __init__(
        self,
        number_of_countries=12,
    ):
        super().__init__()

        self.number_of_countries = number_of_countries

        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.country_classifier = nn.Linear(
            256,
            number_of_countries,
        )

        # Two offsets for every country:
        # north/south and east/west.
        self.country_offset_heads = nn.Linear(
            256,
            number_of_countries * 2,
        )

    @staticmethod
    def _conv_block(
        input_channels,
        output_channels,
    ):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images):
        features = self.features(images)
        embedding = self.projection(features)

        country_logits = self.country_classifier(
            embedding
        )

        country_offsets = self.country_offset_heads(
            embedding
        )

        country_offsets = country_offsets.view(
            -1,
            self.number_of_countries,
            2,
        )

        return {
            "country_logits": country_logits,
            "country_offsets": country_offsets,
        }


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )