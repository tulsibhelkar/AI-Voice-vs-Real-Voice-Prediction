import torch
import torch.nn as nn


class CNNTransformer(nn.Module):

    def __init__(
        self,
        num_classes=2,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        # -----------------------------------------
        # CNN FEATURE EXTRACTOR
        # -----------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # -----------------------------------------
        # TRANSFORMER
        # -----------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -----------------------------------------
        # FEATURE PROJECTION
        # -----------------------------------------

        self.projection = nn.Linear(
            128,
            transformer_dim
        )

        # -----------------------------------------
        # CLASSIFIER
        # -----------------------------------------

        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),

            nn.Dropout(dropout),

            nn.Linear(
                transformer_dim,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch, 1, 80, 251]

        x = self.cnn(x)

        # Example:
        # [batch, 128, 20, 63]

        # Convert CNN feature map into a sequence.
        # Frequency dimension is averaged.
        x = x.mean(dim=2)

        # [batch, 128, time]

        # Move time dimension before feature dimension.
        x = x.transpose(1, 2)

        # [batch, time, 128]

        x = self.projection(x)

        # [batch, time, transformer_dim]

        x = self.transformer(x)

        # Global average pooling over time.
        x = x.mean(dim=1)

        # [batch, transformer_dim]

        output = self.classifier(x)

        return output


if __name__ == "__main__":

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("CNN + TRANSFORMER MODEL TEST")
    print("=" * 60)

    model = CNNTransformer()

    dummy_input = torch.randn(
        4,
        1,
        80,
        251
    )

    output = model(dummy_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nInput shape :", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Parameters  :", total_parameters)

    print("\nCNN + TRANSFORMER MODEL TEST PASSED")