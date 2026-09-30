import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
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
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("BASELINE CNN TEST")
    print("=" * 60)

    model = BaselineCNN()

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

    print("\nBASELINE CNN TEST PASSED")