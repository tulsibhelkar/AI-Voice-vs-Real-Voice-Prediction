import torch
import torch.nn as nn
from pathlib import Path

from training.dataloader import create_dataloader
from training.model_cnn_transformer import CNNTransformer
from training.config import DEVICE, LEARNING_RATE, NUM_EPOCHS


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu"
):

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (features, labels) in enumerate(loader):

        features = features.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if is_training and (batch_index + 1) % 10 == 0:

            print(
                f"Batch {batch_index + 1} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = (
        correct / total
    ) * 100

    return epoch_loss, epoch_accuracy


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("CNN + TRANSFORMER TRAINING")
    print("=" * 60)

    print("\nDevice:", DEVICE)
    print("Number of epochs:", NUM_EPOCHS)

    save_directory = Path(
        "saved_models"
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading training dataset...")

    train_loader = create_dataloader(
        "train",
        shuffle=True
    )

    print("\nLoading validation dataset...")

    val_loader = create_dataloader(
        "val",
        shuffle=False
    )

    model = CNNTransformer(
        num_classes=2
    ).to(DEVICE)

    print("\nModel created successfully.")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        print("=" * 60)
        print(
            f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
        )
        print("=" * 60)

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE
        )

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=DEVICE
        )

        print("\nEpoch Results:")

        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_accuracy:.2f}%"
        )

        print(
            f"Val Loss       : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : {val_accuracy:.2f}%"
        )

    save_path = (
        save_directory /
        "cnn_transformer_final.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "CNNTransformer",
            "num_classes": 2
        },
        save_path
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nFinal Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Final Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        "\nModel saved successfully:"
    )

    print(save_path)


if __name__ == "__main__":
    main()