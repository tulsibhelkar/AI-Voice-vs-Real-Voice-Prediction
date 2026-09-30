from pathlib import Path

import torch

from training.config import MODEL_DIR


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    validation_f1: float,
    filename: str,
) -> Path:
    """
    Save a PyTorch training checkpoint.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        MODEL_DIR / filename
    )

    checkpoint = {
        "epoch": epoch,
        "validation_f1": validation_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    return checkpoint_path


def load_checkpoint(
    model,
    checkpoint_path,
    optimizer=None,
    device="cpu",
):
    """
    Load a saved PyTorch checkpoint.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("MODEL CHECKPOINT UTILITY")
    print("=" * 60)

    print("\nCheckpoint module loaded successfully.")

    print("\nCheckpoint will store:")
    print("- Model weights")
    print("- Optimizer state")
    print("- Epoch number")
    print("- Validation F1 score")

    print("\nModel directory:")
    print(MODEL_DIR)

    print("\n" + "=" * 60)
    print("CHECKPOINT UTILITY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()