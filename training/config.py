from pathlib import Path

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "saved_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

AUDIO_DURATION = 4.0

TARGET_AUDIO_LENGTH = int(
    SAMPLE_RATE * AUDIO_DURATION
)

N_MELS = 80

N_FFT = 1024

HOP_LENGTH = 256


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 15

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4


# ============================================================
# CLASS CONFIGURATION
# ============================================================

NUM_CLASSES = 2

CLASS_NAMES = [
    "REAL",
    "AI-GENERATED",
]

LABEL_TO_INDEX = {
    "real": 0,
    "fake": 1,
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print("\nAUDIO SETTINGS")

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"Audio duration: "
        f"{AUDIO_DURATION} seconds"
    )

    print(
        f"Target samples: "
        f"{TARGET_AUDIO_LENGTH}"
    )

    print(
        f"Mel bands: "
        f"{N_MELS}"
    )

    print(
        f"FFT size: "
        f"{N_FFT}"
    )

    print(
        f"Hop length: "
        f"{HOP_LENGTH}"
    )

    print("\nTRAINING SETTINGS")

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print("\nCLASS SETTINGS")

    print(
        f"Number of classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Classes: "
        f"{CLASS_NAMES}"
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nModel directory:")
    print(MODEL_DIR)

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()