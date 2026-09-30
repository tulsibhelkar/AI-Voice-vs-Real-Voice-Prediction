from pathlib import Path
import json

from datasets import load_dataset
from sklearn.model_selection import train_test_split


DATASET_NAME = "garystafford/deepfake-audio-detection"
RANDOM_SEED = 42

SPLIT_DIR = Path("data/splits")


def main():
    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("DATASET SPLITTING AND SAVING")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print("\nTotal samples:", len(dataset))

    indices = list(range(len(dataset)))

    labels = dataset["label"]

    # 80% Train + 20% Temporary
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # Temporary -> 10% Validation + 10% Test
    temp_labels = [labels[i] for i in temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels
    )

    print("\nDataset split:")
    print("Train      :", len(train_indices))
    print("Validation :", len(val_indices))
    print("Test       :", len(test_indices))

    def count_classes(index_list):
        real = 0
        fake = 0

        for index in index_list:
            label = int(labels[index])

            if label == 0:
                real += 1
            else:
                fake += 1

        return real, fake

    train_real, train_fake = count_classes(train_indices)
    val_real, val_fake = count_classes(val_indices)
    test_real, test_fake = count_classes(test_indices)

    print("\nClass distribution:")
    print(f"Train      -> REAL: {train_real}, FAKE: {train_fake}")
    print(f"Validation -> REAL: {val_real}, FAKE: {val_fake}")
    print(f"Test       -> REAL: {test_real}, FAKE: {test_fake}")

    # Create split directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Save exact indices
    with open(SPLIT_DIR / "train_indices.json", "w") as file:
        json.dump(train_indices, file, indent=2)

    with open(SPLIT_DIR / "val_indices.json", "w") as file:
        json.dump(val_indices, file, indent=2)

    with open(SPLIT_DIR / "test_indices.json", "w") as file:
        json.dump(test_indices, file, indent=2)

    print("\nSaved files:")
    print("data/splits/train_indices.json")
    print("data/splits/val_indices.json")
    print("data/splits/test_indices.json")

    print("\nDATASET SPLIT SAVED SUCCESSFULLY")


if __name__ == "__main__":
    main()