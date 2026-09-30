from pathlib import Path
import json


SPLIT_DIR = Path("data/splits")


def load_indices(filename):
    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as file:
        return json.load(file)


def main():
    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("SAVED SPLIT VERIFICATION")
    print("=" * 60)

    train_indices = load_indices("train_indices.json")
    val_indices = load_indices("val_indices.json")
    test_indices = load_indices("test_indices.json")

    print("\nTrain indices     :", len(train_indices))
    print("Validation indices:", len(val_indices))
    print("Test indices      :", len(test_indices))

    all_indices = (
        train_indices
        + val_indices
        + test_indices
    )

    print("\nTotal indices:", len(all_indices))

    # Check for duplicate indices
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Duplicate indices detected!")

    # Check that all dataset indices are covered
    expected_indices = set(range(1866))
    actual_indices = set(all_indices)

    if actual_indices != expected_indices:
        raise ValueError("Some dataset indices are missing!")

    print("\nDuplicate check: PASSED")
    print("Coverage check : PASSED")

    print("\nSAVED SPLIT VERIFICATION PASSED")


if __name__ == "__main__":
    main()