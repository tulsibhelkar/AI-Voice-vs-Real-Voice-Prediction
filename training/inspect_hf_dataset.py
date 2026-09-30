from datasets import load_dataset


DATASET_NAME = "garystafford/deepfake-audio-detection"


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("DATASET STRUCTURE CHECK")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print("\nDataset:")
    print(DATASET_NAME)

    print("\nTotal samples:")
    print(len(dataset))

    print("\nColumn names:")
    print(dataset.column_names)

    print("\nDataset features:")
    print(dataset.features)

    print("\nFirst sample keys:")
    print(dataset[0].keys())

    print("\nFirst sample label:")
    print(dataset[0]["label"])

    print("\nDATASET STRUCTURE CHECK PASSED")


if __name__ == "__main__":
    main()