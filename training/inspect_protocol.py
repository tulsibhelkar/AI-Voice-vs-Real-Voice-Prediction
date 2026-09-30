from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ASVspoof5_protocols"
    / "ASVspoof5.train.tsv"
)


COLUMN_NAMES = [
    "speaker_id",
    "audio_id",
    "gender",
    "codec",
    "codec_q",
    "codec_type",
    "attack_id",
    "attack_group",
    "label",
    "trim"
]


def main() -> None:

    print("=" * 70)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("ASVSPOOF 5 TRAINING PROTOCOL INSPECTION")
    print("=" * 70)

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"Protocol file not found:\n{PROTOCOL_PATH}"
        )

    print("\nProtocol file found:")
    print(PROTOCOL_PATH)

    print("\nReading ASVspoof 5 training protocol...")

    df = pd.read_csv(
        PROTOCOL_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python"
    )

    print("\nProtocol loaded successfully.")

    print("\n" + "-" * 70)
    print("DATASET SIZE")
    print("-" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n" + "-" * 70)
    print("COLUMN NAMES")
    print("-" * 70)

    for number, column in enumerate(df.columns, start=1):
        print(f"{number}. {column}")

    print("\n" + "-" * 70)
    print("FIRST 5 RECORDS")
    print("-" * 70)

    print(df.head().to_string(index=False))

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    if "label" in df.columns:

        label_counts = df["label"].value_counts()

        print(label_counts.to_string())

        print("\nPercentages:")

        percentages = (
            df["label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        for label, percentage in percentages.items():
            print(f"{label}: {percentage}%")

    print("\n" + "-" * 70)
    print("SPEAKER INFORMATION")
    print("-" * 70)

    print(
        f"Unique speakers: "
        f"{df['speaker_id'].nunique():,}"
    )

    print("\nGender distribution:")

    print(
        df["gender"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("ATTACK INFORMATION")
    print("-" * 70)

    print("Attack IDs:")

    print(
        df["attack_id"]
        .value_counts()
        .to_string()
    )

    print("\nAttack groups:")

    print(
        df["attack_group"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("ASVSPOOF 5 PROTOCOL INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()