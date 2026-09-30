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
    "trim",
]


LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}


CLASS_NAMES = {
    0: "REAL",
    1: "AI-GENERATED",
}


def load_training_protocol() -> pd.DataFrame:
    """
    Load the official ASVspoof 5 training protocol.

    Returns:
        pandas DataFrame containing the metadata and
        numerical class labels.
    """

    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"ASVspoof protocol not found:\n{PROTOCOL_PATH}"
        )

    df = pd.read_csv(
        PROTOCOL_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    if len(df.columns) != len(COLUMN_NAMES):
        raise ValueError(
            "Unexpected number of columns in ASVspoof protocol."
        )

    unknown_labels = set(df["label"].unique()) - set(LABEL_MAP.keys())

    if unknown_labels:
        raise ValueError(
            f"Unknown ASVspoof labels found: {unknown_labels}"
        )

    df["target"] = df["label"].map(LABEL_MAP)

    return df


def main() -> None:

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("DATASET METADATA MODULE")
    print("=" * 60)

    df = load_training_protocol()

    print("\nProtocol loaded successfully.")
    print(f"Total recordings: {len(df):,}")
    print(f"Unique speakers: {df['speaker_id'].nunique():,}")

    print("\nOriginal ASVspoof labels:")

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    print("\nProject class mapping:")
    print("bonafide -> 0 -> REAL")
    print("spoof    -> 1 -> AI-GENERATED")

    print("\nNumerical target distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst 5 audio IDs:")

    print(
        df[
            [
                "audio_id",
                "speaker_id",
                "label",
                "target",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("DATASET METADATA MODULE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()