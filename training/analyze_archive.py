from pathlib import Path
import tarfile

from training.protocol import load_training_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "flac_T_aa.tar"
)


def get_audio_ids_from_archive(archive_path: Path) -> set[str]:
    """
    Read a TAR archive without extracting it and collect
    the IDs of all FLAC audio files.
    """

    audio_ids = set()

    with tarfile.open(archive_path, mode="r") as archive:

        for member in archive.getmembers():

            if not member.isfile():
                continue

            member_path = Path(member.name)

            if member_path.suffix.lower() != ".flac":
                continue

            audio_ids.add(member_path.stem)

    return audio_ids


def main() -> None:

    print("=" * 65)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("ASVSPOOF 5 AUDIO ARCHIVE ANALYZER")
    print("=" * 65)

    if not ARCHIVE_PATH.exists():

        print("\nArchive is not available yet.")
        print("Expected location:")
        print(ARCHIVE_PATH)

        print(
            "\nThis is normal if flac_T_aa.tar "
            "is still downloading."
        )

        return

    print("\nArchive found:")
    print(ARCHIVE_PATH)

    archive_size_gb = (
        ARCHIVE_PATH.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"\nArchive size: "
        f"{archive_size_gb:.2f} GB"
    )

    print("\nReading FLAC filenames from archive...")
    print("The archive is NOT being extracted.")

    audio_ids = get_audio_ids_from_archive(
        ARCHIVE_PATH
    )

    print(
        f"\nFLAC files discovered: "
        f"{len(audio_ids):,}"
    )

    print("\nLoading official ASVspoof protocol...")

    protocol = load_training_protocol()

    matched = protocol[
        protocol["audio_id"].isin(audio_ids)
    ].copy()

    print("\n" + "-" * 65)
    print("PROTOCOL MATCHING")
    print("-" * 65)

    print(
        f"Archive audio IDs: "
        f"{len(audio_ids):,}"
    )

    print(
        f"Matched protocol records: "
        f"{len(matched):,}"
    )

    unmatched_count = (
        len(audio_ids)
        - len(matched)
    )

    print(
        f"Unmatched archive IDs: "
        f"{unmatched_count:,}"
    )

    if matched.empty:

        print(
            "\nNo archive filenames matched "
            "the training protocol."
        )

        print(
            "We must inspect the archive naming "
            "format before continuing."
        )

        return

    print("\n" + "-" * 65)
    print("CLASS DISTRIBUTION IN THIS ARCHIVE")
    print("-" * 65)

    print(
        matched["label"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 65)
    print("SPEAKER INFORMATION")
    print("-" * 65)

    print(
        f"Unique speakers: "
        f"{matched['speaker_id'].nunique():,}"
    )

    print("\nGender distribution:")

    print(
        matched["gender"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 65)
    print("ATTACK GROUP DISTRIBUTION")
    print("-" * 65)

    print(
        matched["attack_group"]
        .value_counts()
        .to_string()
    )

    print("\n" + "-" * 65)
    print("FIRST 10 MATCHED RECORDINGS")
    print("-" * 65)

    print(
        matched[
            [
                "audio_id",
                "speaker_id",
                "gender",
                "attack_group",
                "label",
                "target",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("ASVSPOOF ARCHIVE ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()