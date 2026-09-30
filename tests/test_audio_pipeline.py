import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.preprocess import (
    SAMPLE_RATE,
    TARGET_LENGTH,
    load_and_preprocess_audio,
)

from training.features import extract_log_mel_spectrogram


def create_test_audio(output_path: Path) -> None:
    """
    Create a temporary 3-second sine-wave audio file.

    This file is used only to test whether the audio-processing
    pipeline works correctly.
    """

    duration = 3.0
    frequency = 440.0

    time = np.linspace(
        0,
        duration,
        int(SAMPLE_RATE * duration),
        endpoint=False,
    )

    waveform = 0.5 * np.sin(
        2 * np.pi * frequency * time
    )

    sf.write(
        output_path,
        waveform,
        SAMPLE_RATE,
    )


def main() -> None:

    temp_audio_path = PROJECT_ROOT / "tests" / "temporary_test_audio.wav"

    print("=" * 55)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("AUDIO PIPELINE TEST")
    print("=" * 55)

    print("\n[1] Creating temporary test audio...")

    create_test_audio(temp_audio_path)

    print("Test audio created successfully.")

    print("\n[2] Running audio preprocessing...")

    waveform, sample_rate = load_and_preprocess_audio(
        str(temp_audio_path)
    )

    print("Audio preprocessing successful.")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Waveform samples: {len(waveform)}")

    if sample_rate != SAMPLE_RATE:
        raise AssertionError(
            f"Expected sample rate {SAMPLE_RATE}, got {sample_rate}"
        )

    if len(waveform) != TARGET_LENGTH:
        raise AssertionError(
            f"Expected {TARGET_LENGTH} samples, got {len(waveform)}"
        )

    print("\n[3] Extracting Log-Mel Spectrogram...")

    log_mel = extract_log_mel_spectrogram(
        waveform,
        sample_rate,
    )

    print("Log-Mel Spectrogram generated successfully.")
    print(f"Spectrogram shape: {log_mel.shape}")
    print(f"Data type: {log_mel.dtype}")
    print(f"Minimum value: {log_mel.min():.2f} dB")
    print(f"Maximum value: {log_mel.max():.2f} dB")

    if log_mel.shape[0] != 80:
        raise AssertionError(
            f"Expected 80 Mel bands, got {log_mel.shape[0]}"
        )

    print("\n" + "=" * 55)
    print("AUDIO PIPELINE TEST PASSED")
    print("=" * 55)

    # Remove temporary audio after successful test.
    if temp_audio_path.exists():
        temp_audio_path.unlink()

    print("\nTemporary test audio removed.")


if __name__ == "__main__":
    main()