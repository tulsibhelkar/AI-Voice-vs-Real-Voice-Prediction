from pathlib import Path

import librosa
import numpy as np


# Standard audio settings used throughout the project
SAMPLE_RATE = 16000
DURATION = 4.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)


def load_and_preprocess_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load and preprocess an audio file.

    Processing:
    1. Load audio.
    2. Convert to mono.
    3. Resample to 16 kHz.
    4. Trim leading/trailing silence.
    5. Normalize amplitude.
    6. Pad or crop to exactly 4 seconds.

    Returns:
        waveform: NumPy array containing processed audio.
        sample_rate: 16000 Hz.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio as mono and resample directly to 16 kHz.
    waveform, sample_rate = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if waveform.size == 0:
        raise ValueError("The audio file contains no usable samples.")

    # Remove leading and trailing silence.
    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=30
    )

    if waveform.size == 0:
        raise ValueError("No usable audio remained after silence trimming.")

    # Peak normalization.
    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    # Make every recording exactly 4 seconds.
    if len(waveform) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(waveform)

        waveform = np.pad(
            waveform,
            (0, padding),
            mode="constant"
        )

    elif len(waveform) > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]

    waveform = waveform.astype(np.float32)

    return waveform, sample_rate


if __name__ == "__main__":
    print("AI Voice vs Real Voice Prediction")
    print("Audio preprocessing module loaded successfully.")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Samples per audio: {TARGET_LENGTH}")