from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from training.preprocess import SAMPLE_RATE
from training.features import extract_log_mel_spectrogram


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
SPECTROGRAM_DIR = OUTPUT_DIR / "spectrograms"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)


def save_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    filename: str = "test_waveform.png"
) -> Path:

    output_path = PLOTS_DIR / filename

    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(
        waveform,
        sr=sample_rate
    )

    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def save_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    filename: str = "test_log_mel_spectrogram.png"
) -> Path:

    log_mel = extract_log_mel_spectrogram(
        waveform,
        sample_rate
    )

    output_path = SPECTROGRAM_DIR / filename

    plt.figure(figsize=(12, 5))

    librosa.display.specshow(
        log_mel,
        sr=sample_rate,
        hop_length=256,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel Frequency")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_demo_audio() -> np.ndarray:

    duration = 4.0
    frequency = 440.0

    time = np.linspace(
        0,
        duration,
        int(SAMPLE_RATE * duration),
        endpoint=False
    )

    waveform = 0.5 * np.sin(
        2 * np.pi * frequency * time
    )

    return waveform.astype(np.float32)


def main():

    print("=" * 55)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("AUDIO VISUALIZATION TEST")
    print("=" * 55)

    print("\nCreating temporary demo waveform...")

    waveform = create_demo_audio()

    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Waveform samples: {len(waveform)}")

    print("\nGenerating waveform visualization...")

    waveform_path = save_waveform(
        waveform,
        SAMPLE_RATE
    )

    print("Waveform saved successfully.")
    print(f"Saved to: {waveform_path}")

    print("\nGenerating Log-Mel Spectrogram...")

    spectrogram_path = save_spectrogram(
        waveform,
        SAMPLE_RATE
    )

    print("Spectrogram saved successfully.")
    print(f"Saved to: {spectrogram_path}")

    print("\n" + "=" * 55)
    print("AUDIO VISUALIZATION TEST PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()