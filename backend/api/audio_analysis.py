import base64
import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 16000
DURATION = 4.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256


def prepare_audio(audio_path: str):
    """
    Load and prepare audio for visualization.
    """

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if waveform.size == 0:
        raise ValueError("Audio file contains no usable samples.")

    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=30,
    )

    if waveform.size == 0:
        raise ValueError(
            "No usable audio remained after silence trimming."
        )

    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    if len(waveform) < TARGET_LENGTH:

        waveform = np.pad(
            waveform,
            (
                0,
                TARGET_LENGTH - len(waveform),
            ),
            mode="constant",
        )

    elif len(waveform) > TARGET_LENGTH:

        waveform = waveform[:TARGET_LENGTH]

    return waveform.astype(np.float32)


def figure_to_base64():
    """
    Convert the current Matplotlib figure into a base64 PNG.
    """

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=140,
    )

    plt.close()

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


def create_waveform_image(
    waveform: np.ndarray,
    sample_rate: int,
):
    """
    Create waveform visualization.
    """

    plt.figure(
        figsize=(10, 3),
    )

    time_axis = np.arange(
        len(waveform)
    ) / sample_rate

    plt.plot(
        time_axis,
        waveform,
        linewidth=0.8,
    )

    plt.title(
        "Audio Waveform"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.tight_layout()

    return figure_to_base64()


def create_log_mel_image(
    waveform: np.ndarray,
    sample_rate: int,
):
    """
    Create Log-Mel spectrogram visualization.
    """

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    plt.figure(
        figsize=(10, 4),
    )

    librosa.display.specshow(
        log_mel,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        "Log-Mel Spectrogram"
    )

    plt.tight_layout()

    return figure_to_base64()


def analyze_audio_visuals(
    audio_path: str,
):
    """
    Generate visual analysis data for an audio file.
    """

    waveform = prepare_audio(
        audio_path
    )

    waveform_image = create_waveform_image(
        waveform,
        SAMPLE_RATE,
    )

    log_mel_image = create_log_mel_image(
        waveform,
        SAMPLE_RATE,
    )

    return {
        "waveform": waveform_image,
        "log_mel": log_mel_image,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "feature_shape": [
            N_MELS,
            251,
        ],
    }