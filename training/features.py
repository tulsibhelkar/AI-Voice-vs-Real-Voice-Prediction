import numpy as np
import librosa

from training.preprocess import SAMPLE_RATE


# ============================================================
# Log-Mel Spectrogram Settings
# ============================================================

N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# Log-Mel Spectrogram Extraction
# ============================================================

def extract_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Convert an audio waveform into a Log-Mel Spectrogram.

    Parameters
    ----------
    waveform : np.ndarray
        1D processed audio waveform.

    sample_rate : int
        Sampling rate of the waveform.

    Returns
    -------
    np.ndarray
        Log-Mel Spectrogram with shape:
        (80, 251)
    """

    # Check waveform dimensions
    if waveform.ndim != 1:
        raise ValueError(
            "Waveform must be a 1D NumPy array."
        )

    # Check empty waveform
    if waveform.size == 0:
        raise ValueError(
            "Waveform is empty."
        )

    # Make sure waveform is float32
    waveform = waveform.astype(
        np.float32
    )

    # Create Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Convert power spectrogram to decibels
    log_mel = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )

    # Return float32
    return log_mel.astype(
        np.float32
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("LOG-MEL FEATURE EXTRACTION")
    print("=" * 60)

    print(
        "Mel bands:",
        N_MELS
    )

    print(
        "FFT size:",
        N_FFT
    )

    print(
        "Hop length:",
        HOP_LENGTH
    )

    print(
        "Sample rate:",
        SAMPLE_RATE
    )

    print("\nFEATURE EXTRACTION MODULE LOADED SUCCESSFULLY")