from datasets import load_dataset
from torch.utils.data import Dataset

from training.preprocess import SAMPLE_RATE, TARGET_LENGTH
from training.features import extract_log_mel_spectrogram

import librosa
import numpy as np


# Hugging Face dataset
DATASET_NAME = "garystafford/deepfake-audio-detection"


def preprocess_dataset_audio(
    waveform: np.ndarray,
    sample_rate: int
) -> np.ndarray:
    """
    Preprocess audio coming directly from the Hugging Face dataset.

    Steps:
    1. Convert to NumPy float32
    2. Convert to mono
    3. Resample to 16 kHz
    4. Trim silence
    5. Normalize amplitude
    6. Pad/crop to exactly 4 seconds
    """

    # Convert to NumPy array
    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    # Convert stereo/multi-channel audio to mono
    if waveform.ndim > 1:
        waveform = np.mean(
            waveform,
            axis=0
        )

    # Resample to 16 kHz
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    # Remove leading and trailing silence
    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=30
    )

    # Check audio
    if waveform.size == 0:
        raise ValueError(
            "No usable audio remained after silence trimming."
        )

    # Peak normalization
    peak = np.max(
        np.abs(waveform)
    )

    if peak > 0:
        waveform = waveform / peak

    # Pad if shorter than 4 seconds
    if len(waveform) < TARGET_LENGTH:

        padding = TARGET_LENGTH - len(waveform)

        waveform = np.pad(
            waveform,
            (0, padding),
            mode="constant"
        )

    # Crop if longer than 4 seconds
    elif len(waveform) > TARGET_LENGTH:

        waveform = waveform[
            :TARGET_LENGTH
        ]

    return waveform.astype(
        np.float32
    )


class VoiceDataset(Dataset):

    def __init__(self, split="train"):

        self.dataset = load_dataset(
            DATASET_NAME,
            split=split
        )

    def __len__(self):

        return len(
            self.dataset
        )

    def __getitem__(self, index):

        # Get one dataset sample
        sample = self.dataset[index]

        # Get audio information
        audio = sample["audio"]

        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]

        # Preprocess audio
        waveform = preprocess_dataset_audio(
            waveform,
            sample_rate
        )

        # Extract Log-Mel Spectrogram
        log_mel = extract_log_mel_spectrogram(
            waveform,
            SAMPLE_RATE
        )

        # Dataset label:
        # 0 = REAL
        # 1 = FAKE / AI-GENERATED
        label = int(
            sample["label"]
        )

        return log_mel, label


# ---------------------------------------------------------
# TEST DATASET LOADER
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = VoiceDataset(
        split="train"
    )

    print("=" * 60)
    print("VOICE DATASET TEST")
    print("=" * 60)

    print(
        "Dataset size:",
        len(dataset)
    )

    # Load first sample
    features, label = dataset[0]

    print(
        "Feature shape:",
        features.shape
    )

    print(
        "Feature dtype:",
        features.dtype
    )

    print(
        "Label:",
        label
    )

    print("\nDATASET LOADER TEST PASSED")