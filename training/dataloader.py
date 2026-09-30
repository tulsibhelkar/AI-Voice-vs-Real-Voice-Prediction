import json
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset, DataLoader

from training.config import BATCH_SIZE
from training.dataset import preprocess_dataset_audio
from training.features import extract_log_mel_spectrogram
from training.preprocess import SAMPLE_RATE


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "garystafford/deepfake-audio-detection"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


def decode_audio(audio):

    if not isinstance(audio, dict):
        raise TypeError(
            "Expected raw audio dictionary."
        )

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes is not None:

        if not audio_bytes:
            raise ValueError(
                "Audio payload is empty."
            )

        source = BytesIO(audio_bytes)

    elif audio_path:

        source_path = Path(audio_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {source_path}"
            )

        source = str(source_path)

    else:
        raise ValueError(
            "Audio row contains no bytes or path."
        )

    waveform, sample_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True
    )

    if waveform.shape[0] == 0:
        raise ValueError(
            "Decoded audio is empty."
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            "Audio contains invalid values."
        )

    # Convert stereo audio to mono.
    waveform = waveform.mean(
        axis=1,
        dtype=np.float32
    )

    return waveform, int(sample_rate)


class VoiceDataLoaderDataset(Dataset):

    def __init__(self, split_name):

        split_file = (
            SPLIT_DIR /
            f"{split_name}_indices.json"
        )

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as file:
            self.indices = json.load(file)

        self.dataset = load_dataset(
            DATASET_NAME,
            split="train"
        ).cast_column(
            "audio",
            Audio(decode=False)
        )

        self.split_name = split_name

        print(
            f"{split_name.upper()} dataset loaded: "
            f"{len(self.indices)} samples"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        dataset_index = self.indices[position]

        sample = self.dataset[dataset_index]

        waveform, sample_rate = decode_audio(
            sample["audio"]
        )

        waveform = preprocess_dataset_audio(
            waveform,
            sample_rate
        )

        log_mel = extract_log_mel_spectrogram(
            waveform,
            SAMPLE_RATE
        )

        features = torch.tensor(
            log_mel,
            dtype=torch.float32
        )

        features = features.unsqueeze(0)

        label = torch.tensor(
            int(sample["label"]),
            dtype=torch.long
        )

        return features, label


def create_dataloader(
    split_name,
    shuffle=False
):

    dataset = VoiceDataLoaderDataset(
        split_name
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )