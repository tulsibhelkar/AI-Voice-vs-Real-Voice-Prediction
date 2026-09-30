import torch
import librosa
import numpy as np

from training.predict import load_model
from training.features import extract_log_mel_spectrogram
from training.config import DEVICE
from training.preprocess import SAMPLE_RATE, TARGET_LENGTH

AUDIO_PATH = "data/challenge/unseen_ai_01.mp3"

model = load_model()

waveform, original_sr = librosa.load(
    AUDIO_PATH,
    sr=SAMPLE_RATE,
    mono=True
)

print("=" * 60)
print("MULTI-SEGMENT AI AUDIO TEST")
print("=" * 60)

print("\nOriginal processed duration:")
print(f"{len(waveform) / SAMPLE_RATE:.2f} seconds")

segments = [
    ("0-4 sec", 0),
    ("2-6 sec", 2),
    ("4-8 sec", 4),
    ("6-10 sec", 6),
]

for name, start_second in segments:

    start = int(start_second * SAMPLE_RATE)
    end = start + TARGET_LENGTH

    segment = waveform[start:end]

    if len(segment) < TARGET_LENGTH:
        segment = np.pad(
            segment,
            (0, TARGET_LENGTH - len(segment)),
            mode="constant"
        )

    peak = np.max(np.abs(segment))

    if peak > 0:
        segment = segment / peak

    log_mel = extract_log_mel_spectrogram(
        segment.astype(np.float32),
        SAMPLE_RATE
    )

    features = torch.tensor(
        log_mel,
        dtype=torch.float32
    )

    features = (
        features
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)[0]

    real_probability = probabilities[0].item() * 100
    ai_probability = probabilities[1].item() * 100

    print("\n" + "-" * 60)
    print(name)
    print(f"REAL:          {real_probability:.2f}%")
    print(f"AI-GENERATED:  {ai_probability:.2f}%")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)