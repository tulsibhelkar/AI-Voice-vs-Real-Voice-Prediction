import torch
import librosa
import numpy as np

from training.predict import load_model
from training.features import extract_log_mel_spectrogram
from training.config import DEVICE
from training.preprocess import SAMPLE_RATE, TARGET_LENGTH


MODEL_PATH = "saved_models/cnn_transformer_final.pth"

model = load_model()


def analyze_audio(audio_path, name):

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=30
    )

    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    if len(waveform) < TARGET_LENGTH:
        waveform = np.pad(
            waveform,
            (0, TARGET_LENGTH - len(waveform)),
            mode="constant"
        )
    else:
        waveform = waveform[:TARGET_LENGTH]

    log_mel = extract_log_mel_spectrogram(
        waveform.astype(np.float32),
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
        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    real_probability = probabilities[0].item() * 100
    ai_probability = probabilities[1].item() * 100

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Duration: {len(waveform) / SAMPLE_RATE:.2f} sec")
    print(f"Log-Mel shape: {log_mel.shape}")
    print(f"REAL: {real_probability:.2f}%")
    print(f"AI-GENERATED: {ai_probability:.2f}%")


print("=" * 60)
print("AI SAMPLE COMPARISON")
print("=" * 60)

analyze_audio(
    "data/challenge/unseen_ai_01.mp3",
    "YOUR AI-GENERATED WHATSAPP AUDIO"
)

print("\nComparison complete.")