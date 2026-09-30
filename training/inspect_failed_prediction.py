import torch

from training.predict import load_model
from training.preprocess import load_and_preprocess_audio
from training.features import extract_log_mel_spectrogram
from training.config import DEVICE


AUDIO_PATH = "data/challenge/unseen_ai_01.mp3"


waveform, sample_rate = load_and_preprocess_audio(AUDIO_PATH)

log_mel = extract_log_mel_spectrogram(
    waveform,
    sample_rate
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


model = load_model()
model.eval()


with torch.no_grad():

    outputs = model(features)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]


print("=" * 60)
print("FAILED AI SAMPLE MODEL INSPECTION")
print("=" * 60)

print("\nRaw logits:")
print(outputs[0].cpu().tolist())

print("\nREAL probability:")
print(f"{probabilities[0].item() * 100:.4f}%")

print("\nAI-GENERATED probability:")
print(f"{probabilities[1].item() * 100:.4f}%")