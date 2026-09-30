from functools import lru_cache
from pathlib import Path

import torch

from training.model_cnn_transformer import CNNTransformer
from training.preprocess import load_and_preprocess_audio
from training.features import extract_log_mel_spectrogram
from training.config import DEVICE


# ============================================================
# MODEL LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "cnn_transformer_final.pth"
)


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = {
    0: "REAL",
    1: "AI-GENERATED",
}


# ============================================================
# LOAD AND REUSE THE TRAINED MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """
    Load and reuse one model instance in each Python process.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = CNNTransformer(
        num_classes=2
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# DETAILED PREDICTION
# ============================================================

def predict_audio_details(audio_path):
    """
    Return:
        prediction: REAL or AI-GENERATED
        confidence: percentage from 0 to 100
        probabilities: both class scores from 0 to 1
    """

    model = load_model()

    # Use the existing audio preprocessing.
    waveform, sample_rate = load_and_preprocess_audio(
        audio_path
    )

    # Use the existing feature extraction.
    log_mel = extract_log_mel_spectrogram(
        waveform,
        sample_rate,
    )

    features = torch.tensor(
        log_mel,
        dtype=torch.float32,
    )

    # Same input layout as the existing prediction function.
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
            dim=1,
        )

        if not torch.isfinite(
            probabilities
        ).all().item():
            raise ValueError(
                "The model returned invalid class scores."
            )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

    class_scores = {
        name: probabilities[0, index].item()
        for index, name in CLASS_NAMES.items()
    }

    prediction = CLASS_NAMES[predicted_class]

    return {
        "prediction": prediction,
        "confidence": class_scores[prediction] * 100.0,
        "probabilities": class_scores,
    }


# ============================================================
# EXISTING PREDICTION INTERFACE
# ============================================================

def predict_audio(audio_path):
    """
    Keep the existing return format:
        prediction, confidence
    """

    result = predict_audio_details(
        audio_path
    )

    return (
        result["prediction"],
        result["confidence"],
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("SINGLE AUDIO PREDICTION")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    audio_path = input(
        "\nEnter the path of an audio file: "
    ).strip()

    if not audio_path:
        raise ValueError(
            "Audio file path cannot be empty."
        )

    result = predict_audio_details(
        audio_path
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print("\nPrediction:", result["prediction"])

    print(
        f"Confidence: {result['confidence']:.2f}%"
    )

    print(
        "Class scores:",
        result["probabilities"],
    )