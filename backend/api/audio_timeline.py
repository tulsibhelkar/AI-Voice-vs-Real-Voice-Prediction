import numpy as np


SAMPLE_RATE = 16000

# Each model input contains four seconds of audio.
WINDOW_SECONDS = 4.0

# Start a new window every two seconds.
STEP_SECONDS = 2.0

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)


def create_timeline_windows(waveform: np.ndarray):
    """
    Split mono, 16 kHz audio into overlapping four-second windows.
    """
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional mono waveform."
        )

    if waveform.size == 0:
        raise ValueError(
            "The audio contains no usable samples."
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            "The audio contains invalid sample values."
        )

    windows = []
    total_samples = len(waveform)
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(
            start_sample + WINDOW_SAMPLES,
            total_samples,
        )

        audio = waveform[start_sample:end_sample]

        padding_samples = WINDOW_SAMPLES - len(audio)

        # Pad a short final window to four seconds.
        # Its displayed end time remains the actual recording end.
        audio = np.pad(
            audio,
            (0, padding_samples),
            mode="constant",
        )

        windows.append({
            "segment_id": len(windows) + 1,
            "start_seconds": start_sample / SAMPLE_RATE,
            "end_seconds": end_sample / SAMPLE_RATE,
            "valid_duration_seconds": (
                end_sample - start_sample
            ) / SAMPLE_RATE,
            "is_padded": padding_samples > 0,
            "audio": audio,
        })

        # Stop when the final window covers the recording's end.
        if end_sample == total_samples:
            break

        start_sample += STEP_SAMPLES

    return windows


def prepare_audio_timeline(audio_path: str):
    """
    Load the full recording and preserve its original timestamps.
    """
    import librosa

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    # Keep silence and the complete recording so timestamps stay aligned.
    # We will connect model-specific preprocessing in the next step.
    windows = create_timeline_windows(waveform)

    return {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": len(waveform) / SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "segment_count": len(windows),
        "segments": windows,
    }
def predict_audio_timeline(audio_path: str):
    """
    Predict each window and return a JSON-ready timeline.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import soundfile as sf

    from training.predict import (
        predict_audio_details,
        CLASS_NAMES,
    )

    timeline = prepare_audio_timeline(audio_path)

    results = []

    # Each request gets its own temporary directory.
    with TemporaryDirectory(
        prefix="voiceguard_timeline_"
    ) as temp_dir:

        segment_path = Path(temp_dir) / "segment.wav"

        for window in timeline["segments"]:

            # Copy timestamps and metadata, excluding the audio array.
            segment = {
                key: value
                for key, value in window.items()
                if key != "audio"
            }

            valid_samples = round(
                window["valid_duration_seconds"]
                * timeline["sample_rate"]
            )

            # Pass the actual clip to the existing preprocessor.
            # It will handle the model's required padding.
            audio = window["audio"][:valid_samples]

            # Detect all-zero samples only.
            # This is not a speech detection system.
            if not np.any(audio):

                segment.update({
                    "status": "no_signal",
                    "prediction": "NO_SIGNAL",
                    "confidence": None,
                    "probabilities": None,
                })

                results.append(segment)
                continue

            # FLOAT WAV preserves the waveform without integer
            # quantization. The existing prediction function applies
            # the same preprocessing used for normal predictions.
            sf.write(
                str(segment_path),
                audio,
                timeline["sample_rate"],
                format="WAV",
                subtype="FLOAT",
            )

            prediction = predict_audio_details(
                str(segment_path)
            )

            segment.update({
                "status": "analyzed",
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
            })

            results.append(segment)

    # Temporary files have now been removed.

    analyzed = [
        segment
        for segment in results
        if segment["status"] == "analyzed"
    ]

    if analyzed:

        # Baseline aggregation: average each class's scores
        # across all analyzed windows.
        mean_scores = {
            name: float(
                np.mean([
                    segment["probabilities"][name]
                    for segment in analyzed
                ])
            )
            for name in CLASS_NAMES.values()
        }

        overall_prediction = max(
            mean_scores,
            key=mean_scores.get,
        )

        summary = {
            "prediction": overall_prediction,
            "confidence": (
                mean_scores[overall_prediction] * 100.0
            ),
            "probabilities": mean_scores,
        }

    else:

        summary = {
            "prediction": "NO_SIGNAL",
            "confidence": None,
            "probabilities": None,
        }

    return {
        "sample_rate": timeline["sample_rate"],
        "duration_seconds": timeline["duration_seconds"],
        "window_seconds": timeline["window_seconds"],
        "step_seconds": timeline["step_seconds"],
        "segment_count": len(results),
        "analyzed_segment_count": len(analyzed),
        "aggregation": "mean_window_class_scores",
        "summary": summary,
        "segments": results,
    }