"""Bounded window inference and estimated REAL/AI timeline regions."""

import math
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


SAMPLE_RATE = 16000
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)

MAX_AUDIO_SECONDS = 30 * 60
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

# Provisional decision rules, not measured accuracy or calibrated certainty.
MIN_CLASS_SCORE = 0.80
MIN_REGION_SECONDS = 4.0
CLASS_NAMES = ("REAL", "AI-GENERATED")
REGION_LABELS = (*CLASS_NAMES, "UNCERTAIN", "NO_SIGNAL")


def _window_bounds(total_samples, sample_rate):
    width = int(WINDOW_SECONDS * sample_rate)
    step = int(STEP_SECONDS * sample_rate)
    start = 0

    while start < total_samples:
        end = min(start + width, total_samples)
        yield start, end

        if end == total_samples:
            break

        start += step


def _window(audio, start_seconds, end_seconds, segment_id):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or not audio.size:
        raise ValueError("Expected non-empty mono audio.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid sample values.")

    valid_samples = min(len(audio), WINDOW_SAMPLES)

    return {
        "segment_id": segment_id,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "valid_duration_seconds": end_seconds - start_seconds,
        "valid_sample_count": valid_samples,
        "is_padded": valid_samples < WINDOW_SAMPLES,
        "audio": np.pad(
            audio[:valid_samples],
            (0, WINDOW_SAMPLES - valid_samples),
        ),
    }


def create_timeline_windows(waveform: np.ndarray):
    """Compatibility helper for an already loaded mono, 16 kHz waveform."""
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim != 1 or not waveform.size:
        raise ValueError(
            "Expected non-empty one-dimensional mono audio."
        )

    if len(waveform) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        raise ValueError(
            "Audio duration must be 30 minutes or shorter."
        )

    return [
        _window(
            waveform[start:end],
            start / SAMPLE_RATE,
            end / SAMPLE_RATE,
            index,
        )
        for index, (start, end) in enumerate(
            _window_bounds(len(waveform), SAMPLE_RATE),
            start=1,
        )
    ]


def inspect_audio_file(audio_path):
    """Validate file size and decoded-frame metadata before loading the model."""
    import soundfile as sf

    path = Path(audio_path)

    if not path.is_file():
        raise ValueError(f"Audio file not found: {path}")

    if path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
        raise ValueError(
            "Only WAV, FLAC, and MP3 files are supported."
        )

    size = path.stat().st_size

    if size == 0:
        raise ValueError("The uploaded file is empty.")

    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            "Audio file must be 500 MiB or smaller."
        )

    try:
        with sf.SoundFile(str(path)) as source:
            frames = source.frames
            rate = source.samplerate
            channels = source.channels

    except (RuntimeError, OSError) as error:
        raise ValueError(
            "Cannot decode this audio. Check the file and SoundFile codec support; "
            "try a WAV or FLAC copy if MP3 decoding is unavailable."
        ) from error

    if frames <= 0 or rate <= 0 or channels <= 0:
        raise ValueError(
            "The audio contains no usable samples."
        )

    if frames > MAX_AUDIO_SECONDS * rate:
        raise ValueError(
            f"Audio is {frames / rate / 60:.2f} minutes long. "
            "The maximum allowed duration is 30 minutes."
        )

    return {
        "frames": frames,
        "source_sample_rate": rate,
        "channels": channels,
        "duration_seconds": frames / rate,
    }


def _read_audio_windows(audio_path, info):
    """Decode one four-second window at a time; keep original timestamps."""
    import soundfile as sf

    rate = info["source_sample_rate"]

    with sf.SoundFile(str(audio_path)) as source:
        if (
            source.frames,
            source.samplerate,
            source.channels,
        ) != (
            info["frames"],
            rate,
            info["channels"],
        ):
            raise ValueError(
                "The audio changed during analysis. Upload it again."
            )

        for index, (start, end) in enumerate(
            _window_bounds(source.frames, rate),
            start=1,
        ):
            source.seek(start)

            block = source.read(
                end - start,
                dtype="float32",
                always_2d=True,
            )

            if (
                len(block) != end - start
                or not np.all(np.isfinite(block))
            ):
                raise ValueError(
                    "The audio is incomplete or contains invalid samples."
                )

            # SoundFile returns samples x channels.
            audio = block.mean(axis=1)

            if rate != SAMPLE_RATE:
                import librosa

                audio = librosa.resample(
                    audio,
                    orig_sr=rate,
                    target_sr=SAMPLE_RATE,
                )

            yield _window(
                audio,
                start / rate,
                end / rate,
                index,
            )


def prepare_audio_timeline(audio_path: str):
    """Return metadata and a lazy window iterator, not a full recording in RAM."""
    info = inspect_audio_file(audio_path)
    rate = info["source_sample_rate"]

    count = 1 + math.ceil(
        max(
            0,
            info["frames"] - int(WINDOW_SECONDS * rate),
        )
        / int(STEP_SECONDS * rate)
    )

    return {
        "sample_rate": SAMPLE_RATE,
        "source_sample_rate": rate,
        "duration_seconds": info["duration_seconds"],
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "segment_count": count,
        "segments": _read_audio_windows(audio_path, info),
    }


def _merge_regions(regions):
    merged = []

    for region in regions:
        item = dict(region)

        if (
            merged
            and merged[-1]["prediction"] == item["prediction"]
        ):
            previous = merged[-1]

            old_duration = (
                previous["end_seconds"]
                - previous["start_seconds"]
            )

            new_duration = (
                item["end_seconds"]
                - item["start_seconds"]
            )

            if previous["confidence"] is not None:
                previous["confidence"] = (
                    previous["confidence"] * old_duration
                    + item["confidence"] * new_duration
                ) / (old_duration + new_duration)

            previous["end_seconds"] = item["end_seconds"]

        else:
            merged.append(item)

    return merged


def build_timeline_regions(segments, duration):
    """Partition time once; disagreement between overlapping windows is uncertain."""
    edges = sorted(
        {0.0, duration}
        | {
            time
            for segment in segments
            for time in (
                segment["start_seconds"],
                segment["end_seconds"],
            )
        }
    )

    pieces = []

    for start, end in zip(edges, edges[1:]):
        midpoint = (start + end) / 2

        covering = [
            segment
            for segment in segments
            if (
                segment["start_seconds"]
                <= midpoint
                < segment["end_seconds"]
            )
        ]

        labels = {
            segment["prediction"]
            for segment in covering
        }

        label = (
            next(iter(labels))
            if len(labels) == 1
            else "UNCERTAIN"
        )

        confidence = None

        if label in CLASS_NAMES:
            confidence = float(
                np.mean([
                    segment["probabilities"][label]
                    for segment in covering
                ])
            ) * 100

        pieces.append({
            "start_seconds": start,
            "end_seconds": end,
            "prediction": label,
            "confidence": confidence,
        })

    regions = _merge_regions(pieces)

    # Discard isolated short classifications.
    # Retain a uniformly classified short recording.
    minimum = min(MIN_REGION_SECONDS, duration)

    for region in regions:
        region_duration = (
            region["end_seconds"]
            - region["start_seconds"]
        )

        if (
            region["prediction"] in CLASS_NAMES
            and region_duration < minimum - 1e-9
        ):
            region.update(
                prediction="UNCERTAIN",
                confidence=None,
            )

    regions = _merge_regions(regions)

    for index, region in enumerate(regions, start=1):
        region["region_id"] = index
        region["duration_seconds"] = (
            region["end_seconds"]
            - region["start_seconds"]
        )

    return regions


def summarize_regions(regions, duration):
    seconds = {
        label: 0.0
        for label in REGION_LABELS
    }

    for region in regions:
        seconds[region["prediction"]] += (
            region["duration_seconds"]
        )

    present = {
        label
        for label in CLASS_NAMES
        if seconds[label] > 0
    }

    if len(present) == 2:
        label = "MIXED"

    elif seconds["UNCERTAIN"] > 0:
        label = "UNCERTAIN"

    elif present:
        label = next(iter(present))

    else:
        label = "NO_SIGNAL"

    confidence = None
    probabilities = None

    if label in CLASS_NAMES:
        confidence = sum(
            region["confidence"]
            * region["duration_seconds"]
            for region in regions
            if region["prediction"] == label
        ) / seconds[label]

        confidence = min(
            100.0,
            max(0.0, confidence),
        )

        probabilities = {
            key: (
                confidence / 100
                if key == label
                else 1 - confidence / 100
            )
            for key in CLASS_NAMES
        }

    return {
        "prediction": label,
        "confidence": confidence,
        # MIXED has no trained-class probability.
        "probabilities": probabilities,
        "duration_by_label_seconds": seconds,
        "percentage_by_label": {
            key: value / duration * 100
            for key, value in seconds.items()
        },
        "classified_seconds": sum(
            seconds[key]
            for key in CLASS_NAMES
        ),
    }


def predict_audio_timeline(audio_path: str):
    """Run the existing binary model and derive approximate temporal regions."""

    # Validate length and size before importing or calling the predictor.
    timeline = prepare_audio_timeline(audio_path)

    import soundfile as sf
    from training.predict import predict_audio_details

    results = []

    with (
        closing(timeline["segments"]) as windows,
        TemporaryDirectory(
            prefix="audio_timeline_"
        ) as temp_dir,
    ):
        segment_path = Path(temp_dir) / "segment.wav"

        for window in windows:
            segment = {
                key: value
                for key, value in window.items()
                if key != "audio"
            }

            audio = window["audio"][
                :window["valid_sample_count"]
            ]

            # Detect digital silence only, not absence of speech.
            if not np.any(audio):
                segment.update(
                    status="no_signal",
                    prediction="NO_SIGNAL",
                    confidence=None,
                    probabilities=None,
                )

            else:
                sf.write(
                    str(segment_path),
                    audio,
                    SAMPLE_RATE,
                    format="WAV",
                    subtype="FLOAT",
                )

                # Reuse the existing model and its preprocessing.
                result = predict_audio_details(
                    str(segment_path)
                )

                scores = {
                    key: float(
                        result["probabilities"][key]
                    )
                    for key in CLASS_NAMES
                }

                values = np.array(
                    list(scores.values())
                )

                if (
                    not np.all(np.isfinite(values))
                    or np.any(values < 0)
                    or np.any(values > 1)
                    or not np.isclose(
                        values.sum(),
                        1,
                        atol=1e-4,
                    )
                ):
                    raise ValueError(
                        "The model returned invalid class scores."
                    )

                predicted = max(
                    scores,
                    key=scores.get,
                )

                segment.update(
                    status="analyzed",
                    model_prediction=predicted,
                    prediction=(
                        predicted
                        if scores[predicted] >= MIN_CLASS_SCORE
                        else "UNCERTAIN"
                    ),
                    confidence=scores[predicted] * 100,
                    probabilities=scores,
                )

            results.append(segment)

    duration = timeline["duration_seconds"]

    regions = build_timeline_regions(
        results,
        duration,
    )

    analyzed = [
        segment
        for segment in results
        if segment["status"] == "analyzed"
    ]

    mean_scores = (
        {
            key: float(
                np.mean([
                    segment["probabilities"][key]
                    for segment in analyzed
                ])
            )
            for key in CLASS_NAMES
        }
        if analyzed
        else None
    )

    return {
        **{
            key: value
            for key, value in timeline.items()
            if key != "segments"
        },
        "analyzed_segment_count": len(analyzed),
        "aggregation": "non_overlapping_region_consensus",
        "decision_rules": {
            "minimum_class_score": MIN_CLASS_SCORE,
            "minimum_region_seconds": MIN_REGION_SECONDS,
            "validated": False,
        },
        "summary": summarize_regions(
            regions,
            duration,
        ),
        "mean_window_probabilities": mean_scores,
        "segments": results,
        "regions": regions,
    }