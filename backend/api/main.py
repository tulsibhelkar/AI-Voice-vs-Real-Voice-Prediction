

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import shutil
from pathlib import Path

from training.predict import predict_audio
from backend.api.audio_analysis import analyze_audio_visuals
from backend.database.database import (
    create_table,
    save_prediction,
    get_predictions,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="VoiceGuard AI API",
    description="AI Voice vs Real Voice Detection API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

create_table()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "VoiceGuard AI API is running",
        "status": "online",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.get("/history")
def prediction_history():

    rows = get_predictions()

    return {
        "predictions": [
            {
                "id": row[0],
                "filename": row[1],
                "prediction": row[2],
                "confidence": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]
    }


# ============================================================
# VOICE PREDICTION
# ============================================================

@app.post("/predict")
async def predict_voice(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Supported audio formats
    # --------------------------------------------------------

    allowed_extensions = {
        ".wav",
        ".flac",
        ".mp3",
    }

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, FLAC, and MP3 files are supported.",
        )

    # --------------------------------------------------------
    # Create safe temporary file path
    # --------------------------------------------------------

    safe_filename = Path(file.filename).name

    file_path = UPLOAD_DIR / safe_filename

    try:

        # ----------------------------------------------------
        # Save uploaded audio temporarily
        # ----------------------------------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # ----------------------------------------------------
        # Run trained AI model
        # ----------------------------------------------------

        prediction, confidence = predict_audio(
            str(file_path)
        )

        confidence = round(
            confidence,
            2,
        )

        # ----------------------------------------------------
        # Save prediction to SQLite
        # ----------------------------------------------------

        save_prediction(
            safe_filename,
            prediction,
            confidence,
        )

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return {
            "filename": safe_filename,
            "prediction": prediction,
            "confidence": confidence,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded file
        # ----------------------------------------------------

        if file_path.exists():
            file_path.unlink()
# ============================================================
# AUDIO VISUAL ANALYSIS
# ============================================================

@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Supported audio formats
    # --------------------------------------------------------

    allowed_extensions = {
        ".wav",
        ".flac",
        ".mp3",
    }

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, FLAC, and MP3 files are supported.",
        )

    # --------------------------------------------------------
    # Create safe temporary file path
    # --------------------------------------------------------

    safe_filename = Path(file.filename).name

    file_path = UPLOAD_DIR / safe_filename

    try:

        # ----------------------------------------------------
        # Save uploaded audio temporarily
        # ----------------------------------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # ----------------------------------------------------
        # Analyze audio for visualization
        # ----------------------------------------------------

        analysis = analyze_audio_visuals(
            str(file_path)
        )

        # ----------------------------------------------------
        # Return visualization analysis
        # ----------------------------------------------------

        return {
            "filename": safe_filename,
            "analysis": analysis,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded file
        # ----------------------------------------------------

        if file_path.exists():
            file_path.unlink()
# ============================================================
# PREDICTION TIMELINE
# ============================================================

from tempfile import TemporaryDirectory

from backend.api.audio_timeline import predict_audio_timeline


MAX_TIMELINE_UPLOAD_BYTES = 25 * 1024 * 1024


@app.post("/predict-timeline")
def predict_timeline(
    file: UploadFile = File(...)
):
    """
    Upload audio and return predictions for its timed segments.
    """

    try:

        # Validate filename.
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided.",
            )

        safe_filename = Path(
            file.filename.replace("\\", "/")
        ).name

        extension = Path(
            safe_filename
        ).suffix.lower()

        # Validate format.
        if extension not in {
            ".wav",
            ".flac",
            ".mp3",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only WAV, FLAC, and MP3 files "
                    "are supported."
                ),
            )

        # Each upload receives a separate temporary directory.
        with TemporaryDirectory(
            prefix="voiceguard_upload_"
        ) as temp_dir:

            file_path = (
                Path(temp_dir)
                / f"audio{extension}"
            )

            total_bytes = 0

            # Save the upload in chunks.
            with file_path.open("wb") as buffer:

                while True:

                    chunk = file.file.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > MAX_TIMELINE_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Audio file must be "
                                "25 MB or smaller."
                            ),
                        )

                    buffer.write(chunk)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is empty.",
                )

            # Run predictions across the recording.
            timeline = predict_audio_timeline(
                str(file_path)
            )

        # The temporary upload has now been removed.

        # Save one overall prediction per upload.
        summary = timeline["summary"]

        if summary["prediction"] in {
            "REAL",
            "AI-GENERATED",
        }:
            save_prediction(
                safe_filename,
                summary["prediction"],
                round(
                    summary["confidence"],
                    2,
                ),
            )

        return {
            "filename": safe_filename,
            **timeline,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Timeline analysis failed: {error}"
            ),
        ) from error

    finally:
        file.file.close()