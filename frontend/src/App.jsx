import { useEffect, useRef, useState } from "react";
import PredictionTimeline from "./PredictionTimeline";
import "./App.css";

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const MAX_FILE_SIZE = 25 * 1024 * 1024;
const ALLOWED_FORMATS = ["wav", "flac", "mp3"];

// ============================================================
// HELPERS
// ============================================================

function readHistory() {
  try {
    const stored = JSON.parse(
      localStorage.getItem("voiceguard-history") || "[]"
    );

    return Array.isArray(stored)
      ? stored
          .filter((item) => item && typeof item === "object")
          .slice(0, 30)
      : [];
  } catch {
    return [];
  }
}

function apiError(data, fallback) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data?.detail)) {
    return (
      data.detail
        .map((item) => item.msg)
        .filter(Boolean)
        .join("; ") || fallback
    );
  }

  return fallback;
}

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}%`
    : "—";
}

// ============================================================
// ICONS
// ============================================================

function Icon({ name, size = 20 }) {
  const icons = {
    wave: (
      <path d="M2 12h2m2-5v10m3-14v18m3-12v6m3-10v14m3-17v20m3-13v6" />
    ),

    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5" />
        <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
      </>
    ),

    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,

    close: <path d="M18 6 6 18M6 6l12 12" />,

    check: <path d="m5 12 4 4L19 6" />,

    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),

    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),

    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5m0-8h.01" />
      </>
    ),
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {icons[name]}
    </svg>
  );
}

// ============================================================
// APPLICATION
// ============================================================

function App() {
  const [activePage, setActivePage] = useState("Analyzer");

  const [selectedFile, setSelectedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState("");

  const [prediction, setPrediction] = useState(null);
  const [timeline, setTimeline] = useState(null);

  const [visuals, setVisuals] = useState(null);
  const [visualError, setVisualError] = useState("");

  const [error, setError] = useState("");
  const [history, setHistory] = useState(readHistory);

  const fileInputRef = useRef(null);
  const audioRef = useRef(null);
  const audioUrlRef = useRef("");
  const requestRef = useRef(null);

  // Save recent results in this browser.
  useEffect(() => {
    try {
      localStorage.setItem(
        "voiceguard-history",
        JSON.stringify(history)
      );
    } catch {
      // Analysis still works if browser storage is unavailable.
    }
  }, [history]);

  // Clean up when the application unmounts.
  useEffect(() => {
    return () => {
      requestRef.current?.abort();
      requestRef.current = null;

      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  function resetResults() {
    setPrediction(null);
    setTimeline(null);
    setVisuals(null);
    setVisualError("");
    setError("");
  }

  // ==========================================================
  // SELECT AUDIO
  // ==========================================================

  function handleFile(file) {
    if (!file || requestRef.current) return;

    const extension = file.name
      .split(".")
      .pop()
      ?.toLowerCase();

    if (!ALLOWED_FORMATS.includes(extension)) {
      setError("Please select a WAV, FLAC, or MP3 audio file.");
      return;
    }

    if (!file.size) {
      setError(
        "This file is empty. Please select another recording."
      );
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError(
        "Please select an audio file of 25 MB or smaller."
      );
      return;
    }

    audioRef.current?.pause();

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }

    const newUrl = URL.createObjectURL(file);

    audioUrlRef.current = newUrl;

    setSelectedFile(file);
    setAudioUrl(newUrl);
    resetResults();
  }

  function removeFile() {
    if (requestRef.current) return;

    audioRef.current?.pause();

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }

    audioUrlRef.current = "";

    setSelectedFile(null);
    setAudioUrl("");
    resetResults();

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  // ==========================================================
  // TIMELINE PREDICTION
  // ==========================================================

  async function analyzeVoice() {
    if (
      !selectedFile ||
      isAnalyzing ||
      requestRef.current
    ) {
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;

    setIsAnalyzing(true);
    setAnalysisStage("Analyzing audio windows…");
    resetResults();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${API_URL}/predict-timeline`,
        {
          method: "POST",
          body: formData,
          signal: controller.signal,
        }
      );

      const data = await response
        .json()
        .catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          apiError(
            data,
            "Timeline analysis could not be completed."
          )
        );
      }

      if (
        !data.summary ||
        !Array.isArray(data.segments) ||
        !data.segments.length
      ) {
        throw new Error(
          "The server returned an incomplete timeline response."
        );
      }

      const summary = data.summary;

      if (
        !["REAL", "AI-GENERATED", "NO_SIGNAL"].includes(
          summary.prediction
        )
      ) {
        throw new Error(
          "The server returned an unrecognized prediction label."
        );
      }

      if (
        summary.prediction !== "NO_SIGNAL" &&
        (
          typeof summary.confidence !== "number" ||
          !Number.isFinite(summary.confidence) ||
          summary.confidence < 0 ||
          summary.confidence > 100
        )
      ) {
        throw new Error(
          "The server returned an invalid model score."
        );
      }

      setTimeline(data);

      setPrediction({
        ...summary,
        filename: data.filename || selectedFile.name,
      });

      // All-zero recordings do not receive a classification.
      if (summary.prediction === "NO_SIGNAL") {
        return;
      }

      const record = {
        id: Date.now(),
        filename: data.filename || selectedFile.name,
        prediction: summary.prediction,
        confidence: summary.confidence,
        segment_count: data.segment_count,
        duration_seconds: data.duration_seconds,
        mode: "timeline",
        date: new Date().toISOString(),
      };

      setHistory((previous) =>
        [record, ...previous].slice(0, 30)
      );

      // ======================================================
      // OPTIONAL AUDIO VISUALIZATIONS
      // ======================================================

      setAnalysisStage("Preparing audio visualizations…");

      try {
        const visualResponse = await fetch(
          `${API_URL}/analyze-audio`,
          {
            method: "POST",
            body: formData,
            signal: controller.signal,
          }
        );

        const visualData = await visualResponse
          .json()
          .catch(() => ({}));

        if (!visualResponse.ok) {
          throw new Error(
            apiError(
              visualData,
              "Audio visualizations are unavailable."
            )
          );
        }

        // Backend structure:
        // { filename, analysis: { waveform, log_mel, ... } }
        const analysis = visualData.analysis;

        if (
          !analysis ||
          (!analysis.waveform && !analysis.log_mel)
        ) {
          throw new Error(
            "The visualization response does not contain images."
          );
        }

        setVisuals(analysis);
      } catch (visualFailure) {
        if (controller.signal.aborted) {
          throw visualFailure;
        }

        setVisualError(
          "The audio images could not be loaded. Your prediction and timeline are available."
        );
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(
          failure instanceof TypeError
            ? "Cannot connect to the analysis service. Check that FastAPI is running on port 8000."
            : failure.message
        );
      }
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsAnalyzing(false);
        setAnalysisStage("");
      }
    }
  }

  // ==========================================================
  // DISPLAY VALUES
  // ==========================================================

  const resultLabel = prediction?.prediction || "";

  const isHuman = resultLabel === "REAL";
  const isAI = resultLabel === "AI-GENERATED";
  const hasSignal = isHuman || isAI;

  const confidence = hasSignal
    ? prediction.confidence
    : null;

  const humanCount = history.filter((item) =>
    ["REAL", "HUMAN"].includes(
      String(item.prediction).toUpperCase()
    )
  ).length;

  const aiCount = history.filter((item) =>
    ["AI", "FAKE", "AI-GENERATED", "AI_GENERATED"].includes(
      String(item.prediction).toUpperCase()
    )
  ).length;

  return (
    <div className="app">
      {/* NAVIGATION */}
      <header className="topbar">
        <div className="topbar-inner">
          <button
            className="brand"
            onClick={() => setActivePage("Analyzer")}
            aria-label="VoiceGuard home"
          >
            <span className="brand-icon">
              <Icon name="wave" size={22} />
            </span>

            <span className="brand-text">
              <strong>VoiceGuard</strong>
              <small>VOICE AUTHENTICITY ANALYSIS</small>
            </span>
          </button>

          <nav
            className="navigation"
            aria-label="Main navigation"
          >
            {["Analyzer", "Dashboard", "History", "Model"].map(
              (page) => (
                <button
                  key={page}
                  className={
                    activePage === page ? "active" : ""
                  }
                  aria-current={
                    activePage === page ? "page" : undefined
                  }
                  onClick={() => setActivePage(page)}
                >
                  {page}
                </button>
              )
            )}
          </nav>
        </div>
      </header>

      <main className="container">
        {/* ANALYZER PAGE */}
        {activePage === "Analyzer" && (
          <>
            <section className="hero">
              <div className="hero-content">
                <div className="eyebrow">
                  <span className="eyebrow-dot" />
                  VOICE AUTHENTICITY DETECTION
                </div>

                <h1>
                  AI Voice or
                  <br />
                  <span>Human Voice?</span>
                </h1>

                <p>
                  Analyze a voice recording, review its overall
                  classification, and explore predictions across
                  an interactive timeline.
                </p>

                <div className="hero-points">
                  <span>
                    <Icon name="wave" size={17} />
                    Audio analysis
                  </span>

                  <span>
                    <Icon name="clock" size={17} />
                    Prediction timeline
                  </span>
                </div>
              </div>

              <div
                className="hero-visual"
                aria-hidden="true"
              >
                <div className="visual-label">
                  AUDIO SIGNAL
                </div>

                <div className="wave-bars">
                  {[
                    22, 36, 18, 52, 76, 34, 58, 92, 45, 72,
                    29, 62, 86, 41, 69, 31, 56, 98, 48, 73,
                    38, 84, 51, 24, 64, 42, 79, 30, 57, 20,
                  ].map((height, index) => (
                    <span
                      key={index}
                      style={{ height: `${height}%` }}
                    />
                  ))}
                </div>

                <div className="visual-footer">
                  <span>INPUT SIGNAL</span>
                  <span>→</span>
                  <span>MODEL ANALYSIS</span>
                </div>
              </div>
            </section>

            {/* UPLOAD AREA */}
            <div className="workspace-grid">
              <section className="upload-card">
                <div className="card-heading">
                  <div>
                    <span className="section-label">
                      01 / UPLOAD
                    </span>

                    <h2>Analyze your recording</h2>
                    <p>Select a voice clip to begin.</p>
                  </div>

                  <span className="file-types">
                    WAV · FLAC · MP3
                  </span>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  className="visually-hidden"
                  aria-label="Select an audio recording"
                  accept=".wav,.flac,.mp3,audio/wav,audio/flac,audio/mpeg"
                  disabled={isAnalyzing}
                  onChange={(event) => {
                    handleFile(event.target.files?.[0]);
                    event.target.value = "";
                  }}
                />

                {!selectedFile ? (
                  <div
                    className={`drop-area ${
                      isDragging ? "dragging" : ""
                    }`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setIsDragging(true);
                    }}
                    onDragLeave={(event) => {
                      event.preventDefault();
                      setIsDragging(false);
                    }}
                    onDrop={(event) => {
                      event.preventDefault();
                      setIsDragging(false);
                      handleFile(
                        event.dataTransfer.files?.[0]
                      );
                    }}
                  >
                    <div className="upload-icon">
                      <Icon name="upload" size={27} />
                    </div>

                    <h3>Drop an audio file here</h3>
                    <p>or choose a file from your computer</p>

                    <button
                      className="browse-button"
                      onClick={() =>
                        fileInputRef.current?.click()
                      }
                    >
                      Browse files
                      <Icon name="arrow" size={17} />
                    </button>

                    <small>Maximum file size: 25 MB</small>
                  </div>
                ) : (
                  <div className="selected-audio">
                    <div className="selected-audio-header">
                      <div className="audio-file-icon">
                        <Icon name="wave" size={23} />
                      </div>

                      <div className="audio-file-info">
                        <strong title={selectedFile.name}>
                          {selectedFile.name}
                        </strong>

                        <span>
                          {(
                            selectedFile.size /
                            (1024 * 1024)
                          ).toFixed(2)}
                          {" MB · "}
                          {isAnalyzing
                            ? "Analysis in progress"
                            : timeline
                              ? "Analysis complete"
                              : "Ready for analysis"}
                        </span>
                      </div>

                      <button
                        className="remove-button"
                        onClick={removeFile}
                        disabled={isAnalyzing}
                        aria-label="Remove audio file"
                      >
                        <Icon name="close" size={18} />
                      </button>
                    </div>

                    {/* Shared player used by the timeline */}
                    <audio
                      ref={audioRef}
                      controls
                      src={audioUrl}
                      preload="metadata"
                      aria-label="Audio recording playback"
                    >
                      Your browser does not support audio playback.
                    </audio>

                    <div className="audio-actions">
                      <button
                        className="change-button"
                        onClick={() =>
                          fileInputRef.current?.click()
                        }
                        disabled={isAnalyzing}
                      >
                        Choose another file
                      </button>

                      <button
                        className="analyze-button"
                        onClick={analyzeVoice}
                        disabled={isAnalyzing}
                      >
                        {isAnalyzing ? (
                          <>
                            <span className="spinner" />
                            Analyzing…
                          </>
                        ) : (
                          <>
                            Analyze voice
                            <Icon name="arrow" size={18} />
                          </>
                        )}
                      </button>
                    </div>

                    {isAnalyzing && (
                      <p
                        className="model-warning"
                        role="status"
                      >
                        {analysisStage}
                      </p>
                    )}
                  </div>
                )}

                {error && (
                  <div
                    className="error-message"
                    role="alert"
                  >
                    <Icon name="info" size={18} />
                    <span>{error}</span>
                  </div>
                )}
              </section>

              {/* PROCESS CARD */}
              <aside className="process-card">
                <span className="section-label">
                  THE PROCESS
                </span>

                <h2>From recording to result</h2>

                <div className="process-step">
                  <span className="step-number">01</span>

                  <div>
                    <strong>Upload audio</strong>
                    <p>Select a supported voice recording.</p>
                  </div>
                </div>

                <div className="process-step">
                  <span className="step-number">02</span>

                  <div>
                    <strong>Analyze each window</strong>
                    <p>
                      The model processes overlapping sections.
                    </p>
                  </div>
                </div>

                <div className="process-step">
                  <span className="step-number">03</span>

                  <div>
                    <strong>Explore the timeline</strong>
                    <p>
                      Compare predictions and listen to each section.
                    </p>
                  </div>
                </div>

                <div className="process-note">
                  <Icon name="info" size={17} />
                  <span>
                    Use a clear recording with minimal
                    background noise.
                  </span>
                </div>
              </aside>
            </div>

            {/* RESULTS */}
            {prediction && (
              <section
                className="results"
                aria-live="polite"
              >
                <div className="results-heading">
                  <span className="section-label">
                    02 / RESULT
                  </span>

                  <h2>Recording analysis</h2>

                  <p>
                    The overall result combines the scores
                    of the analyzed windows.
                  </p>
                </div>

                <div className="result-grid">
                  <div
                    className={`verdict-card ${
                      isHuman
                        ? "human-result"
                        : isAI
                          ? "ai-result"
                          : ""
                    }`}
                  >
                    <span className="result-overline">
                      OVERALL CLASSIFICATION
                    </span>

                    <div className="result-symbol">
                      <Icon
                        name={
                          isHuman
                            ? "check"
                            : isAI
                              ? "wave"
                              : "info"
                        }
                        size={30}
                      />
                    </div>

                    <h3>
                      {isHuman
                        ? "Likely human voice"
                        : isAI
                          ? "Likely AI-generated voice"
                          : "No audio signal"}
                    </h3>

                    {hasSignal ? (
                      <p>
                        The combined prediction is{" "}
                        <strong>{resultLabel}</strong>.
                      </p>
                    ) : (
                      <p>
                        All analysis windows contained only
                        zero-valued samples. Try another recording.
                      </p>
                    )}
                  </div>

                  <div className="confidence-card">
                    <span className="result-overline">
                      COMBINED MODEL SCORE
                    </span>

                    <div className="confidence-value">
                      {formatScore(confidence)}
                    </div>

                    {hasSignal && (
                      <div
                        className="confidence-track"
                        role="progressbar"
                        aria-label="Combined model score"
                        aria-valuenow={confidence}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      >
                        <span
                          style={{
                            width: `${confidence}%`,
                          }}
                        />
                      </div>
                    )}

                    <p>
                      {hasSignal
                        ? "Mean class score across analyzed windows"
                        : "No classification score is available"}
                    </p>

                    <div className="analyzed-file">
                      <Icon name="wave" size={16} />
                      <span>{prediction.filename}</span>
                    </div>
                  </div>
                </div>

                {/* CLICKABLE TIMELINE */}
                <PredictionTimeline
                  timeline={timeline}
                  audioRef={audioRef}
                />

                {/* WAVEFORM AND SPECTROGRAM */}
                {visuals && (
                  <div className="visualizations">
                    <div className="results-heading">
                      <span className="section-label">
                        03 / AUDIO VIEW
                      </span>

                      <h2>Audio feature preview</h2>

                      <p>
                        These plots show a preprocessed
                        four-second preview. The timeline
                        covers the full recording.
                      </p>
                    </div>

                    <div className="visualization-grid">
                      {visuals.waveform && (
                        <div className="visualization-card">
                          <h3>Waveform</h3>

                          <img
                            src={`data:image/png;base64,${visuals.waveform}`}
                            alt="Waveform of the preprocessed audio preview"
                          />
                        </div>
                      )}

                      {visuals.log_mel && (
                        <div className="visualization-card">
                          <h3>Log-Mel spectrogram</h3>

                          <img
                            src={`data:image/png;base64,${visuals.log_mel}`}
                            alt="Log-Mel spectrogram of the preprocessed audio preview"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {visualError && (
                  <p
                    className="model-warning"
                    role="status"
                  >
                    {visualError}
                  </p>
                )}

                <div className="result-disclaimer">
                  <Icon name="info" size={18} />

                  <span>
                    Results are model estimates. The combined
                    score is not the percentage of AI-generated
                    audio in the recording.
                  </span>
                </div>
              </section>
            )}
          </>
        )}

        {/* DASHBOARD PAGE */}
        {activePage === "Dashboard" && (
          <section className="inner-page">
            <span className="section-label">OVERVIEW</span>

            <h1>Analysis dashboard</h1>

            <p>
              Up to 30 recent classification results saved
              in this browser.
            </p>

            <div className="stats-grid">
              <div className="stat-card">
                <span>Total classifications</span>
                <strong>{history.length}</strong>
              </div>

              <div className="stat-card">
                <span>Human classifications</span>
                <strong>{humanCount}</strong>
              </div>

              <div className="stat-card">
                <span>AI classifications</span>
                <strong>{aiCount}</strong>
              </div>
            </div>

            <button
              className="browse-button"
              onClick={() => setActivePage("History")}
            >
              View history
              <Icon name="arrow" size={17} />
            </button>
          </section>
        )}

        {/* HISTORY PAGE */}
        {activePage === "History" && (
          <section className="inner-page">
            <span className="section-label">
              RECENT ACTIVITY
            </span>

            <h1>Analysis history</h1>

            <p>
              Up to 30 recent results saved in this browser.
            </p>

            {history.length > 0 ? (
              <div className="history-list">
                {history.map((item, index) => (
                  <div
                    className="history-item"
                    key={item.id ?? index}
                  >
                    <div className="history-file-icon">
                      <Icon name="wave" size={20} />
                    </div>

                    <div className="history-details">
                      <strong>{item.filename}</strong>

                      <small>
                        {new Date(item.date).toLocaleString()}

                        {item.mode === "timeline"
                          ? ` · ${item.segment_count} analysis windows`
                          : ""}
                      </small>
                    </div>

                    <div className="history-result">
                      <strong>{item.prediction}</strong>

                      <small>
                        {formatScore(item.confidence)} model score
                      </small>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-history">
                <Icon name="clock" size={30} />

                <h2>No recordings analyzed yet</h2>

                <p>
                  Your results will appear here after
                  the first analysis.
                </p>

                <button
                  className="browse-button"
                  onClick={() => setActivePage("Analyzer")}
                >
                  Open analyzer
                </button>
              </div>
            )}
          </section>
        )}

        {/* UPDATED MODEL PAGE */}
        {activePage === "Model" && (
          <section className="inner-page">
            <span className="section-label">
              MODEL OVERVIEW
            </span>

            <h1>AI Voice vs Real Voice Prediction</h1>

            <p>
              A trained CNN + Transformer classifier analyzes
              overlapping audio windows. The system combines
              their class scores and displays each window
              on a playback timeline.
            </p>

            <div className="stats-grid">
              <div className="stat-card">
                <span>Timeline sample rate</span>
                <strong>16 kHz</strong>
              </div>

              <div className="stat-card">
                <span>Analysis window</span>
                <strong>4 sec</strong>
              </div>

              <div className="stat-card">
                <span>Window step</span>
                <strong>2 sec</strong>
              </div>
            </div>

            <div className="model-grid">
              <div>
                <span>01 / FEATURES</span>

                <h2>Log-Mel spectrogram</h2>

                <p>
                  Each window passes through the existing
                  preprocessing and feature extraction pipeline.
                </p>
              </div>

              <div>
                <span>02 / CLASSIFIER</span>

                <h2>CNN + Transformer</h2>

                <p>
                  The trained model returns two class scores:
                  REAL (class 0) and AI-GENERATED (class 1).
                </p>
              </div>

              <div>
                <span>03 / SUMMARY</span>

                <h2>Combined prediction</h2>

                <p>
                  The system averages each class score across
                  analyzed windows and selects the class
                  with the higher mean.
                </p>
              </div>
            </div>

            <details
              style={{
                border: "1px solid #e1e8f6",
                borderRadius: 10,
                padding: 20,
              }}
            >
              <summary
                style={{
                  cursor: "pointer",
                  fontWeight: 700,
                  color: "#2454c6",
                }}
              >
                Understanding the timeline and scores
              </summary>

              <ul
                style={{
                  color: "#62728c",
                  fontSize: 13,
                  lineHeight: 1.8,
                  paddingLeft: 20,
                }}
              >
                <li>
                  Windows are created across the full recording
                  before model-specific preprocessing.
                </li>

                <li>
                  Four-second windows start every two seconds.
                  Overlapping windows appear on separate
                  timeline rows.
                </li>

                <li>
                  A short final window uses padding for model
                  input; its displayed end time stays at the
                  recording’s actual end.
                </li>

                <li>
                  Clicking a window starts playback at its
                  timestamp and continues through the recording.
                </li>

                <li>
                  All-zero windows are marked “No signal”
                  and excluded from the combined score.
                  This check does not detect speech or
                  background noise.
                </li>

                <li>
                  The combined score is a mean model score.
                  It is not test-set accuracy or the percentage
                  of AI-generated speech.
                </li>

                <li>
                  Segment predictions indicate model assessments
                  within windows; they do not establish exact
                  editing boundaries.
                </li>
              </ul>
            </details>
          </section>
        )}
      </main>

      <footer className="footer">
        <span>VoiceGuard AI</span>
        <span>AI Voice Detection · Final Year Project</span>
      </footer>
    </div>
  );
}

export default App;