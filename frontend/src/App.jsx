
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const MAX_FILE_SIZE = 25 * 1024 * 1024;

const ICONS = {
  logo: "◈",
  analyze: "◉",
  dashboard: "▦",
  history: "◷",
  model: "◇",
  upload: "↑",
  audio: "♫",
  play: "▶",
  close: "×",
};

const LABELS = {
  REAL: "Human Voice",
  "AI-GENERATED": "AI-Generated Voice",
  HYBRID: "Mixed Voice",
  NO_SIGNAL: "No Signal",
};

function formatTime(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const minutes = Math.floor(seconds / 60);

  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function formatConfidence(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  const percentage = value <= 1 ? value * 100 : value;
  return `${Math.min(100, Math.max(0, percentage)).toFixed(1)}%`;
}

function loadHistory() {
  try {
    const stored = JSON.parse(
      localStorage.getItem("voiceguard-history") || "[]"
    );

    return Array.isArray(stored) ? stored.slice(0, 30) : [];
  } catch {
    return [];
  }
}

function getApiError(data, fallback) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg || "Validation error")
      .join(", ");
  }

  return fallback;
}

function StatusBadge({ prediction }) {
  const type =
    prediction === "REAL"
      ? "human"
      : prediction === "AI-GENERATED"
        ? "synthetic"
        : prediction === "HYBRID"
          ? "mixed"
          : "neutral";

  return (
    <span className={`status-badge ${type}`}>
      <span className="status-indicator" />
      {LABELS[prediction] || prediction}
    </span>
  );
}

function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="page-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}

function HistoryList({ items }) {
  if (!items.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">◷</div>
        <strong>No recordings yet</strong>
        <p>Your completed analyses will appear here.</p>
      </div>
    );
  }

  return (
    <div className="history-list">
      {items.map((item, index) => (
        <div className="history-item" key={item.id ?? index}>
          <div className="history-icon">♫</div>

          <div className="history-info">
            <strong title={item.filename}>
              {item.filename || "Audio recording"}
            </strong>

            <span>
              {item.date
                ? new Date(item.date).toLocaleString()
                : "Previous analysis"}
              {" · "}
              {item.mode === "hybrid" ? "Hybrid" : "Standard"}
            </span>
          </div>

          <StatusBadge prediction={item.prediction} />
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------
// HYBRID TIMELINE
// ------------------------------------------------------------

function HybridTimeline({ result, audioRef }) {
  const [active, setActive] = useState(-1);

  const duration = Number(result?.duration_seconds) || 0;

  const sections = useMemo(() => {
    if (!duration || !Array.isArray(result?.segments)) {
      return [];
    }

    const windows = result.segments
      .map((segment) => ({
        start: Math.max(0, Number(segment.start_time) || 0),
        end: Math.min(
          duration,
          Number(segment.end_time) || 0
        ),
        prediction: segment.prediction,
      }))
      .filter(
        (segment) =>
          segment.end > segment.start &&
          ["REAL", "AI-GENERATED"].includes(segment.prediction)
      );

    if (!windows.length) return [];

    const boundaries = [
      0,
      duration,
      ...windows.flatMap((segment) => [
        segment.start,
        segment.end,
      ]),
    ]
      .sort((a, b) => a - b)
      .filter(
        (value, index, array) =>
          index === 0 || value - array[index - 1] > 0.001
      );

    const output = [];

    for (let index = 0; index < boundaries.length - 1; index++) {
      const start = boundaries[index];
      const end = boundaries[index + 1];
      const middle = (start + end) / 2;

      const covering = windows.filter(
        (segment) =>
          segment.start <= middle &&
          segment.end >= middle
      );

      if (!covering.length) continue;

      const humanVotes = covering.filter(
        (segment) => segment.prediction === "REAL"
      ).length;

      const aiVotes = covering.filter(
        (segment) =>
          segment.prediction === "AI-GENERATED"
      ).length;

      let prediction;

      if (humanVotes > aiVotes) {
        prediction = "REAL";
      } else if (aiVotes > humanVotes) {
        prediction = "AI-GENERATED";
      } else {
        // Display-only tie breaker.
        // This does not modify the model prediction.
        const nearest = [...covering].sort(
          (a, b) =>
            Math.abs((a.start + a.end) / 2 - middle) -
            Math.abs((b.start + b.end) / 2 - middle)
        )[0];

        prediction = nearest.prediction;
      }

      const previous = output[output.length - 1];

      if (
        previous &&
        previous.prediction === prediction &&
        Math.abs(previous.end - start) < 0.001
      ) {
        previous.end = end;
      } else {
        output.push({ start, end, prediction });
      }
    }

    return output;
  }, [result, duration]);

  useEffect(() => {
    setActive(-1);
  }, [result]);

  function playSegment(segment, index) {
    setActive(index);

    if (!audioRef.current) return;

    audioRef.current.currentTime = segment.start;
    audioRef.current.play().catch(() => {});
  }

  if (!sections.length) return null;

  return (
    <section className="surface timeline-surface">
      <div className="surface-heading">
        <div>
          <span className="eyebrow">TEMPORAL ANALYSIS</span>
          <h2>Voice detection timeline</h2>
          <p>
            Explore classified sections throughout the recording.
          </p>
        </div>

        <span className="subtle-chip">
          {formatTime(duration)} total
        </span>
      </div>

      <div className="timeline-legend">
        <span>
          <i className="legend-dot human-dot" />
          Human Voice
        </span>

        <span>
          <i className="legend-dot ai-dot" />
          AI-Generated Voice
        </span>
      </div>

      <div className="timeline-bar">
        {sections.map((segment, index) => (
          <button
            key={index}
            type="button"
            className={`timeline-segment ${
              segment.prediction === "REAL"
                ? "human-segment"
                : "ai-segment"
            } ${active === index ? "segment-active" : ""}`}
            style={{
              width: `${
                ((segment.end - segment.start) / duration) * 100
              }%`,
            }}
            title={`${LABELS[segment.prediction]} · ${formatTime(
              segment.start
            )} to ${formatTime(segment.end)}`}
            onClick={() => playSegment(segment, index)}
          />
        ))}
      </div>

      <div className="timeline-axis">
        <span>0:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      <div className="segments-heading">
        <h3>Detected sections</h3>
        <span>Select a section to play</span>
      </div>

      <div className="segment-list">
        {sections.map((segment, index) => (
          <button
            type="button"
            key={index}
            className={`segment-row ${
              active === index ? "row-active" : ""
            }`}
            onClick={() => playSegment(segment, index)}
          >
            <span className="segment-name">
              <i
                className={`legend-dot ${
                  segment.prediction === "REAL"
                    ? "human-dot"
                    : "ai-dot"
                }`}
              />
              {LABELS[segment.prediction]}
            </span>

            <span className="segment-range">
              {formatTime(segment.start)}
              {" — "}
              {formatTime(segment.end)}
            </span>

            <span className="play-indicator">▶</span>
          </button>
        ))}
      </div>

      <p className="timeline-note">
        Section boundaries are estimates from overlapping
        four-second windows. Tied votes are displayed using
        the nearest window's label.
      </p>
    </section>
  );
}

function StandardTimeline({ result, audioRef }) {
  const segments = Array.isArray(result?.segments)
    ? result.segments
    : [];

  if (!segments.length) return null;

  return (
    <section className="surface timeline-surface">
      <div className="surface-heading">
        <div>
          <span className="eyebrow">SEGMENT RESULTS</span>
          <h2>Prediction timeline</h2>
          <p>Review each analyzed audio window.</p>
        </div>
      </div>

      <div className="segment-list">
        {segments.map((segment, index) => (
          <button
            type="button"
            className="segment-row"
            key={index}
            onClick={() => {
              if (!audioRef.current) return;

              audioRef.current.currentTime =
                Number(segment.start_time) || 0;

              audioRef.current.play().catch(() => {});
            }}
          >
            <span className="segment-name">
              <i
                className={`legend-dot ${
                  segment.prediction === "REAL"
                    ? "human-dot"
                    : "ai-dot"
                }`}
              />

              {LABELS[segment.prediction] ||
                segment.prediction}
            </span>

            <span className="segment-range">
              {formatTime(segment.start_time)}
              {" — "}
              {formatTime(segment.end_time)}
            </span>

            <span className="play-indicator">▶</span>
          </button>
        ))}
      </div>
    </section>
  );
}

// ------------------------------------------------------------
// MAIN APPLICATION
// ------------------------------------------------------------

export default function App() {
  const [page, setPage] = useState("Analyzer");

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [dragging, setDragging] = useState(false);

  const [mode, setMode] = useState("hybrid");
  const [resultMode, setResultMode] = useState("hybrid");

  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [history, setHistory] = useState(loadHistory);
  const [search, setSearch] = useState("");

  const inputRef = useRef(null);
  const audioRef = useRef(null);
  const objectUrlRef = useRef("");
  const controllerRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(
        "voiceguard-history",
        JSON.stringify(history)
      );
    } catch {
      // Browser storage may be disabled.
    }
  }, [history]);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  function resetResult() {
    setResult(null);
    setError("");
  }

  function selectFile(selected) {
    if (!selected || controllerRef.current) return;

    const extension = selected.name
      .split(".")
      .pop()
      ?.toLowerCase();

    if (!["wav", "mp3", "flac"].includes(extension)) {
      setError("Please upload a WAV, MP3, or FLAC audio file.");
      return;
    }

    if (!selected.size || selected.size > MAX_FILE_SIZE) {
      setError("File size must be greater than 0 and under 25 MB.");
      return;
    }

    audioRef.current?.pause();

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    const nextUrl = URL.createObjectURL(selected);

    objectUrlRef.current = nextUrl;

    setFile(selected);
    setAudioUrl(nextUrl);

    resetResult();
  }

  function removeFile() {
    if (controllerRef.current) return;

    audioRef.current?.pause();

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }

    objectUrlRef.current = "";

    setFile(null);
    setAudioUrl("");

    resetResult();
  }

  async function analyzeAudio() {
    if (!file || controllerRef.current) return;

    const controller = new AbortController();
    controllerRef.current = controller;

    const currentMode = mode;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setLoadingMessage("Processing audio recording...");
    setResultMode(currentMode);
    resetResult();

    try {
      const endpoint =
        currentMode === "hybrid"
          ? "/analyze-hybrid"
          : "/predict-timeline";

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          getApiError(data, "Audio analysis failed.")
        );
      }

      if (
        !data.summary ||
        !Array.isArray(data.segments)
      ) {
        throw new Error(
          "The backend returned an incomplete analysis."
        );
      }

      const prediction = data.summary.prediction;

      if (
        ![
          "REAL",
          "AI-GENERATED",
          "HYBRID",
          "NO_SIGNAL",
        ].includes(prediction)
      ) {
        throw new Error(
          "The backend returned an unknown prediction."
        );
      }

      setResult(data);

      if (prediction !== "NO_SIGNAL") {
        setHistory((previous) =>
          [
            {
              id: `${Date.now()}-${Math.random()}`,
              filename: file.name,
              prediction,
              mode: currentMode,
              date: new Date().toISOString(),
            },
            ...previous,
          ].slice(0, 30)
        );
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(
          failure instanceof TypeError
            ? "Cannot connect to the FastAPI backend. Make sure the server is running."
            : failure.message
        );
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setLoading(false);
        setLoadingMessage("");
      }
    }
  }

  const summary = result?.summary;
  const prediction = summary?.prediction;

  const segments = Array.isArray(result?.segments)
    ? result.segments
    : [];

  const duration = Number(result?.duration_seconds) || 0;

  const totalSegments =
    result?.segment_count ??
    result?.total_segments ??
    segments.length;

  const humanSegments =
    result?.real_segment_count ??
    segments.filter((s) => s.prediction === "REAL").length;

  const aiSegments =
    result?.ai_segment_count ??
    segments.filter(
      (s) => s.prediction === "AI-GENERATED"
    ).length;

  const stats = useMemo(
    () => ({
      total: history.length,
      human: history.filter(
        (item) => item.prediction === "REAL"
      ).length,
      ai: history.filter(
        (item) => item.prediction === "AI-GENERATED"
      ).length,
      mixed: history.filter(
        (item) => item.prediction === "HYBRID"
      ).length,
    }),
    [history]
  );

  const filteredHistory = history.filter((item) =>
    String(item.filename || "")
      .toLowerCase()
      .includes(search.toLowerCase())
  );

  const navigation = [
    { name: "Analyzer", icon: ICONS.analyze },
    { name: "Dashboard", icon: ICONS.dashboard },
    { name: "History", icon: ICONS.history },
    { name: "Model", icon: ICONS.model },
  ];

  return (
    <div className="vg-app">
      {/* SIDEBAR */}

      <aside className="vg-sidebar">
        <button
          className="vg-brand"
          type="button"
          onClick={() => setPage("Analyzer")}
        >
          <span className="brand-symbol">{ICONS.logo}</span>

          <span className="brand-text">
            <strong>VoiceGuard</strong>
            <small>AI AUDIO INTELLIGENCE</small>
          </span>
        </button>

        <span className="nav-section-label">WORKSPACE</span>

        <nav className="vg-navigation">
          {navigation.map((item) => (
            <button
              type="button"
              key={item.name}
              className={
                page === item.name ? "nav-active" : ""
              }
              onClick={() => setPage(item.name)}
            >
              <span className="navigation-icon">
                {item.icon}
              </span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-divider" />

          <div className="sidebar-profile">
            <div className="profile-icon">VG</div>

            <div>
              <strong>VoiceGuard AI</strong>
              <small>Detection workspace</small>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <div className="vg-main">
        <header className="vg-header">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>{page}</strong>
          </div>

          <div className="header-right">
            <span className="header-project">
              AI Voice Detection
            </span>
            <div className="header-avatar">VG</div>
          </div>
        </header>

        <main className="vg-content">
          {/* ANALYZER PAGE */}

          {page === "Analyzer" && (
            <>
              <PageHeading
                eyebrow="AUDIO INTELLIGENCE"
                title="Voice Analyzer"
                description="Identify human and AI-generated speech using intelligent audio classification."
              />

              <section className="surface upload-surface">
                <div className="surface-heading">
                  <div>
                    <span className="eyebrow">
                      AUDIO INPUT
                    </span>
                    <h2>Upload recording</h2>
                    <p>
                      Select an audio file to start a new detection.
                    </p>
                  </div>

                  <span className="subtle-chip">
                    WAV · MP3 · FLAC
                  </span>
                </div>

                <input
                  ref={inputRef}
                  type="file"
                  accept=".wav,.mp3,.flac"
                  hidden
                  disabled={loading}
                  onChange={(event) => {
                    selectFile(event.target.files?.[0]);
                    event.target.value = "";
                  }}
                />

                {!file ? (
                  <div
                    className={`upload-dropzone ${
                      dragging ? "dragging" : ""
                    }`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={(event) => {
                      event.preventDefault();
                      setDragging(false);
                    }}
                    onDrop={(event) => {
                      event.preventDefault();
                      setDragging(false);

                      selectFile(
                        event.dataTransfer.files?.[0]
                      );
                    }}
                  >
                    <div className="upload-icon">
                      {ICONS.upload}
                    </div>

                    <h3>Drag and drop your audio file</h3>

                    <p>
                      Or browse your device to select a recording
                    </p>

                    <button
                      type="button"
                      className="primary-button"
                      onClick={() =>
                        inputRef.current?.click()
                      }
                    >
                      Browse files
                      <span>→</span>
                    </button>

                    <small>
                      Supports WAV, MP3, FLAC · Maximum 25 MB
                    </small>
                  </div>
                ) : (
                  <div className="selected-audio">
                    <div className="file-card">
                      <div className="file-icon">
                        {ICONS.audio}
                      </div>

                      <div className="file-details">
                        <strong title={file.name}>
                          {file.name}
                        </strong>

                        <span>
                          {(file.size / 1048576).toFixed(2)}
                          {" MB · "}Ready for analysis
                        </span>
                      </div>

                      <button
                        type="button"
                        className="remove-file"
                        disabled={loading}
                        onClick={removeFile}
                        aria-label="Remove audio file"
                      >
                        ×
                      </button>
                    </div>

                    <audio
                      ref={audioRef}
                      controls
                      preload="metadata"
                      src={audioUrl}
                      className="audio-player"
                    />

                    <div className="mode-header">
                      <h3>Detection mode</h3>
                      <span>
                        Choose your analysis method
                      </span>
                    </div>

                    <div className="detection-modes">
                      <button
                        type="button"
                        disabled={loading}
                        className={`detection-mode ${
                          mode === "standard"
                            ? "mode-active"
                            : ""
                        }`}
                        onClick={() => setMode("standard")}
                      >
                        <span className="mode-icon">◉</span>

                        <strong>
                          Standard Detection
                        </strong>

                        <small>
                          Classify human or AI-generated speech
                        </small>

                        <i className="mode-selector" />
                      </button>

                      <button
                        type="button"
                        disabled={loading}
                        className={`detection-mode ${
                          mode === "hybrid"
                            ? "mode-active"
                            : ""
                        }`}
                        onClick={() => setMode("hybrid")}
                      >
                        <span className="mode-icon">◇</span>

                        <strong>
                          Hybrid Detection
                        </strong>

                        <small>
                          Identify human and AI voice sections
                        </small>

                        <i className="mode-selector" />
                      </button>
                    </div>

                    <button
                      type="button"
                      className="primary-button run-analysis"
                      disabled={loading}
                      onClick={analyzeAudio}
                    >
                      {loading
                        ? "Analyzing recording..."
                        : "Analyze Recording"}

                      {!loading && <span>→</span>}
                    </button>

                    <button
                      type="button"
                      className="change-audio"
                      disabled={loading}
                      onClick={() =>
                        inputRef.current?.click()
                      }
                    >
                      Choose another recording
                    </button>

                    {loading && (
                      <div className="analysis-loading" role="status">
                        <div className="progress-track">
                          <span />
                        </div>

                        <p>{loadingMessage}</p>
                      </div>
                    )}
                  </div>
                )}

                {error && (
                  <div className="error-message" role="alert">
                    {error}
                  </div>
                )}
              </section>

              {/* RESULTS */}

              {result && (
                <div className="analysis-results">
                  <div className="results-heading">
                    <div>
                      <span className="eyebrow">
                        ANALYSIS COMPLETE
                      </span>

                      <h2>Detection Results</h2>

                      <p>
                        Classification summary for your recording.
                      </p>
                    </div>
                  </div>

                  <div className="result-grid">
                    <section className="surface verdict-surface">
                      <span className="eyebrow">
                        OVERALL PREDICTION
                      </span>

                      <div className="verdict-icon">
                        {prediction === "REAL" ? "✓" : "◇"}
                      </div>

                      <h2>
                        {LABELS[prediction] || prediction}
                      </h2>

                      <StatusBadge prediction={prediction} />

                      <p>
                        {prediction === "HYBRID"
                          ? "The analyzed windows contain both human and AI-generated classifications."
                          : prediction === "REAL"
                            ? "The model classified the recording as human speech."
                            : prediction === "AI-GENERATED"
                              ? "The model classified the recording as AI-generated speech."
                              : "The recording did not produce a usable voice classification."}
                      </p>
                    </section>

                    <section className="surface summary-surface">
                      <span className="eyebrow">
                        ANALYSIS SUMMARY
                      </span>

                      <div className="summary-line">
                        <span>Recording duration</span>
                        <strong>
                          {formatTime(duration)}
                        </strong>
                      </div>

                      <div className="summary-line">
                        <span>Detection mode</span>
                        <strong>
                          {resultMode === "hybrid"
                            ? "Hybrid"
                            : "Standard"}
                        </strong>
                      </div>

                      <div className="summary-line">
                        <span>Analyzed windows</span>
                        <strong>{totalSegments}</strong>
                      </div>

                      {resultMode === "hybrid" ? (
                        <>
                          <div className="summary-line">
                            <span>Human windows</span>
                            <strong>{humanSegments}</strong>
                          </div>

                          <div className="summary-line">
                            <span>AI-generated windows</span>
                            <strong>{aiSegments}</strong>
                          </div>
                        </>
                      ) : (
                        <div className="summary-line">
                          <span>Model confidence</span>
                          <strong>
                            {formatConfidence(
                              summary?.confidence
                            )}
                          </strong>
                        </div>
                      )}
                    </section>
                  </div>

                  {resultMode === "hybrid" ? (
                    <HybridTimeline
                      result={result}
                      audioRef={audioRef}
                    />
                  ) : (
                    <StandardTimeline
                      result={result}
                      audioRef={audioRef}
                    />
                  )}

                  <p className="result-disclaimer">
                    VoiceGuard provides model-based predictions.
                    These results are not definitive proof of
                    authenticity. Hybrid detection accuracy
                    requires validation on mixed recordings.
                  </p>
                </div>
              )}
            </>
          )}

          {/* DASHBOARD */}

          {page === "Dashboard" && (
            <>
              <PageHeading
                eyebrow="WORKSPACE OVERVIEW"
                title="Dashboard"
                description="Review recent voice detection activity and classification results."
              />

              <div className="statistics-grid">
                {[
                  ["Total Analyses", stats.total],
                  ["Human Voice", stats.human],
                  ["AI-Generated", stats.ai],
                  ["Mixed Voice", stats.mixed],
                ].map(([label, value]) => (
                  <div className="surface statistic" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>

              <section className="surface">
                <div className="surface-heading">
                  <div>
                    <span className="eyebrow">
                      ACTIVITY
                    </span>
                    <h2>Recent analyses</h2>
                  </div>

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setPage("History")}
                  >
                    View all →
                  </button>
                </div>

                <HistoryList items={history.slice(0, 5)} />
              </section>
            </>
          )}

          {/* HISTORY */}

          {page === "History" && (
            <>
              <PageHeading
                eyebrow="ANALYSIS RECORDS"
                title="History"
                description="Browse previous audio classification results saved in this browser."
              />

              <section className="surface">
                <div className="surface-heading">
                  <div>
                    <h2>Analysis history</h2>
                    <p>
                      {history.length} saved recordings
                    </p>
                  </div>

                  {history.length > 0 && (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => {
                        if (
                          window.confirm(
                            "Clear all analysis history?"
                          )
                        ) {
                          setHistory([]);
                        }
                      }}
                    >
                      Clear history
                    </button>
                  )}
                </div>

                <input
                  className="history-search"
                  type="search"
                  placeholder="Search recordings..."
                  value={search}
                  onChange={(event) =>
                    setSearch(event.target.value)
                  }
                />

                <HistoryList items={filteredHistory} />
              </section>
            </>
          )}

          {/* MODEL */}

          {page === "Model" && (
            <>
              <PageHeading
                eyebrow="MODEL INFORMATION"
                title="Detection Model"
                description="Technical information about the VoiceGuard classification pipeline."
              />

              <div className="model-grid">
                {[
                  [
                    "01",
                    "Audio Processing",
                    "Audio is prepared as mono speech sampled at 16 kHz.",
                  ],
                  [
                    "02",
                    "Feature Extraction",
                    "Log-Mel spectrograms represent the audio features.",
                  ],
                  [
                    "03",
                    "CNN + Transformer",
                    "The trained network predicts human or AI-generated speech.",
                  ],
                  [
                    "04",
                    "Hybrid Detection",
                    "Overlapping four-second windows analyze the complete recording.",
                  ],
                ].map(([number, title, description]) => (
                  <div className="surface model-card" key={number}>
                    <span>{number}</span>
                    <h3>{title}</h3>
                    <p>{description}</p>
                  </div>
                ))}
              </div>

              <section className="surface">
                <div className="surface-heading">
                  <div>
                    <span className="eyebrow">
                      SPECIFICATIONS
                    </span>
                    <h2>Model configuration</h2>
                  </div>
                </div>

                {[
                  ["Architecture", "CNN + Transformer"],
                  ["Sample rate", "16,000 Hz"],
                  ["Input features", "80 × 251 Log-Mel"],
                  ["Input duration", "4 seconds"],
                  ["Hybrid step", "2 seconds"],
                  ["Classification", "REAL / AI-GENERATED"],
                ].map(([label, value]) => (
                  <div className="summary-line" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </section>
            </>
          )}
        </main>

        <footer className="vg-footer">
          <span>
            © VoiceGuard AI · Voice Authenticity Detection
          </span>
          <span>
            CNN + Transformer
          </span>
        </footer>
      </div>
    </div>
  );
}
