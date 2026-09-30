import { useEffect, useState } from "react";

function formatTime(value) {
  return `${Number(Number(value).toFixed(3))}s`;
}

function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}%`
    : "—";
}

function getResult(segment) {
  if (segment.prediction === "REAL") {
    return {
      label: "Likely human",
      tone: "human",
    };
  }

  if (segment.prediction === "AI-GENERATED") {
    return {
      label: "Likely AI generated",
      tone: "ai",
    };
  }

  return {
    label:
      segment.prediction === "NO_SIGNAL"
        ? "No signal"
        : "Unavailable",
    tone: "empty",
  };
}

export default function PredictionTimeline({
  timeline,
  audioRef,
}) {
  const [selectedId, setSelectedId] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackError, setPlaybackError] = useState("");

  useEffect(() => {
    setSelectedId(null);
    setPlaybackError("");

    const audio = audioRef.current;

    if (!audio) return;

    const updateTime = () => {
      setCurrentTime(audio.currentTime || 0);
    };

    updateTime();

    audio.addEventListener("timeupdate", updateTime);
    audio.addEventListener("seeked", updateTime);
    audio.addEventListener("loadedmetadata", updateTime);

    return () => {
      audio.removeEventListener("timeupdate", updateTime);
      audio.removeEventListener("seeked", updateTime);
      audio.removeEventListener("loadedmetadata", updateTime);
    };
  }, [timeline, audioRef]);

  if (!timeline || !Array.isArray(timeline.segments)) {
    return null;
  }

  const segments = timeline.segments;
  const duration = Number(timeline.duration_seconds);

  if (
    !segments.length ||
    !Number.isFinite(duration) ||
    duration <= 0
  ) {
    return null;
  }

  const laneCount = segments.length > 1 ? 2 : 1;

  const playhead = Math.min(
    100,
    Math.max(0, (currentTime / duration) * 100)
  );

  async function playFromSegment(segment) {
    const audio = audioRef.current;

    setPlaybackError("");

    if (!audio) {
      setPlaybackError(
        "The audio preview is unavailable. Select the file again."
      );
      return;
    }

    try {
      audio.currentTime = segment.start_seconds;

      setCurrentTime(segment.start_seconds);
      setSelectedId(segment.segment_id);

      await audio.play();
    } catch {
      setPlaybackError(
        "Playback could not start. Try the audio controls above, or use a WAV file."
      );
    }
  }

  return (
    <section
      className="vg-timeline"
      aria-label="Prediction timeline"
    >
      <style>{timelineStyles}</style>

      <div className="vg-timeline-heading">
        <div>
          <span className="vg-timeline-eyebrow">
            SEGMENT ANALYSIS
          </span>

          <h2>Prediction timeline</h2>

          <p>
            Click a window to play the recording from its start time.
          </p>
        </div>

        <span className="vg-timeline-count">
          {segments.length}{" "}
          {segments.length === 1 ? "window" : "windows"}
        </span>
      </div>

      <div className="vg-timeline-meta">
        <span>
          Duration: <strong>{formatTime(duration)}</strong>
        </span>

        <span>
          Window:{" "}
          <strong>{formatTime(timeline.window_seconds)}</strong>
        </span>

        <span>
          Step:{" "}
          <strong>{formatTime(timeline.step_seconds)}</strong>
        </span>
      </div>

      <div className="vg-timeline-legend">
        <span>
          <i className="vg-dot-human" />
          Likely human
        </span>

        <span>
          <i className="vg-dot-ai" />
          Likely AI generated
        </span>

        <span>
          <i className="vg-dot-empty" />
          No signal
        </span>
      </div>

      <div className="vg-timeline-scroll">
        <div
          style={{
            minWidth: Math.max(300, segments.length * 32),
          }}
        >
          <div
            className="vg-timeline-track"
            style={{
              height: laneCount * 40 + 12,
            }}
          >
            {segments.map((segment, index) => {
              const result = getResult(segment);

              return (
                <button
                  type="button"
                  key={segment.segment_id}
                  className={`vg-window vg-window-${result.tone}`}
                  aria-pressed={
                    selectedId === segment.segment_id
                  }
                  aria-label={
                    `Play from window ${segment.segment_id}, ` +
                    `${formatTime(segment.start_seconds)} to ` +
                    `${formatTime(segment.end_seconds)}, ` +
                    result.label
                  }
                  title={
                    `${formatTime(segment.start_seconds)}–` +
                    `${formatTime(segment.end_seconds)} · ` +
                    `${result.label} · ` +
                    formatScore(segment.confidence)
                  }
                  style={{
                    left: `${
                      (segment.start_seconds / duration) * 100
                    }%`,
                    width: `${
                      (
                        (segment.end_seconds -
                          segment.start_seconds) /
                        duration
                      ) * 100
                    }%`,
                    top: 8 + (index % laneCount) * 40,
                  }}
                  onClick={() => playFromSegment(segment)}
                >
                  {segment.segment_id}
                </button>
              );
            })}

            <div
              className="vg-playhead"
              style={{
                left: `${playhead}%`,
              }}
              aria-hidden="true"
            />
          </div>

          <div className="vg-timeline-axis">
            <span>0s</span>
            <span>{formatTime(duration / 2)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
      </div>

      <p className="vg-timeline-note">
        Windows overlap and appear on separate rows. Their scores
        are model estimates, not confirmed boundaries of edited audio.
      </p>

      {playbackError && (
        <p className="vg-timeline-error" role="alert">
          {playbackError}
        </p>
      )}

      <div className="vg-timeline-table-wrap">
        <table className="vg-timeline-table">
          <caption className="vg-visually-hidden">
            Predictions for each audio window
          </caption>

          <thead>
            <tr>
              <th scope="col">Window</th>
              <th scope="col">Time range</th>
              <th scope="col">Prediction</th>
              <th scope="col">Model score</th>
              <th scope="col">Listen</th>
            </tr>
          </thead>

          <tbody>
            {segments.map((segment) => {
              const result = getResult(segment);

              return (
                <tr
                  key={segment.segment_id}
                  className={
                    selectedId === segment.segment_id
                      ? "vg-row-selected"
                      : ""
                  }
                >
                  <th scope="row">
                    {segment.segment_id}
                  </th>

                  <td>
                    {formatTime(segment.start_seconds)}–
                    {formatTime(segment.end_seconds)}
                  </td>

                  <td>
                    <span
                      className={`vg-label vg-label-${result.tone}`}
                    >
                      {result.label}
                    </span>
                  </td>

                  <td>
                    {formatScore(segment.confidence)}
                  </td>

                  <td>
                    <button
                      type="button"
                      className="vg-play-button"
                      onClick={() => playFromSegment(segment)}
                      aria-label={
                        "Play recording from " +
                        formatTime(segment.start_seconds)
                      }
                    >
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 12 12"
                        aria-hidden="true"
                      >
                        <path
                          d="M3 1.5 10 6 3 10.5Z"
                          fill="currentColor"
                        />
                      </svg>

                      Play from here
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const timelineStyles = `
.vg-timeline {
  margin-top: 32px;
  padding: 28px;
  background: #fff;
  border: 1px solid #e2e9f5;
  border-radius: 15px;
  box-shadow: 0 10px 28px #264a900b;
}

.vg-timeline-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.vg-timeline-eyebrow {
  color: #2454c6;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.5px;
}

.vg-timeline-heading h2 {
  margin: 8px 0;
  font: 800 22px Manrope, Arial, sans-serif;
  color: #17243b;
  letter-spacing: -0.6px;
}

.vg-timeline-heading p {
  margin: 0;
  color: #62728c;
  font-size: 12px;
  line-height: 1.6;
}

.vg-timeline-count {
  flex: none;
  background: #eef3ff;
  color: #2454c6;
  padding: 8px 11px;
  border-radius: 7px;
  font-size: 11px;
  font-weight: 700;
}

.vg-timeline-meta {
  display: flex;
  gap: 22px;
  flex-wrap: wrap;
  margin-top: 20px;
  color: #71809a;
  font-size: 12px;
}

.vg-timeline-meta strong {
  color: #243b65;
}

.vg-timeline-legend {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin: 22px 0 14px;
  color: #596c89;
  font-size: 11px;
}

.vg-timeline-legend span {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.vg-timeline-legend i {
  width: 9px;
  height: 9px;
  border-radius: 3px;
}

.vg-dot-human {
  background: #3974ea;
}

.vg-dot-ai {
  background: #7059c6;
}

.vg-dot-empty {
  background: #8995a9;
}

.vg-timeline-scroll {
  overflow-x: auto;
  padding: 5px 2px 10px;
}

.vg-timeline-track {
  position: relative;
  background: repeating-linear-gradient(
    to right,
    #edf1f8 0,
    #edf1f8 1px,
    transparent 1px,
    transparent 10%
  );
  border-bottom: 1px solid #dfe7f4;
}

.vg-window {
  position: absolute;
  height: 30px;
  padding: 0;
  border: 2px solid white;
  border-radius: 7px;
  color: #fff;
  font-size: 10px;
  font-weight: 800;
  overflow: hidden;
  cursor: pointer;
}

.vg-window-human {
  background: #3974ea;
}

.vg-window-ai {
  background: #7059c6;
}

.vg-window-empty {
  background: #8995a9;
}

.vg-window:hover {
  filter: brightness(0.92);
}

.vg-window[aria-pressed="true"] {
  outline: 2px solid #182e59;
  outline-offset: 1px;
  z-index: 2;
}

.vg-window:focus-visible,
.vg-play-button:focus-visible {
  outline: 3px solid #173e9c;
  outline-offset: 2px;
  z-index: 3;
}

.vg-playhead {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #1b2d4c;
  pointer-events: none;
  z-index: 4;
  transform: translateX(-1px);
}

.vg-timeline-axis {
  display: flex;
  justify-content: space-between;
  margin-top: 9px;
  color: #7585a0;
  font-size: 10px;
}

.vg-timeline-note {
  margin: 12px 0 20px;
  color: #72809a;
  font-size: 11px;
  line-height: 1.7;
}

.vg-timeline-error {
  background: #fff2ee;
  color: #9d4632;
  padding: 12px;
  border-radius: 8px;
  font-size: 12px;
}

.vg-timeline-table-wrap {
  overflow: auto;
  max-height: 380px;
  border: 1px solid #e3eaf5;
  border-radius: 9px;
}

.vg-timeline-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 12px;
}

.vg-timeline-table th,
.vg-timeline-table td {
  padding: 13px 14px;
  border-bottom: 1px solid #e8edf6;
  white-space: nowrap;
}

.vg-timeline-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #f7f9fe;
  color: #687c9d;
  font-size: 10px;
  letter-spacing: 0.4px;
}

.vg-timeline-table tbody th {
  color: #52688c;
}

.vg-timeline-table tbody tr:last-child > * {
  border-bottom: 0;
}

.vg-row-selected {
  background: #f0f5ff;
}

.vg-label {
  display: inline-block;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 10px;
  font-weight: 700;
}

.vg-label-human {
  color: #2454c6;
  background: #edf3ff;
}

.vg-label-ai {
  color: #624bae;
  background: #f2effc;
}

.vg-label-empty {
  color: #62728a;
  background: #eff2f6;
}

.vg-play-button {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 7px 10px;
  background: white;
  color: #2454c6;
  border: 1px solid #c9d8f5;
  border-radius: 7px;
  font-size: 11px;
  cursor: pointer;
}

.vg-play-button:hover {
  background: #edf3ff;
}

.vg-visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
}

@media (max-width: 650px) {
  .vg-timeline {
    padding: 19px;
  }

  .vg-timeline-heading {
    flex-direction: column;
  }

  .vg-timeline-meta,
  .vg-timeline-legend {
    gap: 12px;
  }
}
`;