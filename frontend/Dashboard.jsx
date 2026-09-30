import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadHistory = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error("Unable to load prediction history.");
      }

      const data = await response.json();

      setPredictions(data.predictions || []);
    } catch (err) {
      console.error("Dashboard error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const totalAnalyses = predictions.length;

  const realCount = predictions.filter(
    (item) => item.prediction === "REAL"
  ).length;

  const aiCount = predictions.filter(
    (item) => item.prediction === "AI-GENERATED"
  ).length;

  const averageConfidence =
    predictions.length > 0
      ? (
          predictions.reduce(
            (total, item) => total + Number(item.confidence),
            0
          ) / predictions.length
        ).toFixed(2)
      : "0.00";

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <div>
          <div className="status-badge">
            <span className="status-dot"></span>
            VOICE ANALYTICS
          </div>

          <h1>Analysis Dashboard</h1>

          <p>
            Overview of your VoiceGuard AI voice detection activity.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={loadHistory}
        >
          ↻ Refresh
        </button>
      </div>

      {loading && (
        <div className="dashboard-message">
          Loading analysis data...
        </div>
      )}

      {error && (
        <div className="dashboard-error">
          {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <span>Total Analyses</span>
              <strong>{totalAnalyses}</strong>
              <small>All predictions</small>
            </div>

            <div className="stat-card">
              <span>Real Voices</span>
              <strong>{realCount}</strong>
              <small>Human classification</small>
            </div>

            <div className="stat-card">
              <span>AI Voices</span>
              <strong>{aiCount}</strong>
              <small>AI classification</small>
            </div>

            <div className="stat-card">
              <span>Average Confidence</span>
              <strong>{averageConfidence}%</strong>
              <small>Across all analyses</small>
            </div>
          </div>

          <div className="dashboard-content">
            <div className="distribution-card">
              <div className="section-heading">
                <div>
                  <h2>Detection Distribution</h2>
                  <p>Classification breakdown</p>
                </div>
              </div>

              <div className="distribution">
                <div className="distribution-item">
                  <div className="distribution-top">
                    <span>REAL</span>
                    <strong>{realCount}</strong>
                  </div>

                  <div className="distribution-bar">
                    <div
                      className="real-bar"
                      style={{
                        width: totalAnalyses
                          ? `${(realCount / totalAnalyses) * 100}%`
                          : "0%",
                      }}
                    />
                  </div>
                </div>

                <div className="distribution-item">
                  <div className="distribution-top">
                    <span>AI-GENERATED</span>
                    <strong>{aiCount}</strong>
                  </div>

                  <div className="distribution-bar">
                    <div
                      className="ai-bar"
                      style={{
                        width: totalAnalyses
                          ? `${(aiCount / totalAnalyses) * 100}%`
                          : "0%",
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="recent-card">
              <div className="section-heading">
                <div>
                  <h2>Recent Analyses</h2>
                  <p>Latest prediction activity</p>
                </div>
              </div>

              {predictions.length === 0 ? (
                <div className="empty-state">
                  No predictions available yet.
                </div>
              ) : (
                <div className="recent-list">
                  {predictions.slice(0, 5).map((item) => (
                    <div
                      className="recent-item"
                      key={item.id}
                    >
                      <div className="recent-file">
                        <div className="recent-icon">
                          ♪
                        </div>

                        <div>
                          <strong>
                            {item.filename}
                          </strong>

                          <span>
                            {item.created_at}
                          </span>
                        </div>
                      </div>

                      <div className="recent-result">
                        <strong
                          className={
                            item.prediction === "REAL"
                              ? "result-real"
                              : "result-ai"
                          }
                        >
                          {item.prediction}
                        </strong>

                        <span>
                          {item.confidence}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;