import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

function History() {
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
      console.error("History error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  return (
    <div className="history-page">
      <div className="history-header">
        <div>
          <div className="status-badge">
            <span className="status-dot"></span>
            ANALYSIS HISTORY
          </div>

          <h1>Prediction History</h1>

          <p>
            Review previous VoiceGuard AI voice detection results.
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
          Loading prediction history...
        </div>
      )}

      {error && (
        <div className="dashboard-error">
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="history-card">
          <div className="history-card-header">
            <div>
              <h2>All Analyses</h2>
              <p>{predictions.length} prediction records</p>
            </div>
          </div>

          {predictions.length === 0 ? (
            <div className="empty-state">
              No prediction history available yet.
            </div>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Audio File</th>
                    <th>Classification</th>
                    <th>Confidence</th>
                    <th>Date & Time</th>
                  </tr>
                </thead>

                <tbody>
                  {predictions.map((item) => (
                    <tr key={item.id}>
                      <td className="history-id">
                        #{item.id}
                      </td>

                      <td>
                        <div className="history-file">
                          <div className="history-file-icon">
                            ♪
                          </div>

                          <span>{item.filename}</span>
                        </div>
                      </td>

                      <td>
                        <span
                          className={
                            item.prediction === "REAL"
                              ? "history-badge history-real"
                              : "history-badge history-ai"
                          }
                        >
                          {item.prediction}
                        </span>
                      </td>

                      <td>
                        <strong className="history-confidence">
                          {item.confidence}%
                        </strong>
                      </td>

                      <td className="history-date">
                        {item.created_at}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default History;