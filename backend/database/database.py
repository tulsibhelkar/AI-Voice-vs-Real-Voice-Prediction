import sqlite3
from pathlib import Path

DATABASE_PATH = Path("backend/database/voiceguard.db")


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def create_table():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(filename, prediction, confidence):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (filename, prediction, confidence)
        VALUES (?, ?, ?)
    """, (filename, prediction, confidence))

    connection.commit()
    connection.close()


def get_predictions():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, filename, prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    connection.close()

    return rows


if __name__ == "__main__":
    create_table()
    print("VoiceGuard SQLite database initialized successfully.")