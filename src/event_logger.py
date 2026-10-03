import os
import sqlite3
from datetime import datetime


DB_PATH = "data/face_tracker.db"
LOG_PATH = "logs/events.log"
IMAGE_DIR = "data/events"


def ensure_directories():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)


def log_event(face_id, event_type, image_path=None):
    """
    Store an event in:
        1. SQLite database
        2. logs/events.log

    Supported event types:
        ENTRY
        EXIT
        RECOGNIZED
        TRACKING
        EMBEDDING_GENERATED
        REGISTERED
    """

    ensure_directories()

    timestamp = datetime.now().isoformat()

    # ---------------------------------
    # SQLite
    # ---------------------------------
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        INSERT INTO events
        (face_id, event_type, timestamp, image_path)
        VALUES (?, ?, ?, ?)
        """,
        (
            face_id,
            event_type,
            timestamp,
            image_path
        )
    )

    conn.commit()
    conn.close()

    # ---------------------------------
    # events.log
    # ---------------------------------
    log_line = (
        f"{timestamp} | "
        f"Face ID: {face_id} | "
        f"Event: {event_type} | "
        f"Image: {image_path}\n"
    )

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_line)

    # ---------------------------------
    # Console output
    # ---------------------------------
    print(
        f"[EVENT] Face {face_id} -> "
        f"{event_type} | {timestamp}"
    )