import sqlite3
from pathlib import Path


# Database location
DB_PATH = Path("data/face_tracker.db")


def get_connection():
    """
    Create and return a SQLite database connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    return connection


def create_tables():
    """
    Create the required database tables.
    """

    connection = get_connection()
    cursor = connection.cursor()

    # -----------------------------------------
    # Faces table
    # -----------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)

    # -----------------------------------------
    # Events table
    # -----------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY (face_id) REFERENCES faces(face_id)
        )
    """)

        # -----------------------------------------
    # Face embeddings table
    # -----------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            face_embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (face_id) REFERENCES faces(face_id)
        )
    """)

    connection.commit()
    connection.close()

    print("✅ Database tables created successfully")


if __name__ == "__main__":
    create_tables()