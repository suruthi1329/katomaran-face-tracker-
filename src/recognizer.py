import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path


# ============================================================
# Database
# ============================================================

DB_PATH = Path("data/face_tracker.db")


# ============================================================
# Recognition Settings
# ============================================================

# Strong match with an already registered face
MATCH_THRESHOLD = 0.45

# Similarity required between observations of the same
# temporary candidate track.
CANDIDATE_THRESHOLD = 0.65

# Number of consistent observations required
# before creating a permanent Face ID.
CANDIDATE_CONFIRMATIONS = 3

# Final global check before creating a new Face ID
NEW_FACE_MAX_SIMILARITY = MATCH_THRESHOLD


# ============================================================
# Track-specific temporary candidates
# ============================================================

pending_candidates = {}


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# Embedding Utilities
# ============================================================

def normalize_embedding(embedding):
    """
    Normalize embedding to unit length.
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def embedding_to_blob(embedding):
    """
    Convert embedding to SQLite-compatible bytes.
    """

    embedding = normalize_embedding(embedding)

    return embedding.astype(
        np.float32
    ).tobytes()


def blob_to_embedding(blob):
    """
    Convert SQLite bytes back to NumPy embedding.
    """

    embedding = np.frombuffer(
        blob,
        dtype=np.float32
    )

    return normalize_embedding(
        embedding
    )


# ============================================================
# Cosine Similarity
# ============================================================

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    """

    embedding1 = normalize_embedding(
        embedding1
    )

    embedding2 = normalize_embedding(
        embedding2
    )

    denominator = (
        np.linalg.norm(embedding1)
        * np.linalg.norm(embedding2)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(
            embedding1,
            embedding2
        ) / denominator
    )


# ============================================================
# Find Best Registered Face
# ============================================================

def find_best_match(embedding):
    """
    Compare the new embedding with ALL stored embeddings.

    For each Face ID, the highest similarity among its
    stored embeddings is used.

    Returns:
        (face_id, similarity)

    or:

        (None, 0.0)
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT face_id, embedding
        FROM face_embeddings
    """)

    rows = cursor.fetchall()

    connection.close()

    # No embeddings stored yet
    if not rows:
        return None, 0.0

    best_face_id = None
    best_similarity = -1.0

    embedding = normalize_embedding(
        embedding
    )

    for face_id, embedding_blob in rows:

        stored_embedding = blob_to_embedding(
            embedding_blob
        )

        similarity = cosine_similarity(
            embedding,
            stored_embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_face_id = face_id

    return (
        best_face_id,
        best_similarity
    )


# ============================================================
# Register Permanent Face
# ============================================================

def register_face(embedding):
    """
    Create a new permanent Face ID.

    The embedding is stored in both:
        1. faces table
        2. face_embeddings table
    """

    connection = get_connection()
    cursor = connection.cursor()

    timestamp = datetime.now().isoformat()

    embedding_blob = embedding_to_blob(
        embedding
    )

    # --------------------------------------------------------
    # Create permanent Face ID
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO faces (
            first_seen,
            embedding
        )
        VALUES (?, ?)
        """,
        (
            timestamp,
            embedding_blob
        )
    )

    face_id = cursor.lastrowid

    # --------------------------------------------------------
    # Store embedding in face_embeddings table
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO face_embeddings (
            face_id,
            embedding,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            face_id,
            embedding_blob,
            timestamp
        )
    )

    connection.commit()
    connection.close()

    print(
        f"🆕 New face registered: "
        f"Face ID = {face_id}"
    )

    return face_id


# ============================================================
# Add New Embedding To Existing Face
# ============================================================

def add_embedding_to_face(face_id, embedding):
    """
    Store another embedding sample for an existing Face ID.

    This allows one person to have multiple embeddings
    representing different poses, angles and lighting.
    """

    connection = get_connection()
    cursor = connection.cursor()

    timestamp = datetime.now().isoformat()

    embedding_blob = embedding_to_blob(
        embedding
    )

    cursor.execute(
        """
        INSERT INTO face_embeddings (
            face_id,
            embedding,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            face_id,
            embedding_blob,
            timestamp
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# Create Track-Specific Candidate
# ============================================================

def create_candidate(track_id, embedding):
    """
    Create a temporary candidate specifically
    for one ByteTrack ID.
    """

    pending_candidates[track_id] = {
        "embedding": normalize_embedding(
            embedding
        ),
        "confirmations": 1
    }

    print(
        f"🟡 New temporary candidate: "
        f"Track ID = {track_id}, "
        f"Confirmation = 1/"
        f"{CANDIDATE_CONFIRMATIONS}"
    )


# ============================================================
# Update Track-Specific Candidate
# ============================================================

def update_candidate(track_id, embedding):
    """
    Update the candidate belonging to
    the specific Track ID.
    """

    candidate = pending_candidates[
        track_id
    ]

    old_embedding = candidate[
        "embedding"
    ]

    new_embedding = normalize_embedding(
        embedding
    )

    # Average old and new embeddings
    combined_embedding = (
        old_embedding + new_embedding
    )

    combined_embedding = normalize_embedding(
        combined_embedding
    )

    candidate[
        "embedding"
    ] = combined_embedding

    candidate[
        "confirmations"
    ] += 1

    confirmations = candidate[
        "confirmations"
    ]

    print(
        f"🟡 Track ID {track_id}: "
        f"Confirmation = "
        f"{confirmations}/"
        f"{CANDIDATE_CONFIRMATIONS}"
    )

    return confirmations


# ============================================================
# Promote Track Candidate
# ============================================================

def promote_candidate(track_id):
    """
    Convert a candidate into a permanent Face ID
    only after performing a final global check.
    """

    candidate = pending_candidates[
        track_id
    ]

    candidate_embedding = candidate[
        "embedding"
    ]

    print(
        f"🔎 Track ID {track_id}: "
        f"Performing final global identity check..."
    )

    # --------------------------------------------------------
    # FINAL GLOBAL CHECK
    # --------------------------------------------------------

    best_face_id, best_similarity = find_best_match(
        candidate_embedding
    )

    if best_face_id is not None:

        print(
            f"🔍 Track ID {track_id}: "
            f"Final global match: "
            f"Face ID = {best_face_id}, "
            f"Similarity = {best_similarity:.4f}"
        )

        # Candidate belongs to existing person
        if best_similarity >= NEW_FACE_MAX_SIMILARITY:

            print(
                f"✅ Track ID {track_id}: "
                f"Candidate belongs to existing "
                f"Face ID = {best_face_id}"
            )

            # Store this new observation
            # as another embedding of the same person.
            add_embedding_to_face(
                best_face_id,
                candidate_embedding
            )

            del pending_candidates[
                track_id
            ]

            return best_face_id

    else:

        print(
            f"ℹ️ Track ID {track_id}: "
            f"No permanent faces available "
            f"during final check."
        )

    # --------------------------------------------------------
    # Register only if final global check says
    # this is a sufficiently different identity.
    # --------------------------------------------------------

    print(
        f"🟢 Track ID {track_id}: "
        f"Candidate confirmed as a new identity."
    )

    face_id = register_face(
        candidate_embedding
    )

    del pending_candidates[
        track_id
    ]

    print(
        f"🔗 Track ID {track_id} "
        f"→ Face ID {face_id}"
    )

    return face_id


# ============================================================
# Main Recognition Function
# ============================================================

def recognize_or_register(
    track_id,
    embedding
):
    """
    Recognize an existing face or safely
    register a new face.

    Process:

    1. Validate embedding
    2. Search all stored face embeddings
    3. If strong match -> return existing Face ID
    4. If weak/no match -> use candidate
    5. Require consistent observations
    6. Perform final global identity check
    7. Register only if still a new identity
    """

    # ========================================================
    # Validate embedding
    # ========================================================

    if embedding is None:

        print(
            f"⚠️ Track ID {track_id}: "
            f"Invalid embedding"
        )

        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.shape != (512,):

        print(
            f"⚠️ Track ID {track_id}: "
            f"Invalid embedding shape: "
            f"{embedding.shape}"
        )

        return None

    embedding = normalize_embedding(
        embedding
    )

    # ========================================================
    # STEP 1
    # Search all permanently stored embeddings
    # ========================================================

    best_face_id, best_similarity = (
        find_best_match(
            embedding
        )
    )

    if best_face_id is not None:

        print(
            f"🔍 Track ID {track_id}: "
            f"Best match: "
            f"Face ID = {best_face_id}, "
            f"Similarity = "
            f"{best_similarity:.4f}"
        )

        # ====================================================
        # Strong match
        # ====================================================

        if best_similarity >= MATCH_THRESHOLD:

            print(
                f"✅ Track ID {track_id}: "
                f"Recognized existing face: "
                f"Face ID = {best_face_id}"
            )

            # Store this observation as another
            # embedding for the same person.
            add_embedding_to_face(
                best_face_id,
                embedding
            )

            # Remove pending candidate
            if track_id in pending_candidates:

                del pending_candidates[
                    track_id
                ]

            return best_face_id

        # ====================================================
        # Weak match
        # ====================================================

        print(
            f"⚠️ Track ID {track_id}: "
            f"Weak match with registered faces: "
            f"{best_similarity:.4f}"
        )

    else:

        print(
            f"ℹ️ Track ID {track_id}: "
            f"No registered faces yet."
        )

    # ========================================================
    # STEP 2
    # Track-specific candidate
    # ========================================================

    if track_id in pending_candidates:

        candidate = pending_candidates[
            track_id
        ]

        candidate_embedding = candidate[
            "embedding"
        ]

        candidate_similarity = (
            cosine_similarity(
                embedding,
                candidate_embedding
            )
        )

        print(
            f"🔎 Track ID {track_id}: "
            f"Candidate similarity = "
            f"{candidate_similarity:.4f}"
        )

        # ====================================================
        # Candidate matches
        # ====================================================

        if candidate_similarity >= CANDIDATE_THRESHOLD:

            confirmations = update_candidate(
                track_id,
                embedding
            )

            # =================================================
            # Candidate confirmed
            # =================================================

            if confirmations >= CANDIDATE_CONFIRMATIONS:

                return promote_candidate(
                    track_id
                )

            # Still waiting
            return None

        else:

            # Candidate does not look consistent.
            print(
                f"⚠️ Track ID {track_id}: "
                f"Candidate similarity too low. "
                f"Waiting for a consistent observation."
            )

            return None

    # ========================================================
    # STEP 3
    # Create a new candidate for this Track ID
    # ========================================================

    create_candidate(
        track_id,
        embedding
    )

    return None