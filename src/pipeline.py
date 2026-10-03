import cv2
import numpy as np
import os
import sqlite3
import json
from datetime import datetime

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from recognizer import recognize_or_register
from event_logger import log_event


# =========================================================
# CONFIGURATION
# =========================================================

CONFIG_PATH = "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

VIDEO_PATH = CONFIG["video_path"]
RTSP_URL = CONFIG["rtsp_url"]

YOLO_MODEL_PATH = "models/yolov8n-face.pt"

DETECTION_CONFIDENCE = CONFIG["detection_confidence"]
FRAME_SKIP = CONFIG["frame_skip"]
EMBEDDING_INTERVAL = CONFIG["embedding_interval"]
EXIT_MISSING_FRAMES = CONFIG["exit_missing_frames"]

EVENT_IMAGE_DIR = "data/events"
DB_PATH = "data/face_tracker.db"

os.makedirs(EVENT_IMAGE_DIR, exist_ok=True)


# =========================================================
# LOAD YOLO
# =========================================================

print("Loading YOLOv8 Face model...")

yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLO loaded")


# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("Loading InsightFace...")

face_app = FaceAnalysis(name="buffalo_l")

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded")


# =========================================================
# IOU FUNCTION
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# =========================================================
# SAVE FACE CROP
# =========================================================

def save_face_crop(frame, box, face_id, event_type):

    if frame is None or box is None or face_id is None:
        return None

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = box

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"face_{face_id}_"
        f"{event_type}_"
        f"{timestamp}.jpg"
    )

    image_path = os.path.join(
        EVENT_IMAGE_DIR,
        filename
    )

    success = cv2.imwrite(
        image_path,
        face_crop
    )

    if not success:

        print(
            f"[WARNING] Could not save face image: "
            f"{image_path}"
        )

        return None

    print(
        f"[IMAGE] Saved face crop: "
        f"{image_path}"
    )

    return image_path


# =========================================================
# LOAD EXISTING FACE IDs
# =========================================================

def load_existing_face_ids():

    existing_ids = set()

    try:

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.execute(
            "SELECT face_id FROM faces"
        )

        rows = cursor.fetchall()

        for row in rows:
            existing_ids.add(row[0])

        conn.close()

    except Exception as e:

        print(
            f"[WARNING] Could not load existing "
            f"Face IDs: {e}"
        )

    return existing_ids


# =========================================================
# OPEN INPUT SOURCE
# =========================================================

SOURCE = RTSP_URL if RTSP_URL.strip() else VIDEO_PATH

print(f"[INFO] Input source: {SOURCE}")

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open input source: {SOURCE}"
    )

# =========================================================
# OUTPUT VIDEO
# =========================================================

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

output_path = "data/detection_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

print(f"[INFO] Output video: {output_path}")

# =========================================================
# TRACKING / RECOGNITION DATA
# =========================================================

frame_number = 0

# ByteTrack Track ID -> Permanent Face ID
track_to_face = {}

# Track ID -> averaged embedding
track_embeddings = {}

# Track ID -> recent embeddings
track_embedding_history = {}


# =========================================================
# EVENT STATE
# =========================================================

# One ENTRY per permanent Face ID
entry_logged_faces = set()

# One EXIT per permanent Face ID
exit_logged_faces = set()

# Currently active Face IDs
active_face_ids = set()

# Missing frame counter
missing_frames = {}

# Face ID -> latest frame and bounding box
latest_face_crops = {}

# Track ID + Face ID pairs
logged_tracking_pairs = set()

# Face IDs existing before this run
existing_face_ids = load_existing_face_ids()

# Face IDs that were actually recognized
recognized_face_ids = set()

# Face IDs registered during this run
registered_face_ids = set()


# =========================================================
# PIPELINE START
# =========================================================

print("\nStarting pipeline...\n")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # =====================================================
    # FRAME SKIP
    # =====================================================

    if FRAME_SKIP > 1:

        if frame_number % FRAME_SKIP != 0:
            continue


    # =====================================================
    # YOLO + BYTE TRACK
    # =====================================================

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=DETECTION_CONFIDENCE,
        verbose=False
    )

    if not results:

        video_writer.write(frame)

        cv2.imshow(
            "Face Tracker Pipeline",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    result = results[0]

    if result.boxes is None:

        video_writer.write(frame)

        cv2.imshow(
            "Face Tracker Pipeline",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # =====================================================
    # GET DETECTIONS
    # =====================================================

    boxes = result.boxes.xyxy.cpu().numpy()

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

    else:

        track_ids = [
            None
        ] * len(boxes)


    print(
        f"\nFrame {frame_number} | "
        f"Detections: {len(boxes)}"
    )


    # =====================================================
    # DRAW YOLO BOXES
    # =====================================================

    for box, track_id in zip(
        boxes,
        track_ids
    ):

        x1, y1, x2, y2 = box

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if track_id is not None:

            if track_id in track_to_face:

                label = (
                    f"Face ID: "
                    f"{track_to_face[track_id]}"
                )

            else:

                label = (
                    f"Track ID: "
                    f"{track_id}"
                )

        else:

            label = "Face"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # =====================================================
    # RECOGNITION INTERVAL
    # =====================================================

    if (
        EMBEDDING_INTERVAL > 1
        and frame_number % EMBEDDING_INTERVAL != 0
    ):

        video_writer.write(frame)

        cv2.imshow(
            "Face Tracker Pipeline",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # =====================================================
    # INSIGHTFACE
    # =====================================================

    insight_faces = face_app.get(frame)

    print(
        f"InsightFace faces: "
        f"{len(insight_faces)}"
    )


    # =====================================================
    # MATCH YOLO + INSIGHTFACE
    # =====================================================

    for box, track_id in zip(
        boxes,
        track_ids
    ):

        if track_id is None:
            continue

        x1, y1, x2, y2 = box

        yolo_box = [
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ]

        best_face = None
        best_iou = 0.0


        # -------------------------------------------------
        # Find matching InsightFace face
        # -------------------------------------------------

        for face in insight_faces:

            fx1, fy1, fx2, fy2 = face.bbox

            insight_box = [
                float(fx1),
                float(fy1),
                float(fx2),
                float(fy2)
            ]

            iou = calculate_iou(
                yolo_box,
                insight_box
            )

            if iou > best_iou:

                best_iou = iou
                best_face = face


        # -------------------------------------------------
        # No matching face
        # -------------------------------------------------

        if (
            best_face is None
            or best_iou < 0.30
        ):
            continue


        # -------------------------------------------------
        # Get embedding
        # -------------------------------------------------

        embedding = best_face.embedding

        if embedding is None:
            continue


        # -------------------------------------------------
        # Validate embedding
        # -------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding.shape != (512,):

            print(
                f"[WARNING] Track {track_id}: "
                f"Invalid embedding shape "
                f"{embedding.shape}"
            )

            continue


        # =================================================
        # LOG EMBEDDING GENERATION
        # =================================================

        log_event(
            None,
            "EMBEDDING_GENERATED"
        )


        # =================================================
        # NEW TRACK
        # =================================================

        if track_id not in track_to_face:

            print(
                f"\nNew Track ID "
                f"{track_id} needs recognition..."
            )


            # ---------------------------------------------
            # Create embedding history
            # ---------------------------------------------

            if track_id not in track_embedding_history:

                track_embedding_history[
                    track_id
                ] = []


            # ---------------------------------------------
            # Add embedding
            # ---------------------------------------------

            track_embedding_history[
                track_id
            ].append(embedding)


            # ---------------------------------------------
            # Keep latest 2 embeddings
            # ---------------------------------------------

            if len(
                track_embedding_history[
                    track_id
                ]
            ) > 2:

                track_embedding_history[
                    track_id
                ].pop(0)


            history = (
                track_embedding_history[
                    track_id
                ]
            )


            # ---------------------------------------------
            # Wait for 2 observations
            # ---------------------------------------------

            if len(history) < 2:

                print(
                    f"Waiting for Track ID "
                    f"{track_id}: "
                    f"{len(history)}/2 embeddings"
                )

                continue


            # ---------------------------------------------
            # Average embeddings
            # ---------------------------------------------

            averaged_embedding = np.mean(
                history,
                axis=0
            ).astype(np.float32)


            # Normalize averaged embedding
            norm = np.linalg.norm(
                averaged_embedding
            )

            if norm > 0:

                averaged_embedding = (
                    averaged_embedding / norm
                )


            # =================================================
            # RECOGNIZE / REGISTER
            # =================================================

            face_id = recognize_or_register(
                track_id,
                averaged_embedding
            )

            if face_id is None:
                continue


            # =================================================
            # DETERMINE EVENT TYPE
            # =================================================

            if face_id in existing_face_ids:

                # Existing person
                recognized_face_ids.add(
                    face_id
                )

                log_event(
                    face_id,
                    "RECOGNIZED"
                )

            else:

                # New person
                registered_face_ids.add(
                    face_id
                )

                existing_face_ids.add(
                    face_id
                )

                log_event(
                    face_id,
                    "REGISTERED"
                )


            # =================================================
            # ENTRY EVENT
            # =================================================

            if face_id not in entry_logged_faces:

                image_path = save_face_crop(
                    frame,
                    box,
                    face_id,
                    "ENTRY"
                )

                log_event(
                    face_id,
                    "ENTRY",
                    image_path
                )

                entry_logged_faces.add(
                    face_id
                )


            # =================================================
            # ACTIVE FACE
            # =================================================

            active_face_ids.add(
                face_id
            )

            missing_frames[
                face_id
            ] = 0


            # =================================================
            # SAVE TRACK -> FACE
            # =================================================

            track_to_face[
                track_id
            ] = face_id

            track_embeddings[
                track_id
            ] = averaged_embedding


            # =================================================
            # SAVE LATEST CROP
            # =================================================

            latest_face_crops[
                face_id
            ] = (
                frame.copy(),
                box.copy()
            )


            # =================================================
            # TRACKING EVENT
            # =================================================

            tracking_key = (
                track_id,
                face_id
            )

            if tracking_key not in logged_tracking_pairs:

                log_event(
                    face_id,
                    "TRACKING"
                )

                logged_tracking_pairs.add(
                    tracking_key
                )


            print(
                f"Track ID {track_id} "
                f"-> Face ID {face_id}"
            )


        # =================================================
        # EXISTING TRACK
        # =================================================

        else:

            face_id = track_to_face[
                track_id
            ]


            # ---------------------------------------------
            # Update latest crop
            # ---------------------------------------------

            latest_face_crops[
                face_id
            ] = (
                frame.copy(),
                box.copy()
            )


            # ---------------------------------------------
            # Face visible
            # ---------------------------------------------

            active_face_ids.add(
                face_id
            )

            missing_frames[
                face_id
            ] = 0


            # ---------------------------------------------
            # TRACKING EVENT
            # ---------------------------------------------

            tracking_key = (
                track_id,
                face_id
            )

            if tracking_key not in logged_tracking_pairs:

                log_event(
                    face_id,
                    "TRACKING"
                )

                logged_tracking_pairs.add(
                    tracking_key
                )


            print(
                f"Track ID {track_id} "
                f"-> Face ID {face_id}"
            )


    # =====================================================
    # EXIT DETECTION
    # =====================================================

    visible_face_ids = set()

    for track_id in track_ids:

        if (
            track_id is not None
            and track_id in track_to_face
        ):

            visible_face_ids.add(
                track_to_face[track_id]
            )


    # -----------------------------------------------------
    # Check active faces
    # -----------------------------------------------------

    for face_id in list(active_face_ids):

        if face_id in visible_face_ids:

            missing_frames[
                face_id
            ] = 0

        else:

            missing_frames[
                face_id
            ] = (
                missing_frames.get(
                    face_id,
                    0
                ) + 1
            )


            # ---------------------------------------------
            # EXIT threshold
            # ---------------------------------------------

            if (
                missing_frames[face_id]
                >= EXIT_MISSING_FRAMES
            ):

                if face_id not in exit_logged_faces:

                    image_path = None

                    if face_id in latest_face_crops:

                        last_frame, last_box = (
                            latest_face_crops[
                                face_id
                            ]
                        )

                        image_path = save_face_crop(
                            last_frame,
                            last_box,
                            face_id,
                            "EXIT"
                        )


                    # -------------------------------------
                    # Log EXIT
                    # -------------------------------------

                    log_event(
                        face_id,
                        "EXIT",
                        image_path
                    )

                    exit_logged_faces.add(
                        face_id
                    )

                    print(
                        f"[EXIT] Face ID {face_id} "
                        f"left the frame"
                    )


                active_face_ids.discard(
                    face_id
                )

                missing_frames.pop(
                    face_id,
                    None
                )


    # =====================================================
    # UPDATE DISPLAY LABELS
    # =====================================================

    for box, track_id in zip(
        boxes,
        track_ids
    ):

        if track_id is None:
            continue

        if track_id not in track_to_face:
            continue

        x1, y1, x2, y2 = box

        x1 = int(x1)
        y1 = int(y1)

        cv2.putText(
            frame,
            f"Face ID: {track_to_face[track_id]}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # =====================================================
    # SAVE PROCESSED FRAME TO OUTPUT VIDEO
    # =====================================================

    video_writer.write(frame)

    # =====================================================
    # DISPLAY VIDEO
    # =====================================================

    cv2.imshow(
        "Face Tracker Pipeline",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================================================
# END OF VIDEO
# =========================================================

for face_id in list(active_face_ids):

    if face_id in exit_logged_faces:
        continue

    image_path = None

    if face_id in latest_face_crops:

        last_frame, last_box = (
            latest_face_crops[
                face_id
            ]
        )

        image_path = save_face_crop(
            last_frame,
            last_box,
            face_id,
            "EXIT"
        )


    log_event(
        face_id,
        "EXIT",
        image_path
    )

    exit_logged_faces.add(
        face_id
    )

    print(
        f"[EXIT] Face ID {face_id} "
        f"exited at end of video"
    )


# =========================================================
# CLEANUP
# =========================================================

video_writer.release()
cap.release()

cv2.destroyAllWindows()

print(f"[INFO] Output video saved: {output_path}")


# =========================================================
# FINAL SUMMARY
# =========================================================

permanent_face_ids = set(
    track_to_face.values()
)

print("\n========================================")
print("PIPELINE COMPLETED")
print("========================================")

print(
    f"Permanent Face IDs found: "
    f"{len(permanent_face_ids)}"
)

print(
    f"New faces registered: "
    f"{len(registered_face_ids)}"
)

print(
    f"Existing faces recognized: "
    f"{len(recognized_face_ids)}"
)

print(
    f"Unique ENTRY events: "
    f"{len(entry_logged_faces)}"
)

print(
    f"Unique EXIT events: "
    f"{len(exit_logged_faces)}"
)

print("========================================")