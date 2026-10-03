from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2
import numpy as np


# -----------------------------
# Load YOLO Face Detector
# -----------------------------
print("🔄 Loading YOLOv8 Face...")

yolo = YOLO("models/yolov8n-face.pt")

print("✅ YOLO loaded")


# -----------------------------
# Load InsightFace
# -----------------------------
print("🔄 Loading buffalo_l...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1)

print("✅ InsightFace loaded")


# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture("data/video_sample1.mp4")

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

print("✅ Video opened")
print("▶️ Starting Track + Embedding test...")


# Store embedding for each track
track_embeddings = {}

frame_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO + ByteTrack
    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    result = results[0]

    # Check if tracking IDs exist
    if result.boxes is None or not result.boxes.is_track:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):

        x1, y1, x2, y2 = map(int, box)

        # Keep coordinates inside frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Crop face
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Generate InsightFace embedding
        faces = app.get(face_crop)

        if len(faces) == 0:
            continue

        # Select largest face inside crop
        face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        embedding = face.embedding

        # First time seeing this Track ID
        if track_id not in track_embeddings:

            track_embeddings[track_id] = embedding

            print(
                f"🆕 Track ID {track_id} "
                f"→ embedding stored"
            )

        # Already have embedding
        else:

            old_embedding = track_embeddings[track_id]

            similarity = np.dot(
                old_embedding,
                embedding
            ) / (
                np.linalg.norm(old_embedding) *
                np.linalg.norm(embedding)
            )

            # Print every 30 frames
            if frame_count % 30 == 0:

                print(
                    f"🔍 Track ID {track_id} "
                    f"→ similarity = {similarity:.4f}"
                )


cap.release()

print("\n==============================")
print("✅ Track + Embedding test completed")
print(f"👤 Track IDs found: {len(track_embeddings)}")
print("==============================")