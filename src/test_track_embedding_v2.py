from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2
import numpy as np


print("🔄 Loading YOLOv8 Face...")
yolo = YOLO("models/yolov8n-face.pt")
print("✅ YOLO loaded")


print("🔄 Loading InsightFace buffalo_l...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1)
print("✅ InsightFace loaded")


cap = cv2.VideoCapture("data/video_sample1.mp4")

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

print("✅ Video opened")
print("▶️ Starting Track + Embedding V2 test...")


# Store one embedding for each ByteTrack ID
track_embeddings = {}

frame_count = 0

# Run InsightFace only once every 15 frames
recognition_interval = 15


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # -----------------------------------------
    # 1. YOLO + ByteTrack
    # -----------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or not result.boxes.is_track:
        continue

    yolo_boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()


    # -----------------------------------------
    # 2. Run InsightFace periodically
    # -----------------------------------------

    if frame_count % recognition_interval != 0:
        continue

    faces = app.get(frame)

    print(
        f"\nFrame {frame_count} | "
        f"YOLO tracks: {len(track_ids)} | "
        f"InsightFace faces: {len(faces)}"
    )


    # -----------------------------------------
    # 3. Match YOLO track with InsightFace face
    # -----------------------------------------

    for yolo_box, track_id in zip(yolo_boxes, track_ids):

        yolo_box = yolo_box.tolist()

        best_face = None
        best_iou = 0.0

        for face in faces:

            face_box = face.bbox.tolist()

            iou = calculate_iou(
                yolo_box,
                face_box
            )

            if iou > best_iou:
                best_iou = iou
                best_face = face


        # No reliable match
        if best_face is None or best_iou < 0.3:
            continue


        # -----------------------------------------
        # 4. Get ArcFace embedding
        # -----------------------------------------

        embedding = best_face.embedding


        # -----------------------------------------
        # 5. First embedding for this Track ID
        # -----------------------------------------

        if track_id not in track_embeddings:

            track_embeddings[track_id] = embedding

            print(
                f"🆕 Track ID {track_id} "
                f"→ embedding stored "
                f"(IoU={best_iou:.2f})"
            )


        # -----------------------------------------
        # 6. Compare with previous embedding
        # -----------------------------------------

        else:

            old_embedding = track_embeddings[track_id]

            similarity = np.dot(
                old_embedding,
                embedding
            ) / (
                np.linalg.norm(old_embedding)
                *
                np.linalg.norm(embedding)
            )

            print(
                f"🔍 Track ID {track_id} "
                f"→ similarity = {similarity:.4f} "
                f"(IoU={best_iou:.2f})"
            )


cap.release()

print("\n================================")
print("✅ Track + Embedding V2 completed")
print(f"👤 Track IDs found: {len(track_embeddings)}")
print("================================")