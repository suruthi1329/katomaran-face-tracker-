from insightface.app import FaceAnalysis
import cv2
import numpy as np


print("🔄 Loading buffalo_l...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1)

print("✅ Model loaded")


# Open video
cap = cv2.VideoCapture("data/video_sample1.mp4")

if not cap.isOpened():
    print("❌ Could not open video")
    exit()


# ---------- FRAME 1 ----------
ret, frame1 = cap.read()

if not ret:
    print("❌ Could not read frame 1")
    exit()


# Skip some frames
for _ in range(30):
    cap.read()


# ---------- FRAME 2 ----------
ret, frame2 = cap.read()

if not ret:
    print("❌ Could not read frame 2")
    exit()

cap.release()


# Get faces
faces1 = app.get(frame1)
faces2 = app.get(frame2)

print(f"👤 Frame 1 faces: {len(faces1)}")
print(f"👤 Frame 2 faces: {len(faces2)}")


if len(faces1) == 0 or len(faces2) == 0:
    print("❌ Face not found in one of the frames")
    exit()


# Take first face from each frame
emb1 = faces1[0].embedding
emb2 = faces2[0].embedding


# Cosine similarity
similarity = np.dot(emb1, emb2) / (
    np.linalg.norm(emb1) * np.linalg.norm(emb2)
)


print("\n========== RESULT ==========")
print(f"Cosine Similarity: {similarity:.4f}")


if similarity >= 0.65:
    print("✅ Likely SAME PERSON")
else:
    print("❌ Likely DIFFERENT PERSON")

print("============================")