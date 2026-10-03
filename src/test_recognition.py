from insightface.app import FaceAnalysis
import cv2
import numpy as np

print("🔄 Loading InsightFace buffalo_l...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1)

print("✅ InsightFace loaded")

# Read sample video
cap = cv2.VideoCapture("data/video_sample1.mp4")

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

# Read one frame
ret, frame = cap.read()

cap.release()

if not ret:
    print("❌ Could not read frame")
    exit()

print("✅ Frame captured")

# Detect faces and generate embeddings
faces = app.get(frame)

print(f"👤 Faces detected: {len(faces)}")

for i, face in enumerate(faces):
    embedding = face.embedding

    print(f"\nFace {i + 1}")
    print("Embedding shape:", embedding.shape)
    print("Embedding length:", len(embedding))
    print("Embedding norm:", np.linalg.norm(embedding))

print("\n✅ Recognition / embedding test completed")