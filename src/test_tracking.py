from ultralytics import YOLO
import cv2

# Load face detection model
model = YOLO("models/yolov8n-face.pt")

# Open video
cap = cv2.VideoCapture("data/video_sample1.mp4")

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

print("✅ Video opened")
print("▶️ Starting ByteTrack...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False
    )

    # Draw boxes + Track IDs
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Face + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Tracking test completed")