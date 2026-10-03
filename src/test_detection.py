from ultralytics import YOLO
import cv2

# YOLOv8 Face model
model = YOLO("models/yolov8n-face.pt")

# Input video
video_path = "data/video_sample1.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "data/detection_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("✅ Video opened")
print("▶️ Starting face detection...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Face detection
    results = model(frame, conf=0.4, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save output frame
    out.write(annotated_frame)

    # Display
    cv2.imshow("YOLOv8 Face Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Detection completed")
print(f"📁 Output saved at: {output_path}")