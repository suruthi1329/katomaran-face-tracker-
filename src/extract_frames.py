import cv2
import os

video_path = "data/video_sample1.mp4"
output_dir = "data/sample_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frames_to_save = [0, 40, 80, 120, 160, 200, 239]

for frame_number in frames_to_save:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if ret:
        output_path = os.path.join(
            output_dir,
            f"frame_{frame_number}.jpg"
        )

        cv2.imwrite(output_path, frame)
        print("Saved:", output_path)

cap.release()

print("Done.")