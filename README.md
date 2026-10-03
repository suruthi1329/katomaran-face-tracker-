# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven unique visitor counter built for the Katomaran Hackathon. It processes a video file (development) or a live RTSP stream (interview/production), detects faces with YOLOv8, recognises them with InsightFace (ArcFace-based embeddings), tracks them with ByteTrack, and logs every entry and exit. The final output is the number of **unique** visitors.

**Demo video:** https://youtu.be/O5rbQZa9iQ4

---

## Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [AI Planning Document](#3-ai-planning-document)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [Setup Instructions](#7-setup-instructions)
8. [Configuration (config.json)](#8-configuration-configjson)
9. [Running the Project](#9-running-the-project)
10. [Database Schema](#10-database-schema)
11. [Logging System](#11-logging-system)
12. [Sample Output](#12-sample-output)
13. [Compute Load Estimate](#13-compute-load-estimate)
14. [Assumptions](#14-assumptions)
15. [Error Handling and Resilience](#15-error-handling-and-resilience)
16. [Privacy Considerations](#16-privacy-considerations)
17. [Future Improvements](#17-future-improvements)

---

## 1. Overview

The system processes a video stream to:

- Detect faces using YOLOv8
- Generate 512-dimensional face embeddings using InsightFace
- Track faces across frames using ByteTrack
- Automatically register new faces with a unique Face ID
- Recognise previously registered faces without counting them again
- Track each visitor until they leave the frame
- Log exactly one ENTRY and one EXIT event per visit
- Store metadata and embeddings in SQLite
- Save timestamped cropped face images locally
- Maintain a detailed `events.log`
- Support both a sample video file and an RTSP camera stream

The main objective is an accurate unique visitor count: the same person re-appearing later must not increment the count.

---

## 2. Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Face detection | YOLOv8 face model detects faces in each processed frame |
| 2 | Face recognition | InsightFace generates 512-D embeddings; matching uses cosine similarity |
| 3 | Auto-registration | A new face gets a unique Face ID, timestamp and embedding stored in the database |
| 4 | Tracking | ByteTrack keeps the same track ID for a person while they remain visible |
| 5 | Entry / exit detection | One ENTRY when a visitor is first identified, one EXIT when they leave or the stream ends |
| 6 | Event logging | ENTRY, EXIT, RECOGNIZED, TRACKING, EMBEDDING_GENERATED and REGISTERED events go to SQLite and `logs/events.log` |
| 7 | Face image storage | Cropped face images saved for every entry and exit event |
| 8 | Unique visitor count | Derived from the `faces` table; re-identification does not increment it |
| 9 | Configurable frame skipping | `frame_skip` in `config.json` controls how many frames are skipped between detection cycles |
| 10 | RTSP support | Uses the RTSP URL when provided, otherwise the sample video |
| 11 | Persistence | Registered faces are stored in SQLite, so identities survive restarts |

---

## 3. AI Planning Document

### 3.1 Problem

Count unique visitors in a video stream by automatically registering new faces, recognising returning ones, and logging every entry and exit.

### 3.2 Planned workflow

1. Capture a video frame (file or RTSP).
2. Skip frames according to `frame_skip`.
3. Detect faces with YOLOv8.
4. Track detections with ByteTrack to get stable track IDs.
5. Crop each tracked face.
6. Generate an InsightFace embedding.
7. Compare the embedding with registered identities using cosine similarity.
8. If no match: register a new identity. If match: reuse the existing Face ID.
9. Generate ENTRY (and TRACKING) events.
10. When a track is missing for `exit_missing_frames`, generate an EXIT event.
11. Store images, database rows and log lines.
12. Report the unique visitor count.

### 3.3 Technology choices

| Choice | Reason |
|--------|--------|
| YOLOv8n-face | Fast, lightweight, suitable for real-time CPU use |
| InsightFace (ArcFace) | State-of-the-art embeddings, far more reliable than the `face_recognition` library |
| ByteTrack | Lightweight tracker built into Ultralytics |
| SQLite | No server setup, enough for single-process deployment |
| JSON config | Simple, editable without touching code |

### 3.4 Key design decisions

- **Identity comes from embeddings, not from the tracker.** ByteTrack IDs can change, so permanent Face IDs are assigned by embedding matching.
- **Embeddings are generated at intervals**, not on every frame (`embedding_interval`), to reduce compute load.
- **Every database write is committed immediately** so an interruption loses at most the event in progress.

---

## 4. System Architecture

```text
                Video file / RTSP stream
                          |
                          v
                   OpenCV video input
                          |
                          v
              Frame skipping (config.json)
                          |
                          v
                 YOLOv8 face detection
                          |
                          v
                  ByteTrack tracking
                          |
                          v
                   Face crop (ROI)
                          |
                          v
            InsightFace 512-D face embedding
                          |
                          v
                  Cosine similarity
                          |
                +---------+---------+
                |                   |
                v                   v
          Existing face         New face
                |                   |
                v                   v
          Recognise ID         Register new ID
                |                   |
                +---------+---------+
                          |
                          v
                 ENTRY / TRACKING
                          |
                          v
                    EXIT event
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
     SQLite database  events.log   Face images
           |
           v
   Unique visitor count
```

---

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Face detection | YOLOv8 (face model) |
| Face recognition | InsightFace (ArcFace-based) |
| Tracking | ByteTrack |
| Video input | OpenCV |
| Database | SQLite |
| Configuration | JSON |
| Logging | Python `logging` + `logs/events.log` + local image store + database |
| Input | Video file (development), RTSP stream (interview) |

---

## 6. Project Structure

```text
face_tracker/
│
├── config.json
├── README.md
│
├── data/
│   ├── video_sample1.mp4
│   ├── events/              # saved entry / exit face crops
│   ├── sample_frames/
│   └── face_tracker.db      # SQLite database
│
├── docs/                    # planning notes, diagrams
│
├── logs/
│   └── events.log
│
├── models/
│   └── yolov8n-face.pt
│
├── src/
│   ├── pipeline.py          # main entry point and processing loop
│   ├── database.py          # SQLite access
│   ├── recognizer.py        # InsightFace embedding and matching
│   ├── event_logger.py      # image saving, DB and log writes
│   └── extract_frames.py    # helper to extract sample frames
│
└── venv/
```

---

## 7. Setup Instructions

**Requirements:** Python 3.10 or 3.11, and an internet connection on first run (InsightFace downloads its model).

### 7.1 Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd face_tracker
```

### 7.2 Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 7.3 Install dependencies

```bash
pip install numpy opencv-python ultralytics insightface onnxruntime scipy scikit-learn
```

### 7.4 Add the model and video

1. Place the YOLOv8 face weights at `models/yolov8n-face.pt`.
2. Place the sample video at `data/video_sample1.mp4`.

> If `insightface` fails to install on Windows with a "Microsoft Visual C++ 14.0 required" error, install the Microsoft C++ Build Tools and retry.

---

## 8. Configuration (config.json)

Sample `config.json`:

```json
{
    "video_path": "data/video_sample1.mp4",
    "rtsp_url": "",
    "frame_skip": 1,
    "embedding_interval": 30,
    "detection_confidence": 0.30,
    "exit_missing_frames": 30
}
```

| Parameter | Description |
|-----------|-------------|
| `video_path` | Path to the sample video file |
| `rtsp_url` | RTSP camera URL. If empty, `video_path` is used |
| `frame_skip` | Number of frames skipped between processing cycles |
| `embedding_interval` | Interval at which embeddings are generated for a tracked face |
| `detection_confidence` | Minimum YOLO detection confidence |
| `exit_missing_frames` | Number of consecutive missing frames before an EXIT event is triggered |

---

## 9. Running the Project

Make sure the virtual environment is activated.

**Sample video:**

```bash
python src\pipeline.py
```

**RTSP camera:** set the URL in `config.json`:

```json
"rtsp_url": "rtsp://USER:PASSWORD@CAMERA_IP:554/stream"
```

Then run the same command. The pipeline automatically uses the RTSP stream whenever `rtsp_url` is not empty.

Press **Q** in the preview window (if enabled) to stop. Remaining active visitors are closed with EXIT events.

---

## 10. Database Schema

Database file: `data/face_tracker.db`

| Table | Columns |
|-------|---------|
| `faces` | Face ID (primary key), first seen timestamp, face embedding |
| `events` | Event ID (primary key), Face ID, event type (`entry` / `exit`), timestamp, image path |

**Unique visitor count:**

```sql
SELECT COUNT(*) FROM faces;
```

---

## 11. Logging System

Three places are kept consistent:

1. **`logs/events.log`**: every critical system event
2. **Local image store**: cropped face image for every entry and exit
3. **SQLite `events` table**: metadata for every entry and exit

Each ENTRY and EXIT record contains the cropped face image, timestamp, event type and Face ID.

Example `events.log` lines:

```text
Face ID: 1 | Event: REGISTERED
Face ID: 1 | Event: ENTRY
Face ID: 1 | Event: TRACKING
Face ID: 1 | Event: RECOGNIZED
Face ID: 1 | Event: EXIT
```

Embedding generation events are also recorded.

---

## 12. Sample Output

Console summary from a completed run on the sample video:

```text
PIPELINE COMPLETED

Permanent Face IDs found: 15
New faces registered: 0
Existing faces recognized: 15
Unique ENTRY events: 15
Unique EXIT events: 15
```

> This run reused a database that already contained the 15 identities from an earlier run, which is why 0 new faces were registered. On a fresh database the same video registers the faces as new.

Sample output files are committed in the repository:

| Output | Location |
|--------|----------|
| Event log | `logs/events.log` |
| Entry / exit face images | `data/events/` |
| Database | `data/face_tracker.db` |

Add screenshots here:

```text
docs/screenshots/events_log.png
docs/screenshots/entry_images.png
docs/screenshots/db_rows.png
```

---

## 13. Compute Load Estimate

Approximate figures for a CPU-only laptop. Replace them with values measured on your machine (Task Manager).

| Resource | CPU only | With GPU (optional) |
|----------|----------|---------------------|
| YOLOv8n-face detection | ~30-60 ms per processed frame | ~5-10 ms |
| InsightFace embedding | ~100-300 ms, only when an embedding is generated | ~10-20 ms |
| RAM | ~1.5-2.5 GB | similar |
| VRAM | not used | ~1-2 GB |
| Typical CPU usage | 50-90% depending on `frame_skip` | 20-40% |

CPU usage depends on video resolution, number of faces, processing rate and `frame_skip`. On CPU-only machines, increasing `frame_skip` or `embedding_interval` reduces load.

---

## 14. Assumptions

- The camera gives a reasonably clear view of visitors.
- Faces are visible enough for reliable embedding generation.
- The sample video is used for development; an RTSP stream is provided in the interview environment.
- Temporary occlusion or extreme face angles can reduce recognition quality.
- Timestamps use the system clock.
- Two embeddings above the similarity threshold belong to the same person.
- SQLite is sufficient for single-process deployment.
- Face crops and metadata are stored locally.

---

## 15. Error Handling and Resilience

- Invalid video / RTSP input is detected and reported.
- Frames with no valid face embedding are skipped safely.
- Empty detections and empty recognition results are handled.
- Database writes are committed immediately, so data survives unexpected interruptions.
- Registered faces are loaded from the database at startup, so identities persist across restarts.
- The RTSP input path is designed to reconnect when the stream is lost.
- Active visitors receive EXIT events when the video ends or the program is stopped.

---

## 16. Privacy Considerations

The system processes facial biometric data. For production use:

- Restrict access to stored embeddings and face images.
- Control database access.
- Follow appropriate consent and privacy policies.
- Define data retention rules.

---

## 17. Future Improvements

- Stronger multi-view face recognition and re-identification across large angle changes
- GPU acceleration
- Multi-camera tracking
- Web-based monitoring dashboard and real-time analytics
- Centralised database deployment
- Advanced face quality filtering
- Improved duplicate identity handling

---

This project is a part of a hackathon run by https://katomaran.com