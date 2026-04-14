# MoodLens-CV

A real-time emotion and stress monitoring desktop app that uses your webcam to track facial expressions, gaze, and stress levels — then adapts your environment to help you stay focused and calm.

---

## Features

### Emotion & Stress Detection
- **Emotion recognition** via [DeepFace](https://github.com/serengil/deepface) — detects happy, sad, angry, fear, surprise, disgust, neutral
- **Facial Action Units (AUs)** computed from [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) landmarks for a nuanced, continuous stress score
- **Predictive stress model** — learns your personal stress patterns over a session and predicts rising stress 2–5 minutes before it peaks, using a GradientBoosting classifier (falls back to a heuristic until enough data is collected)

### Gaze Tracking
- Detects when you look away from the screen using head yaw estimation
- After **60 seconds** of looking away, the Lock-In widget appears as a nudge

### Ambient Feedback
- **Warm tint overlay** — a full-screen colour wash that intensifies with stress level, click-through so it never interrupts your work
- **Edge glow** — a red glow around the screen border that reacts to distraction events
- **Sound alert** — plays rain ambient noise when sustained stress is detected (toggleable)

### Focus Tools
- **Lock-In Challenge** — a floating 10-minute focus timer that tracks screen time, nudges you when distracted, and triggers confetti on completion; dismissable via the × button
- **Breathing break** — a guided breathing exercise offered after 15 minutes of sustained stress

### Dashboard
- Live emotion bar chart and stress gauge
- Focus time, distraction count, and average stress metrics
- Hourly stress heatmap
- Emotion–app correlation log (tracks which app you were using with each emotion)
- Predictive stress insight panel
- Sound toggle

---

## Requirements

- macOS (gaze-away app detection uses `osascript`)
- Python 3.12
- Webcam

### Python dependencies

```
opencv-python
mediapipe
deepface
PyQt5
PyQt5-Qt5
numpy
scikit-learn
pynput
```

Install with:

```bash
pip install opencv-python mediapipe deepface PyQt5 numpy scikit-learn pynput
```

### Model file

The MediaPipe face landmarker model must be present in the project directory:

```
face_landmarker.task
```

Download from [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models) and place it alongside `moodlens_gui.py`.

---

## Running

```bash
python3.12 moodlens_gui.py
```

---

## Project Structure

| File | Description |
|---|---|
| `moodlens_gui.py` | Main application — GUI, webcam loop, gaze/stress/emotion wiring |
| `dashboard.py` | Dashboard widget with charts, metrics, and insights |
| `stress_predictor.py` | Predictive stress model (heuristic + trained GradientBoosting) |
| `test_breathing.py` | Standalone test for the breathing exercise overlay |
| `face_landmarker.task` | MediaPipe face landmark model |
| `stress_alert.mp3` | Alert sound played on sustained stress |

---

## How It Works

1. A background thread captures webcam frames and runs two parallel pipelines:
   - **DeepFace** analyses every 5th frame for dominant emotion
   - **MediaPipe FaceLandmarker** runs on every 3rd frame to compute Action Units and gaze direction
2. AU scores are blended with a stress weight formula to produce a 0–100 stress score each cycle
3. The stress predictor snapshots features every 10 seconds and builds a rolling model of your session
4. When predicted or measured stress crosses thresholds, ambient overlays (tint, glow, sound) activate proportionally
5. Gaze direction is tracked continuously; looking away for 60 seconds triggers the Lock-In nudge and counts as a distraction
6. All emotion/stress/app/time data is logged internally and visualised live on the Dashboard tab
