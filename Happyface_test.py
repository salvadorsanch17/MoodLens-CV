import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

# Run DeepFace emotion analysis every 5 frames
ANALYZE_EVERY_N_FRAMES = 5

# Run MediaPipe geometric smile analysis every 3 frames
AU_ANALYZE_EVERY_N_FRAMES = 3

# Exponential smoothing factor for facial measurements
AU_SMOOTH_ALPHA = 0.6

# Landmark indices used for happy/smile detection
_LM = {
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "mouth_left": 61,
    "mouth_right": 291,
    "lip_upper_in": 13,
    "lip_lower_in": 14,
}

# Path to the MediaPipe face landmarker model
_MODEL_PATH = str(pathlib.Path(__file__).parent / "face_landmarker.task")


def _create_face_landmarker():
    # Create and configure the MediaPipe face landmarker
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _pt(landmarks, key, w, h):
    # Convert a normalized landmark to pixel coordinates
    lm = landmarks[_LM[key]]
    return np.array([lm.x * w, lm.y * h])


def compute_happy_aus_from_landmarks(landmarks, img_w: int, img_h: int) -> dict:
    # Helper to fetch pixel positions for named landmarks
    p = lambda key: _pt(landmarks, key, img_w, img_h)

    # Use inner eye distance as a scale normalizer
    eye_l = p("left_eye_inner")
    eye_r = p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}

    def n(d):
        # Normalize a distance by inter-eye distance
        return d / inter_eye

    aus = {}

    # Measure lip corner rise to estimate smiling
    corner_l = p("mouth_left")
    corner_r = p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2.0
    corner_avg_y = (corner_l[1] + corner_r[1]) / 2.0
    corner_rise = n(lip_mid_y - corner_avg_y)

    # AU12 = lip corner puller, often associated with smiling
    aus["AU12"] = float(np.clip(corner_rise * 8.0 + 2.5, 0.0, 5.0))

    # Wider mouth often indicates a stronger smile
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    aus["SmileWidth"] = float(np.clip((lip_width - 1.1) / 0.5 * 5.0, 0.0, 5.0))

    return aus


def detect_happy_aus(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> dict | None:
    # Convert OpenCV BGR frame to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Wrap the frame in a MediaPipe image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Run face landmark detection
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None

    # Compute smile-related values from the first detected face
    h, w = frame_bgr.shape[:2]
    return compute_happy_aus_from_landmarks(result.face_landmarks[0], w, h)


def compute_happy_score(au_row: dict, deepface_happy: float | None) -> float:
    # Normalize AU values to 0-1
    au12 = float(np.clip(au_row.get("AU12", 0.0) / 5.0, 0.0, 1.0))
    smile_width = float(np.clip(au_row.get("SmileWidth", 0.0) / 5.0, 0.0, 1.0))

    # Weighted AU-based happiness score
    au_score = (au12 * 0.75 + smile_width * 0.25) * 100.0

    # If DeepFace isn't available, just use AU score
    if deepface_happy is None:
        return float(np.clip(au_score, 0.0, 100.0))

    # Blend DeepFace happiness with geometric smile score
    return float(np.clip(0.6 * deepface_happy + 0.4 * au_score, 0.0, 100.0))


def happy_label(score: float):
    # Convert numeric score into a label and display color
    if score < 35:
        return "Not Happy", (0, 0, 255)
    elif score < 70:
        return "Slightly Happy", (0, 255, 255)
    else:
        return "Happy", (0, 255, 0)


def draw_face_box(frame, result):
    # Draw a box around the detected face using DeepFace region data
    region = result.get("region", {})
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_happy_results(frame, deepface_result, happy_score, au_row):
    # Get display label and color
    label, color = happy_label(happy_score)

    # Main title and score
    cv2.putText(frame, f"Happy Face: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Happiness Score: {happy_score:.0f}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Draw score bar
    bar_width = int(happy_score * 2)
    cv2.rectangle(frame, (20, 95), (220, 115), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 95), (20 + bar_width, 115), color, -1)

    # Draw DeepFace outputs if available
    if deepface_result:
        emotions = deepface_result.get("emotion", {})
        happy_val = emotions.get("happy", 0.0)
        dominant = deepface_result.get("dominant_emotion", "")
        cv2.putText(frame, f"DeepFace happy: {happy_val:.1f}%", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Dominant emotion: {dominant}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Draw AU-based details
    if au_row:
        cv2.putText(frame, f"AU12 Smile: {au_row.get('AU12', 0.0):.2f}/5", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Smile Width: {au_row.get('SmileWidth', 0.0):.2f}/5", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    # Create the face landmarker
    landmarker = _create_face_landmarker()
    print("Press 'q' to quit.")

    # Tracking variables
    frame_count = 0
    timestamp_ms = 0
    last_result = None
    last_au_row = None
    last_happy_score = None
    last_deepface_happy = None

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        timestamp_ms += 33

        # Periodically run DeepFace emotion analysis
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
                last_deepface_happy = last_result.get("emotion", {}).get("happy", 0.0)
            except Exception:
                last_result = None
                last_deepface_happy = None

        # Periodically run MediaPipe smile analysis
        if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
            raw_aus = detect_happy_aus(landmarker, frame, timestamp_ms)
            if raw_aus:
                if last_au_row is None:
                    last_au_row = raw_aus
                else:
                    # Smooth values to reduce jitter
                    for au in raw_aus:
                        prev = last_au_row.get(au, raw_aus[au])
                        last_au_row[au] = AU_SMOOTH_ALPHA * prev + (1 - AU_SMOOTH_ALPHA) * raw_aus[au]

                # Compute combined happy score
                last_happy_score = compute_happy_score(last_au_row, last_deepface_happy)

        # Draw face box if DeepFace found a face
        if last_result:
            draw_face_box(frame, last_result)

        # Draw results or fallback message
        if last_happy_score is not None and last_au_row is not None:
            draw_happy_results(frame, last_result, last_happy_score, last_au_row)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show output window
        cv2.imshow("Happy Face Detector", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
