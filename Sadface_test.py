import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

# Run DeepFace every 5 frames
ANALYZE_EVERY_N_FRAMES = 5

# Run MediaPipe facial geometry analysis every 3 frames
AU_ANALYZE_EVERY_N_FRAMES = 3

# Smoothing factor for facial signals
AU_SMOOTH_ALPHA = 0.6

# Landmark indices used for sadness cues
_LM = {
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "left_lid_upper": 159,
    "left_lid_lower": 145,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_lid_upper": 386,
    "right_lid_lower": 374,
    "left_brow_inner": 107,
    "right_brow_inner": 336,
    "mouth_left": 61,
    "mouth_right": 291,
    "lip_upper_in": 13,
    "lip_lower_in": 14,
}

# Path to MediaPipe face model
_MODEL_PATH = str(pathlib.Path(__file__).parent / "face_landmarker.task")


def _create_face_landmarker():
    # Build MediaPipe face landmarker
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
    # Convert normalized landmark to pixel coordinates
    lm = landmarks[_LM[key]]
    return np.array([lm.x * w, lm.y * h])


def compute_sad_aus_from_landmarks(landmarks, img_w: int, img_h: int) -> dict:
    # Point helper
    p = lambda key: _pt(landmarks, key, img_w, img_h)

    # Normalize by inter-eye distance
    eye_l = p("left_eye_inner")
    eye_r = p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}

    def n(d):
        return d / inter_eye

    aus = {}

    # Smaller eye opening can be a sadness cue
    upper_l, lower_l = p("left_lid_upper"), p("left_lid_lower")
    upper_r, lower_r = p("right_lid_upper"), p("right_lid_lower")
    open_l = n(abs(lower_l[1] - upper_l[1]))
    open_r = n(abs(lower_r[1] - upper_r[1]))
    avg_open = (open_l + open_r) / 2.0
    aus["EyeClosure"] = float(np.clip((0.28 - avg_open) / 0.18 * 5.0, 0.0, 5.0))

    # Mouth corners lower in sadness
    corner_l = p("mouth_left")
    corner_r = p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2.0
    corner_avg_y = (corner_l[1] + corner_r[1]) / 2.0
    corner_rise = n(lip_mid_y - corner_avg_y)

    # AU12 smile is included as a negative signal
    aus["AU12"] = float(np.clip(corner_rise * 8.0 + 2.5, 0.0, 5.0))

    # Mouth corner depressor approximation
    aus["MouthCornerDepressor"] = float(np.clip((-corner_rise + 0.15) / 0.45 * 5.0, 0.0, 5.0))

    # Smaller mouth width can correlate with subdued expression
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    aus["LipStretchLow"] = float(np.clip((1.25 - lip_width) / 0.35 * 5.0, 0.0, 5.0))

    # Raised inner brows are common in sadness
    brow_l = p("left_brow_inner")
    brow_r = p("right_brow_inner")
    lid_top_y = (p("left_lid_upper")[1] + p("right_lid_upper")[1]) / 2.0
    brow_y = (brow_l[1] + brow_r[1]) / 2.0
    brow_raise = n(lid_top_y - brow_y)
    aus["InnerBrowRaise"] = float(np.clip((brow_raise - 0.42) / 0.18 * 5.0, 0.0, 5.0))

    return aus


def detect_sad_aus(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> dict | None:
    # Convert BGR frame to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect landmarks
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None

    # Compute sadness-related geometry
    h, w = frame_bgr.shape[:2]
    return compute_sad_aus_from_landmarks(result.face_landmarks[0], w, h)


def compute_sad_score(au_row: dict, deepface_sad: float | None) -> float:
    # Normalize features to 0-1
    eye_closure = float(np.clip(au_row.get("EyeClosure", 0.0) / 5.0, 0.0, 1.0))
    mouth_down = float(np.clip(au_row.get("MouthCornerDepressor", 0.0) / 5.0, 0.0, 1.0))
    lip_stretch_low = float(np.clip(au_row.get("LipStretchLow", 0.0) / 5.0, 0.0, 1.0))
    inner_brow_raise = float(np.clip(au_row.get("InnerBrowRaise", 0.0) / 5.0, 0.0, 1.0))
    smile = float(np.clip(au_row.get("AU12", 0.0) / 5.0, 0.0, 1.0))

    # Weighted sadness score
    au_score = (
        eye_closure * 0.20 +
        mouth_down * 0.35 +
        lip_stretch_low * 0.15 +
        inner_brow_raise * 0.30 -
        smile * 0.20
    ) * 100.0

    au_score = float(np.clip(au_score, 0.0, 100.0))

    # If DeepFace unavailable, use AU score only
    if deepface_sad is None:
        return au_score

    # Blend DeepFace sadness with geometry-based score
    return float(np.clip(0.6 * deepface_sad + 0.4 * au_score, 0.0, 100.0))


def sad_label(score: float):
    # Convert score into label and color
    if score < 35:
        return "Not Sad", (0, 200, 0)
    elif score < 70:
        return "Slightly Sad", (0, 255, 255)
    else:
        return "Sad", (255, 0, 0)


def draw_face_box(frame, result):
    # Draw face rectangle from DeepFace region output
    region = result.get("region", {})
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_sad_results(frame, deepface_result, sad_score, au_row):
    # Get status label and display color
    label, color = sad_label(sad_score)

    # Draw main sadness label and score
    cv2.putText(frame, f"Sad Face: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Sadness Score: {sad_score:.0f}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Draw score bar
    bar_width = int(sad_score * 2)
    cv2.rectangle(frame, (20, 95), (220, 115), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 95), (20 + bar_width, 115), color, -1)

    # Show DeepFace sadness if present
    if deepface_result:
        emotions = deepface_result.get("emotion", {})
        sad_val = emotions.get("sad", 0.0)
        dominant = deepface_result.get("dominant_emotion", "")
        cv2.putText(frame, f"DeepFace sad: {sad_val:.1f}%", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Dominant emotion: {dominant}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Show AU/geometry values
    if au_row:
        cv2.putText(frame, f"Eye Closure: {au_row.get('EyeClosure', 0.0):.2f}/5", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Mouth Down: {au_row.get('MouthCornerDepressor', 0.0):.2f}/5", (20, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Inner Brow Raise: {au_row.get('InnerBrowRaise', 0.0):.2f}/5", (20, 251),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Low Lip Stretch: {au_row.get('LipStretchLow', 0.0):.2f}/5", (20, 274),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    # Create landmark detector
    landmarker = _create_face_landmarker()
    print("Press 'q' to quit.")

    # State variables
    frame_count = 0
    timestamp_ms = 0
    last_result = None
    last_au_row = None
    last_sad_score = None
    last_deepface_sad = None

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        timestamp_ms += 33

        # Run DeepFace periodically
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
                last_deepface_sad = last_result.get("emotion", {}).get("sad", 0.0)
            except Exception:
                last_result = None
                last_deepface_sad = None

        # Run MediaPipe AU-based sadness detection
        if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
            raw_aus = detect_sad_aus(landmarker, frame, timestamp_ms)
            if raw_aus:
                if last_au_row is None:
                    last_au_row = raw_aus
                else:
                    # Smooth values to reduce noise
                    for au in raw_aus:
                        prev = last_au_row.get(au, raw_aus[au])
                        last_au_row[au] = AU_SMOOTH_ALPHA * prev + (1 - AU_SMOOTH_ALPHA) * raw_aus[au]

                # Compute final sadness score
                last_sad_score = compute_sad_score(last_au_row, last_deepface_sad)

        # Draw face box
        if last_result:
            draw_face_box(frame, last_result)

        # Draw output or no-face message
        if last_sad_score is not None and last_au_row is not None:
            draw_sad_results(frame, last_result, last_sad_score, last_au_row)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show live result
        cv2.imshow("Sad Face Detector", frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
