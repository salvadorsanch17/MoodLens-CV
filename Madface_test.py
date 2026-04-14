import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

# Run DeepFace every 5 frames
ANALYZE_EVERY_N_FRAMES = 5

# Run MediaPipe AU-style analysis every 3 frames
AU_ANALYZE_EVERY_N_FRAMES = 3

# Smoothing factor for AU values
AU_SMOOTH_ALPHA = 0.6

# Landmark indices used for anger-related features
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

# MediaPipe model path
_MODEL_PATH = str(pathlib.Path(__file__).parent / "face_landmarker.task")


def _create_face_landmarker():
    # Create MediaPipe face landmarker for video mode
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
    # Convert landmark from normalized coordinates to pixel coordinates
    lm = landmarks[_LM[key]]
    return np.array([lm.x * w, lm.y * h])


def compute_angry_aus_from_landmarks(landmarks, img_w: int, img_h: int) -> dict:
    # Helper to access landmark points
    p = lambda key: _pt(landmarks, key, img_w, img_h)

    # Normalize all distances by inner-eye distance
    eye_l = p("left_eye_inner")
    eye_r = p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}

    def n(d):
        return d / inter_eye

    aus = {}

    # AU04: Brow lowerer
    brow_l = p("left_brow_inner")
    brow_r = p("right_brow_inner")
    lid_l = p("left_lid_upper")
    lid_r = p("right_lid_upper")
    gap_l = n(lid_l[1] - brow_l[1])
    gap_r = n(lid_r[1] - brow_r[1])
    avg_gap = (gap_l + gap_r) / 2.0
    aus["AU04"] = float(np.clip((0.55 - avg_gap) / 0.55 * 5.0, 0.0, 5.0))

    # AU07: Lid tightener
    upper_l, lower_l = p("left_lid_upper"), p("left_lid_lower")
    upper_r, lower_r = p("right_lid_upper"), p("right_lid_lower")
    open_l = n(abs(lower_l[1] - upper_l[1]))
    open_r = n(abs(lower_r[1] - upper_r[1]))
    avg_open = (open_l + open_r) / 2.0
    aus["AU07"] = float(np.clip((0.30 - avg_open) / 0.30 * 5.0, 0.0, 5.0))

    # AU12: Smile signal, used as a negative cue for anger
    corner_l = p("mouth_left")
    corner_r = p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2.0
    corner_avg_y = (corner_l[1] + corner_r[1]) / 2.0
    corner_rise = n(lip_mid_y - corner_avg_y)
    aus["AU12"] = float(np.clip(corner_rise * 8.0 + 2.5, 0.0, 5.0))

    # AU20: Lip stretcher
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    aus["AU20"] = float(np.clip((lip_width - 1.3) / 0.4 * 5.0, 0.0, 5.0))

    # AU23 / AU24: Lip tightener / lip pressor
    inner_gap = n(abs(p("lip_lower_in")[1] - p("lip_upper_in")[1]))
    aus["AU23"] = float(np.clip((0.10 - inner_gap) / 0.10 * 5.0, 0.0, 5.0))
    aus["AU24"] = float(np.clip((0.06 - inner_gap) / 0.06 * 5.0, 0.0, 5.0))

    return aus


def detect_angry_aus(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> dict | None:
    # Convert OpenCV frame to RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect face landmarks
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None

    # Compute anger-related AU values
    h, w = frame_bgr.shape[:2]
    return compute_angry_aus_from_landmarks(result.face_landmarks[0], w, h)


def compute_angry_score(au_row: dict, deepface_angry: float | None) -> float:
    # Normalize AU values
    au04 = float(np.clip(au_row.get("AU04", 0.0) / 5.0, 0.0, 1.0))
    au07 = float(np.clip(au_row.get("AU07", 0.0) / 5.0, 0.0, 1.0))
    au20 = float(np.clip(au_row.get("AU20", 0.0) / 5.0, 0.0, 1.0))
    au23 = float(np.clip(au_row.get("AU23", 0.0) / 5.0, 0.0, 1.0))
    au24 = float(np.clip(au_row.get("AU24", 0.0) / 5.0, 0.0, 1.0))
    au12 = float(np.clip(au_row.get("AU12", 0.0) / 5.0, 0.0, 1.0))

    # Weighted anger score from geometry
    au_score = (
        au04 * 0.30 +
        au07 * 0.20 +
        au20 * 0.15 +
        au23 * 0.20 +
        au24 * 0.20 -
        au12 * 0.15
    ) * 100.0

    au_score = float(np.clip(au_score, 0.0, 100.0))

    # If DeepFace is unavailable, use AU score only
    if deepface_angry is None:
        return au_score

    # Blend DeepFace anger with AU score
    return float(np.clip(0.6 * deepface_angry + 0.4 * au_score, 0.0, 100.0))


def angry_label(score: float):
    # Convert anger score to label and color
    if score < 35:
        return "Not Angry", (0, 200, 0)
    elif score < 70:
        return "Slightly Angry", (0, 165, 255)
    else:
        return "Angry", (0, 0, 255)


def draw_face_box(frame, result):
    # Draw face bounding box
    region = result.get("region", {})
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_angry_results(frame, deepface_result, angry_score, au_row):
    # Get label and bar color
    label, color = angry_label(angry_score)

    # Main output text
    cv2.putText(frame, f"Angry Face: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Anger Score: {angry_score:.0f}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Draw anger bar
    bar_width = int(angry_score * 2)
    cv2.rectangle(frame, (20, 95), (220, 115), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 95), (20 + bar_width, 115), color, -1)

    # Draw DeepFace details
    if deepface_result:
        emotions = deepface_result.get("emotion", {})
        angry_val = emotions.get("angry", 0.0)
        dominant = deepface_result.get("dominant_emotion", "")
        cv2.putText(frame, f"DeepFace angry: {angry_val:.1f}%", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Dominant emotion: {dominant}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Draw AU values
    if au_row:
        cv2.putText(frame, f"AU04 Brow Lowerer: {au_row.get('AU04', 0.0):.2f}/5", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"AU07 Lid Tightener: {au_row.get('AU07', 0.0):.2f}/5", (20, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"AU20 Lip Stretcher: {au_row.get('AU20', 0.0):.2f}/5", (20, 251),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"AU23 Lip Tightener: {au_row.get('AU23', 0.0):.2f}/5", (20, 274),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"AU24 Lip Pressor: {au_row.get('AU24', 0.0):.2f}/5", (20, 297),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    # Create landmarker
    landmarker = _create_face_landmarker()
    print("Press 'q' to quit.")

    # State variables
    frame_count = 0
    timestamp_ms = 0
    last_result = None
    last_au_row = None
    last_angry_score = None
    last_deepface_angry = None

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        timestamp_ms += 33

        # DeepFace emotion analysis
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
                last_deepface_angry = last_result.get("emotion", {}).get("angry", 0.0)
            except Exception:
                last_result = None
                last_deepface_angry = None

        # MediaPipe anger analysis
        if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
            raw_aus = detect_angry_aus(landmarker, frame, timestamp_ms)
            if raw_aus:
                if last_au_row is None:
                    last_au_row = raw_aus
                else:
                    # Smooth the AU values
                    for au in raw_aus:
                        prev = last_au_row.get(au, raw_aus[au])
                        last_au_row[au] = AU_SMOOTH_ALPHA * prev + (1 - AU_SMOOTH_ALPHA) * raw_aus[au]

                # Compute overall anger score
                last_angry_score = compute_angry_score(last_au_row, last_deepface_angry)

        # Draw face box
        if last_result:
            draw_face_box(frame, last_result)

        # Draw output or no-face message
        if last_angry_score is not None and last_au_row is not None:
            draw_angry_results(frame, last_result, last_angry_score, last_au_row)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show live window
        cv2.imshow("Angry Face Detector", frame)

        # Quit with q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
