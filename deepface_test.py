import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

# How often to run each analysis pass (every N frames)
ANALYZE_EVERY_N_FRAMES = 5    # DeepFace emotion
AU_ANALYZE_EVERY_N_FRAMES = 3  # MediaPipe AU (very fast, can run often)

# Smoothing factor for AU values (exponential moving average, 0=no smooth, 1=frozen)
AU_SMOOTH_ALPHA = 0.6

# ---------------------------------------------------------------------------
# Facial Action Unit stress model
# AUs are computed geometrically from MediaPipe face mesh landmarks (0–5 scale).
# Positive weight → raises stress score | Negative → lowers stress score
# ---------------------------------------------------------------------------
AU_STRESS_WEIGHTS = {
    "AU04": 0.35,   # Brow Lowerer      — concentration / distress
    "AU07": 0.20,   # Lid Tightener     — muscular tension
    "AU20": 0.15,   # Lip Stretcher     — fear / stress
    "AU23": 0.15,   # Lip Tightener     — suppressed anger / stress
    "AU24": 0.10,   # Lip Pressor       — effort / stress
    "AU12": -0.25,  # Lip Corner Puller — smile / relaxation
}

_POS_SUM = sum(w for w in AU_STRESS_WEIGHTS.values() if w > 0)   # 0.95
_NEG_SUM = abs(sum(w for w in AU_STRESS_WEIGHTS.values() if w < 0))  # 0.25

# ---------------------------------------------------------------------------
# MediaPipe landmark indices used for AU computation
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------
_LM = {
    # Eyes
    "left_eye_inner":  133, "left_eye_outer":   33,
    "left_lid_upper":  159, "left_lid_lower":  145,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "right_lid_upper": 386, "right_lid_lower": 374,
    # Brows
    "left_brow_inner":  107, "right_brow_inner": 336,
    "left_brow_mid":    105, "right_brow_mid":   334,
    # Mouth
    "mouth_left":   61,  "mouth_right":   291,
    "lip_upper_in": 13,  "lip_lower_in":  14,
    "lip_upper_out": 0,  "lip_lower_out": 17,
}

# Path to the FaceLandmarker model (downloaded alongside this script)
_MODEL_PATH = str(pathlib.Path(__file__).parent / "face_landmarker.task")


def _create_face_landmarker():
    """Create a MediaPipe FaceLandmarker (Tasks API) for video mode."""
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
    """Return pixel (x, y) for a named landmark."""
    lm = landmarks[_LM[key]]
    return np.array([lm.x * w, lm.y * h])


def compute_aus_from_landmarks(landmarks, img_w: int, img_h: int) -> dict:
    """
    Estimate AU intensities (0–5 FACS scale) from MediaPipe face mesh landmarks.
    All distances are normalised by the inter-inner-eye distance so the values
    are pose/distance-invariant.
    """
    p = lambda key: _pt(landmarks, key, img_w, img_h)

    # Normaliser: distance between inner eye corners
    eye_l = p("left_eye_inner")
    eye_r = p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}

    def n(d):
        return d / inter_eye

    aus = {}

    # -- AU04: Brow Lowerer --------------------------------------------------
    # Vertical gap between inner brow and upper eyelid.
    # Decreases when brow descends toward the eye (furrowing/stress).
    brow_l = p("left_brow_inner")
    brow_r = p("right_brow_inner")
    lid_l  = p("left_lid_upper")
    lid_r  = p("right_lid_upper")
    gap_l = n(lid_l[1] - brow_l[1])  # positive → brow above lid
    gap_r = n(lid_r[1] - brow_r[1])
    avg_gap = (gap_l + gap_r) / 2.0
    # Empirical neutral ≈ 0.50; lower = more furrowed
    aus["AU04"] = float(np.clip((0.55 - avg_gap) / 0.55 * 5.0, 0.0, 5.0))

    # -- AU07: Lid Tightener -------------------------------------------------
    # Vertical eye aperture (upper lid to lower lid).
    # Decreases when lids tighten (tension/squinting).
    upper_l, lower_l = p("left_lid_upper"),  p("left_lid_lower")
    upper_r, lower_r = p("right_lid_upper"), p("right_lid_lower")
    open_l = n(abs(lower_l[1] - upper_l[1]))
    open_r = n(abs(lower_r[1] - upper_r[1]))
    avg_open = (open_l + open_r) / 2.0
    # Empirical neutral ≈ 0.28; smaller = more tightened
    aus["AU07"] = float(np.clip((0.30 - avg_open) / 0.30 * 5.0, 0.0, 5.0))

    # -- AU12: Lip Corner Puller (smile) -------------------------------------
    # Positive when lip corners are above (lower y) the mid-lip line.
    corner_l = p("mouth_left")
    corner_r = p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2.0
    corner_avg_y = (corner_l[1] + corner_r[1]) / 2.0
    # Negative n() value = corners are above lip centre → smile
    corner_rise = n(lip_mid_y - corner_avg_y)
    # Scale so neutral (flat) ≈ 2.5, full smile ≈ 5, full frown ≈ 0
    aus["AU12"] = float(np.clip(corner_rise * 8.0 + 2.5, 0.0, 5.0))

    # -- AU20: Lip Stretcher -------------------------------------------------
    # Horizontal lip width relative to inter-eye distance.
    # Increases in fear / stress (lips pulled back and wide).
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    # Empirical neutral ≈ 1.3; wider = more stretched
    aus["AU20"] = float(np.clip((lip_width - 1.3) / 0.4 * 5.0, 0.0, 5.0))

    # -- AU23 & AU24: Lip Tightener / Lip Pressor ----------------------------
    # Vertical gap between inner upper and lower lip surfaces.
    # Approaches zero when lips are pressed together (tension/effort).
    inner_gap = n(abs(p("lip_lower_in")[1] - p("lip_upper_in")[1]))
    # AU23 activates earlier (at a larger remaining gap)
    aus["AU23"] = float(np.clip((0.10 - inner_gap) / 0.10 * 5.0, 0.0, 5.0))
    aus["AU24"] = float(np.clip((0.06 - inner_gap) / 0.06 * 5.0, 0.0, 5.0))

    return aus


def detect_aus(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> dict | None:
    """
    Run MediaPipe FaceLandmarker on a BGR frame and return AU intensities,
    or None if no face is detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    return compute_aus_from_landmarks(result.face_landmarks[0], w, h)


def compute_stress_score(au_row: dict) -> float:
    """Return a 0–100 stress score from a dict of AU name → intensity (0–5 scale)."""
    raw = 0.0
    for au, weight in AU_STRESS_WEIGHTS.items():
        val = float(au_row.get(au, 0.0))
        raw += weight * np.clip(val / 5.0, 0.0, 1.0)
    # raw ∈ [−_NEG_SUM, +_POS_SUM] → map to [0, 100]
    return float(np.clip((raw + _NEG_SUM) / (_POS_SUM + _NEG_SUM) * 100.0, 0.0, 100.0))


def _stress_label(score: float):
    """Return (text, BGR-colour) for a given stress score."""
    if score < 33:
        return "Low", (0, 200, 0)
    elif score < 66:
        return "Moderate", (0, 165, 255)
    else:
        return "High", (0, 0, 255)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_results(frame, result):
    """Draw emotion label and confidence bars on the frame."""
    dominant = result.get("dominant_emotion", "")
    emotions = result.get("emotion", {})

    cv2.putText(frame, f"Mood: {dominant}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    y = 80
    for emotion, score in sorted(emotions.items(), key=lambda x: -x[1]):
        bar_width = int(score * 2)          # 0-100 → 0-200 px
        cv2.rectangle(frame, (20, y), (20 + bar_width, y + 14), (0, 200, 255), -1)
        cv2.putText(frame, f"{emotion}: {score:.1f}%",
                    (230, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22


def draw_stress(frame, stress_score: float, au_row: dict):
    """Draw AU-based stress indicator on the right side of the frame."""
    label, colour = _stress_label(stress_score)
    bar_x = frame.shape[1] - 230

    # Stress label + percentage
    cv2.putText(frame, f"Stress: {label} ({stress_score:.0f}%)",
                (bar_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    # Stress fill bar
    bar_fill = int(stress_score * 1.5)      # 0-100 → 0-150 px
    cv2.rectangle(frame, (bar_x, 55), (bar_x + 150, 72), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, 55), (bar_x + bar_fill, 72), colour, -1)

    # Active AUs
    cv2.putText(frame, "Active AUs:", (bar_x, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y = 112
    for au, weight in AU_STRESS_WEIGHTS.items():
        val = float(au_row.get(au, 0.0))
        if val > 0.5:
            dot_colour = (0, 0, 220) if weight > 0 else (0, 180, 0)
            cv2.circle(frame, (bar_x + 6, y - 3), 4, dot_colour, -1)
            cv2.putText(frame, f"{au}  {val:.1f}/5",
                        (bar_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (220, 220, 220), 1)
            y += 16


def draw_face_box(frame, result):
    """Draw bounding box around the detected face."""
    region = result.get("region", {})
    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
    if w > 0 and h > 0:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    landmarker = _create_face_landmarker()
    print("Press 'q' to quit.")

    frame_count   = 0
    timestamp_ms  = 0
    last_result   = None
    last_au_row   = None     # smoothed AU values
    last_stress   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        timestamp_ms += 33  # ~30 fps monotonic timestamp

        # -- DeepFace emotion (every N frames) --------------------------------
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                last_result = results[0] if isinstance(results, list) else results
            except Exception as e:
                last_result = None
                cv2.putText(frame, f"Error: {e}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # -- AU detection via MediaPipe (every N frames) ----------------------
        if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
            raw_aus = detect_aus(landmarker, frame, timestamp_ms)
            if raw_aus:
                if last_au_row is None:
                    last_au_row = raw_aus
                else:
                    # Exponential moving average to reduce jitter
                    for au in raw_aus:
                        prev = last_au_row.get(au, raw_aus[au])
                        last_au_row[au] = AU_SMOOTH_ALPHA * prev + (1 - AU_SMOOTH_ALPHA) * raw_aus[au]
                last_stress = compute_stress_score(last_au_row)

        # -- Draw overlays ----------------------------------------------------
        if last_result:
            draw_face_box(frame, last_result)
            draw_results(frame, last_result)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if last_stress is not None and last_au_row is not None:
            draw_stress(frame, last_stress, last_au_row)

        cv2.imshow("MoodLens — Real-time Emotion & Stress Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
