import sys
import math
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QLinearGradient, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QVBoxLayout, QPushButton, QHBoxLayout
)

# ── tunables ─────────────────────────────────────────────────────────────────
ANALYZE_EVERY_N_FRAMES = 8
AU_ANALYZE_EVERY_N_FRAMES = 3   # MediaPipe is fast; can run frequently
AU_SMOOTH_ALPHA        = 0.6    # EMA smoothing for AU values
STRESS_THRESHOLD       = 40.0   # 0-100; above this counts as "stressed"
STRESS_HOLD_SECS       = 10     # seconds of sustained stress before glow
GLOW_DURATION_MS       = 2 * 60 * 1000   # 2 minutes
GLOW_LAYERS            = 32              # more layers = smoother gradient
GLOW_DEPTH_PX          = 90             # wider spread = softer edge
PULSE_INTERVAL_MS      = 40
# Breathing-rate pulse: ~8 s per cycle ≈ 7.5 breaths/min (guided breathing)
BREATH_PHASE_INC       = 2 * math.pi / (8000 / PULSE_INTERVAL_MS)  # ≈ 0.0314
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# AU stress model (same weights as deepface_test.py)
# ---------------------------------------------------------------------------
AU_STRESS_WEIGHTS = {
    "AU04": 0.35,   # Brow Lowerer
    "AU07": 0.20,   # Lid Tightener
    "AU20": 0.15,   # Lip Stretcher
    "AU23": 0.15,   # Lip Tightener
    "AU24": 0.10,   # Lip Pressor
    "AU12": -0.25,  # Lip Corner Puller (smile)
}
_POS_SUM = sum(w for w in AU_STRESS_WEIGHTS.values() if w > 0)
_NEG_SUM = abs(sum(w for w in AU_STRESS_WEIGHTS.values() if w < 0))

# MediaPipe landmark indices
_LM = {
    "left_eye_inner": 133, "left_lid_upper": 159, "left_lid_lower": 145,
    "right_eye_inner": 362, "right_lid_upper": 386, "right_lid_lower": 374,
    "left_brow_inner": 107, "right_brow_inner": 336,
    "mouth_left": 61, "mouth_right": 291,
    "lip_upper_in": 13, "lip_lower_in": 14,
}

# Path to the FaceLandmarker model (downloaded alongside this script)
import pathlib
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
    lm = landmarks[_LM[key]]
    return np.array([lm.x * w, lm.y * h])


def compute_aus_from_landmarks(landmarks, img_w, img_h):
    """Estimate AU intensities (0-5) from MediaPipe face mesh landmarks."""
    p = lambda key: _pt(landmarks, key, img_w, img_h)
    eye_l, eye_r = p("left_eye_inner"), p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}
    n = lambda d: d / inter_eye
    aus = {}

    # AU04: Brow Lowerer
    gap_l = n(p("left_lid_upper")[1] - p("left_brow_inner")[1])
    gap_r = n(p("right_lid_upper")[1] - p("right_brow_inner")[1])
    aus["AU04"] = float(np.clip((0.55 - (gap_l + gap_r) / 2) / 0.55 * 5, 0, 5))

    # AU07: Lid Tightener
    open_l = n(abs(p("left_lid_lower")[1] - p("left_lid_upper")[1]))
    open_r = n(abs(p("right_lid_lower")[1] - p("right_lid_upper")[1]))
    aus["AU07"] = float(np.clip((0.30 - (open_l + open_r) / 2) / 0.30 * 5, 0, 5))

    # AU12: Lip Corner Puller
    corner_l, corner_r = p("mouth_left"), p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2
    corner_rise = n(lip_mid_y - (corner_l[1] + corner_r[1]) / 2)
    aus["AU12"] = float(np.clip(corner_rise * 8 + 2.5, 0, 5))

    # AU20: Lip Stretcher
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    aus["AU20"] = float(np.clip((lip_width - 1.3) / 0.4 * 5, 0, 5))

    # AU23 & AU24: Lip Tightener / Pressor
    inner_gap = n(abs(p("lip_lower_in")[1] - p("lip_upper_in")[1]))
    aus["AU23"] = float(np.clip((0.10 - inner_gap) / 0.10 * 5, 0, 5))
    aus["AU24"] = float(np.clip((0.06 - inner_gap) / 0.06 * 5, 0, 5))
    return aus


def compute_stress_score(au_row):
    raw = sum(w * np.clip(float(au_row.get(au, 0)) / 5, 0, 1)
              for au, w in AU_STRESS_WEIGHTS.items())
    return float(np.clip((raw + _NEG_SUM) / (_POS_SUM + _NEG_SUM) * 100, 0, 100))


class EmotionThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    emotion_found = pyqtSignal(str, dict)
    stress_found  = pyqtSignal(float, dict)   # stress_score, smoothed AU row
    log_message   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_message.emit("ERROR: Could not open webcam")
            return

        landmarker = _create_face_landmarker()
        timestamp_ms = 0

        frame_count = 0
        smooth_aus = None

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            timestamp_ms += 33  # ~30 fps assumed for monotonic timestamp
            self.frame_ready.emit(frame.copy())

            # DeepFace emotion analysis
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                try:
                    results = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )
                    result = results[0] if isinstance(results, list) else results
                    dominant = result.get("dominant_emotion", "unknown")
                    scores   = result.get("emotion", {})
                    self.log_message.emit(f"Detected: {dominant}")
                    self.emotion_found.emit(dominant, scores)
                except Exception as e:
                    self.log_message.emit(f"Analysis error: {e}")

            # MediaPipe AU-based stress detection
            if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if mp_result.face_landmarks:
                    h, w = frame.shape[:2]
                    raw = compute_aus_from_landmarks(
                        mp_result.face_landmarks[0], w, h)
                    if raw:
                        if smooth_aus is None:
                            smooth_aus = raw
                        else:
                            for au in raw:
                                prev = smooth_aus.get(au, raw[au])
                                smooth_aus[au] = (AU_SMOOTH_ALPHA * prev
                                                  + (1 - AU_SMOOTH_ALPHA) * raw[au])
                        score = compute_stress_score(smooth_aus)
                        self.stress_found.emit(score, dict(smooth_aus))

        landmarker.close()
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class GlowOverlay(QWidget):
    """Full-screen transparent overlay with animated blue glow border."""

    def __init__(self):
        super().__init__(None)   # no parent — standalone top-level window

        # On macOS, Qt.Tool can hide behind other windows; Qt.Window is safer
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Window
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # Cover the full screen (including macOS menu bar)
        screen_geo = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geo)

        self._alpha  = 0.0
        self._phase  = 0.0
        self._active = False
        self._color  = QColor(30, 144, 255)   # default: blue

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_INTERVAL_MS)
        self._pulse_timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(GLOW_DURATION_MS)
        self._hide_timer.timeout.connect(self.hide_glow)

    # ── public API ────────────────────────────────────────────────────────────

    def show_glow(self, color: QColor = None):
        if color is not None:
            self._color = color
        if self._active:
            # Already showing — restart the 2-min countdown
            self._hide_timer.start()
            return
        self._active = True
        self._phase  = 0.0
        self.show()
        self.raise_()           # ensure it sits on top
        self._pulse_timer.start()
        self._hide_timer.start()

    def hide_glow(self):
        self._active = False
        self._pulse_timer.stop()
        self._hide_timer.stop()
        self.hide()

    @property
    def active(self):
        return self._active

    # ── internals ────────────────────────────────────────────────────────────

    def _tick(self):
        # ~8-second breathing cycle; smooth sine eases in/out like an inhale/exhale
        self._phase = (self._phase + BREATH_PHASE_INC) % (2 * math.pi)
        self._alpha = 0.15 + 0.50 * (0.5 + 0.5 * math.sin(self._phase))
        self.update()

    def paintEvent(self, _):
        if not self._active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        r = self.rect()
        d = GLOW_DEPTH_PX
        a = int(self._alpha * 255)
        glow  = QColor(self._color.red(), self._color.green(), self._color.blue(), a)
        clear = QColor(self._color.red(), self._color.green(), self._color.blue(), 0)

        def make_grad(x1, y1, x2, y2):
            g = QLinearGradient(x1, y1, x2, y2)
            g.setColorAt(0.0, glow)
            g.setColorAt(1.0, clear)
            return QBrush(g)

        # Top edge
        painter.setBrush(make_grad(0, r.top(), 0, r.top() + d))
        painter.drawRect(r.left(), r.top(), r.width(), d)
        # Bottom edge
        painter.setBrush(make_grad(0, r.bottom(), 0, r.bottom() - d))
        painter.drawRect(r.left(), r.bottom() - d, r.width(), d)
        # Left edge
        painter.setBrush(make_grad(r.left(), 0, r.left() + d, 0))
        painter.drawRect(r.left(), r.top(), d, r.height())
        # Right edge
        painter.setBrush(make_grad(r.right(), 0, r.right() - d, 0))
        painter.drawRect(r.right() - d, r.top(), d, r.height())

        painter.end()


class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Starting camera…")
        self.setStyleSheet("color: white; font-size: 16px;")
        self.setMinimumSize(640, 480)

    def update_frame(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pix)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodLens")
        self.setStyleSheet("background-color: #111;")

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._camera = CameraWidget()
        layout.addWidget(self._camera)

        # Stress bar (lives between camera feed and bottom debug bar)
        self._stress_label = QLabel("Stress: —")
        self._stress_label.setStyleSheet(
            "color: #ccc; font-size: 13px; padding: 4px 10px;"
            "background-color: #1a1a1a;"
        )
        self._stress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._stress_label)

        # Bottom bar: debug label + test button
        bar = QWidget()
        bar.setStyleSheet("background-color: #1a1a1a;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 4, 10, 4)

        self._debug_label = QLabel("Waiting for analysis…")
        self._debug_label.setStyleSheet("color: #aaa; font-size: 12px;")
        bar_layout.addWidget(self._debug_label, stretch=1)

        test_btn = QPushButton("Test Glow")
        test_btn.setStyleSheet(
            "background:#1e90ff; color:white; border-radius:4px;"
            "padding: 4px 12px; font-size:12px;"
        )
        test_btn.clicked.connect(self._test_glow)
        bar_layout.addWidget(test_btn)

        hide_btn = QPushButton("Hide Glow")
        hide_btn.setStyleSheet(
            "background:#444; color:white; border-radius:4px;"
            "padding: 4px 12px; font-size:12px;"
        )
        hide_btn.clicked.connect(self._glow.hide_glow if hasattr(self, '_glow') else lambda: None)
        bar_layout.addWidget(hide_btn)

        layout.addWidget(bar)
        self.resize(800, 620)

        # Glow overlay
        self._glow = GlowOverlay()
        # Wire hide button now that _glow exists
        hide_btn.clicked.connect(self._glow.hide_glow)

        # Stress tracking state
        self._stress_start = None   # time.monotonic() when sustained stress began

        # Emotion thread
        self._thread = EmotionThread()
        self._thread.frame_ready.connect(self._camera.update_frame)
        self._thread.emotion_found.connect(self._on_emotion)
        self._thread.stress_found.connect(self._on_stress)
        self._thread.log_message.connect(self._on_log)
        self._thread.start()

    def _test_glow(self):
        self._debug_label.setText("Manual test triggered")
        self._glow.show_glow()

    def _on_log(self, msg: str):
        print(msg)   # also visible in terminal
        self._debug_label.setText(msg)

    def _on_emotion(self, dominant: str, scores: dict):
        top    = sorted(scores.items(), key=lambda x: -x[1])
        status = "  |  ".join(f"{e}: {s:.1f}%" for e, s in top[:4])
        self._debug_label.setText(f"Mood: {dominant.upper()}  —  {status}")

    def _on_stress(self, score: float, au_row: dict):
        now = time.monotonic()

        # -- Always update the live stress readout ----------------------------
        if score < 33:
            level, colour = "Low", "#4caf50"
        elif score < 66:
            level, colour = "Moderate", "#ffa726"
        else:
            level, colour = "High", "#ef5350"

        # Show active AUs (intensity > 0.5) alongside the percentage
        active = [f"{au} {val:.1f}"
                  for au, val in sorted(au_row.items())
                  if val > 0.5]
        au_text = "  |  ".join(active) if active else "none"

        self._stress_label.setText(
            f"Stress: {score:.0f}% ({level})    AUs: {au_text}")
        self._stress_label.setStyleSheet(
            f"color: {colour}; font-size: 13px; font-weight: bold;"
            f"padding: 4px 10px; background-color: #1a1a1a;")

        # -- Glow trigger logic (unchanged) -----------------------------------
        if score >= STRESS_THRESHOLD:
            if self._stress_start is None:
                self._stress_start = now
            elif (now - self._stress_start) >= STRESS_HOLD_SECS:
                if not self._glow.active:
                    self._glow.show_glow(QColor(30, 144, 255))
        else:
            self._stress_start = None
            if self._glow.active:
                self._glow.hide_glow()

    def closeEvent(self, event):
        self._glow.hide_glow()
        self._thread.stop()
        super().closeEvent(event)


def main():
    # Required on macOS for high-DPI and layer compositing
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
