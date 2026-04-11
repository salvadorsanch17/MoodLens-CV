import sys
import math
import time
import random
import pathlib
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
    QVBoxLayout, QPushButton, QHBoxLayout, QProgressBar
)

# ── tunables ─────────────────────────────────────────────────────────────────
ANALYZE_EVERY_N_FRAMES = 8
AU_ANALYZE_EVERY_N_FRAMES = 3
AU_SMOOTH_ALPHA        = 0.6
STRESS_THRESHOLD       = 40.0
STRESS_HOLD_SECS       = 10
GAZE_AWAY_SECS         = 5          # seconds looking away → boredom
BOREDOM_GLOW_MS        = 5000       # red glow duration
LOCKIN_GOAL_SECS       = 10 * 60    # 10 minutes of on-screen gaze
GLOW_DURATION_MS       = 2 * 60 * 1000
GLOW_DEPTH_PX          = 90
PULSE_INTERVAL_MS      = 40
BREATH_PHASE_INC       = 2 * math.pi / (8000 / PULSE_INTERVAL_MS)
GAZE_YAW_THRESHOLD     = 0.15       # head-turn tolerance (0 = dead-centre)
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# AU stress model
# ---------------------------------------------------------------------------
AU_STRESS_WEIGHTS = {
    "AU04": 0.35, "AU07": 0.20, "AU20": 0.15,
    "AU23": 0.15, "AU24": 0.10, "AU12": -0.25,
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

_MODEL_PATH = str(pathlib.Path(__file__).parent / "face_landmarker.task")


def _create_face_landmarker():
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
    p = lambda key: _pt(landmarks, key, img_w, img_h)
    eye_l, eye_r = p("left_eye_inner"), p("right_eye_inner")
    inter_eye = float(np.linalg.norm(eye_r - eye_l))
    if inter_eye < 1.0:
        return {}
    n = lambda d: d / inter_eye
    aus = {}
    gap_l = n(p("left_lid_upper")[1] - p("left_brow_inner")[1])
    gap_r = n(p("right_lid_upper")[1] - p("right_brow_inner")[1])
    aus["AU04"] = float(np.clip((0.55 - (gap_l + gap_r) / 2) / 0.55 * 5, 0, 5))
    open_l = n(abs(p("left_lid_lower")[1] - p("left_lid_upper")[1]))
    open_r = n(abs(p("right_lid_lower")[1] - p("right_lid_upper")[1]))
    aus["AU07"] = float(np.clip((0.30 - (open_l + open_r) / 2) / 0.30 * 5, 0, 5))
    corner_l, corner_r = p("mouth_left"), p("mouth_right")
    lip_mid_y = (p("lip_upper_in")[1] + p("lip_lower_in")[1]) / 2
    corner_rise = n(lip_mid_y - (corner_l[1] + corner_r[1]) / 2)
    aus["AU12"] = float(np.clip(corner_rise * 8 + 2.5, 0, 5))
    lip_width = n(np.linalg.norm(corner_r - corner_l))
    aus["AU20"] = float(np.clip((lip_width - 1.3) / 0.4 * 5, 0, 5))
    inner_gap = n(abs(p("lip_lower_in")[1] - p("lip_upper_in")[1]))
    aus["AU23"] = float(np.clip((0.10 - inner_gap) / 0.10 * 5, 0, 5))
    aus["AU24"] = float(np.clip((0.06 - inner_gap) / 0.06 * 5, 0, 5))
    return aus


def compute_stress_score(au_row):
    raw = sum(w * np.clip(float(au_row.get(au, 0)) / 5, 0, 1)
              for au, w in AU_STRESS_WEIGHTS.items())
    return float(np.clip((raw + _NEG_SUM) / (_POS_SUM + _NEG_SUM) * 100, 0, 100))


def is_looking_at_screen(landmarks) -> bool:
    """Estimate whether user faces the camera from head yaw."""
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_dist = abs(nose.x - left_eye.x)
    right_dist = abs(nose.x - right_eye.x)
    total = left_dist + right_dist
    if total < 0.01:
        return False
    yaw_ratio = left_dist / total          # ≈ 0.5 when facing camera
    return abs(yaw_ratio - 0.5) <= GAZE_YAW_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
#  Threads
# ═══════════════════════════════════════════════════════════════════════════

class EmotionThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    emotion_found = pyqtSignal(str, dict)
    stress_found  = pyqtSignal(float, dict)
    gaze_status   = pyqtSignal(bool)          # True = looking at screen
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
            timestamp_ms += 33
            self.frame_ready.emit(frame.copy())

            # DeepFace emotion analysis
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                try:
                    results = DeepFace.analyze(
                        frame, actions=["emotion"],
                        enforce_detection=False, silent=True,
                    )
                    result = results[0] if isinstance(results, list) else results
                    dominant = result.get("dominant_emotion", "unknown")
                    scores = result.get("emotion", {})
                    self.log_message.emit(f"Detected: {dominant}")
                    self.emotion_found.emit(dominant, scores)
                except Exception as e:
                    self.log_message.emit(f"Analysis error: {e}")

            # MediaPipe: AUs + gaze
            if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if mp_result.face_landmarks:
                    lms = mp_result.face_landmarks[0]
                    # Gaze
                    self.gaze_status.emit(is_looking_at_screen(lms))
                    # AUs
                    h, w = frame.shape[:2]
                    raw = compute_aus_from_landmarks(lms, w, h)
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
                else:
                    # No face → not looking at screen
                    self.gaze_status.emit(False)

        landmarker.close()
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════
#  Overlays
# ═══════════════════════════════════════════════════════════════════════════

class GlowOverlay(QWidget):
    """Full-screen glow border overlay."""
    glow_hidden = pyqtSignal()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Window | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(QApplication.primaryScreen().geometry())

        self._alpha  = 0.0
        self._phase  = 0.0
        self._active = False
        self._color  = QColor(30, 144, 255)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_INTERVAL_MS)
        self._pulse_timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_glow)

    def show_glow(self, color: QColor = None, duration_ms: int = None):
        if color is not None:
            self._color = color
        dur = duration_ms if duration_ms is not None else GLOW_DURATION_MS
        self._hide_timer.setInterval(dur)
        if self._active:
            self._hide_timer.start()
            return
        self._active = True
        self._phase  = 0.0
        self.show()
        self.raise_()
        self._pulse_timer.start()
        self._hide_timer.start()

    def hide_glow(self):
        was = self._active
        self._active = False
        self._pulse_timer.stop()
        self._hide_timer.stop()
        self.hide()
        if was:
            self.glow_hidden.emit()

    @property
    def active(self):
        return self._active

    def _tick(self):
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

        def grad(x1, y1, x2, y2):
            g = QLinearGradient(x1, y1, x2, y2)
            g.setColorAt(0.0, glow)
            g.setColorAt(1.0, clear)
            return QBrush(g)

        painter.setBrush(grad(0, r.top(), 0, r.top() + d))
        painter.drawRect(r.left(), r.top(), r.width(), d)
        painter.setBrush(grad(0, r.bottom(), 0, r.bottom() - d))
        painter.drawRect(r.left(), r.bottom() - d, r.width(), d)
        painter.setBrush(grad(r.left(), 0, r.left() + d, 0))
        painter.drawRect(r.left(), r.top(), d, r.height())
        painter.setBrush(grad(r.right(), 0, r.right() - d, 0))
        painter.drawRect(r.right() - d, r.top(), d, r.height())
        painter.end()


class ConfettiOverlay(QWidget):
    """Full-screen confetti burst."""

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Window | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setGeometry(QApplication.primaryScreen().geometry())

        self._particles = []
        self._active = False

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_confetti)

    def show_confetti(self, duration_ms=6000):
        w, h = self.width(), self.height()
        colors = [
            (255, 87, 87), (255, 189, 46), (46, 213, 115),
            (30, 144, 255), (153, 102, 255), (255, 71, 179),
            (255, 215, 0), (0, 206, 209),
        ]
        self._particles = []
        for _ in range(250):
            self._particles.append({
                "x": random.uniform(0, w),
                "y": random.uniform(-h * 0.6, 0),
                "dx": random.uniform(-3, 3),
                "dy": random.uniform(1, 6),
                "color": random.choice(colors),
                "w": random.randint(6, 12),
                "h": random.randint(3, 8),
                "rot": random.uniform(0, 360),
                "drot": random.uniform(-6, 6),
            })
        self._active = True
        self.show()
        self.raise_()
        self._timer.start()
        self._hide_timer.start(duration_ms)

    def hide_confetti(self):
        self._active = False
        self._timer.stop()
        self._hide_timer.stop()
        self.hide()

    def _tick(self):
        for p in self._particles:
            p["x"] += p["dx"]
            p["y"] += p["dy"]
            p["dy"] += 0.12                     # gravity
            p["dx"] += random.uniform(-0.08, 0.08)  # wobble
            p["rot"] += p["drot"]
        self.update()

    def paintEvent(self, _):
        if not self._active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        for p in self._particles:
            c = p["color"]
            painter.setBrush(QColor(c[0], c[1], c[2], 210))
            painter.save()
            painter.translate(p["x"], p["y"])
            painter.rotate(p["rot"])
            painter.drawRect(-p["w"] // 2, -p["h"] // 2, p["w"], p["h"])
            painter.restore()
        painter.end()


# ═══════════════════════════════════════════════════════════════════════════
#  Lock-In focus timer widget (top-right corner)
# ═══════════════════════════════════════════════════════════════════════════

class LockInWidget(QWidget):
    completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(210, 86)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "LockInWidget { background: rgba(15,15,15,210);"
            "border-radius: 10px; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        self._title = QLabel("Lock In! Focus for 10 min")
        self._title.setStyleSheet(
            "color: #1e90ff; font-size: 13px; font-weight: bold;"
            "background: transparent;")
        self._title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._title)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._bar.setStyleSheet(
            "QProgressBar { background: #333; border-radius: 5px; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #1e90ff, stop:1 #00e5ff);"
            "border-radius: 5px; }")
        lay.addWidget(self._bar)

        self._time_label = QLabel("0:00 / 10:00")
        self._time_label.setStyleSheet(
            "color: #999; font-size: 11px; background: transparent;")
        self._time_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._time_label)

        self._focus_secs = 0.0
        self._completed = False

    def add_focus_time(self, secs: float):
        if self._completed:
            return
        self._focus_secs = min(self._focus_secs + secs, LOCKIN_GOAL_SECS)
        mins = int(self._focus_secs) // 60
        s = int(self._focus_secs) % 60
        self._time_label.setText(f"{mins}:{s:02d} / 10:00")
        self._bar.setValue(int(self._focus_secs / LOCKIN_GOAL_SECS * 1000))
        if self._focus_secs >= LOCKIN_GOAL_SECS:
            self._completed = True
            self._title.setText("You did it!")
            self._title.setStyleSheet(
                "color: #4caf50; font-size: 13px; font-weight: bold;"
                "background: transparent;")
            self._bar.setStyleSheet(
                "QProgressBar { background: #333; border-radius: 5px; }"
                "QProgressBar::chunk { background: #4caf50; border-radius: 5px; }")
            self.completed.emit()

    def set_nudge(self, active: bool):
        """Visually nudge when boredom detected."""
        if self._completed:
            return
        if active:
            self._title.setText("Hey — lock back in!")
            self._title.setStyleSheet(
                "color: #ef5350; font-size: 13px; font-weight: bold;"
                "background: transparent;")
        else:
            self._title.setText("Lock In! Focus for 10 min")
            self._title.setStyleSheet(
                "color: #1e90ff; font-size: 13px; font-weight: bold;"
                "background: transparent;")


# ═══════════════════════════════════════════════════════════════════════════
#  Camera widget
# ═══════════════════════════════════════════════════════════════════════════

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
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)


# ═══════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodLens")
        self.setStyleSheet("background-color: #111;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._camera = CameraWidget()
        layout.addWidget(self._camera)

        # Stress readout
        self._stress_label = QLabel("Stress: —")
        self._stress_label.setStyleSheet(
            "color: #ccc; font-size: 13px; padding: 4px 10px;"
            "background-color: #1a1a1a;")
        self._stress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._stress_label)

        # Bottom bar
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
            "padding: 4px 12px; font-size:12px;")
        test_btn.clicked.connect(self._test_glow)
        bar_layout.addWidget(test_btn)

        hide_btn = QPushButton("Hide Glow")
        hide_btn.setStyleSheet(
            "background:#444; color:white; border-radius:4px;"
            "padding: 4px 12px; font-size:12px;")
        bar_layout.addWidget(hide_btn)

        layout.addWidget(bar)
        self.resize(800, 640)

        # ── Overlays ─────────────────────────────────────────────────────
        self._glow = GlowOverlay()
        self._glow.glow_hidden.connect(self._on_glow_hidden)
        hide_btn.clicked.connect(self._glow.hide_glow)

        self._confetti = ConfettiOverlay()

        # Lock-in widget (child of self so it floats over the camera feed)
        self._lock_in = LockInWidget(self)
        self._lock_in.completed.connect(self._on_lockin_complete)
        self._lock_in.raise_()

        # ── State ────────────────────────────────────────────────────────
        self._stress_start    = None
        self._glow_source     = None   # 'stress' | 'boredom' | None
        self._gaze_away_start = None
        self._boredom_fired   = False  # one red glow per look-away episode
        self._was_looking     = False
        self._gaze_last_time  = None

        # ── Thread ───────────────────────────────────────────────────────
        self._thread = EmotionThread()
        self._thread.frame_ready.connect(self._camera.update_frame)
        self._thread.emotion_found.connect(self._on_emotion)
        self._thread.stress_found.connect(self._on_stress)
        self._thread.gaze_status.connect(self._on_gaze)
        self._thread.log_message.connect(self._on_log)
        self._thread.start()

    # ── layout helpers ────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep lock-in widget pinned to top-right
        self._lock_in.move(
            self.width() - self._lock_in.width() - 12, 12)
        self._lock_in.raise_()

    # ── callbacks ────────────────────────────────────────────────────────

    def _test_glow(self):
        self._debug_label.setText("Manual test triggered")
        self._glow.show_glow()

    def _on_log(self, msg: str):
        print(msg)
        self._debug_label.setText(msg)

    def _on_emotion(self, dominant: str, scores: dict):
        top = sorted(scores.items(), key=lambda x: -x[1])
        status = "  |  ".join(f"{e}: {s:.1f}%" for e, s in top[:4])
        self._debug_label.setText(f"Mood: {dominant.upper()}  —  {status}")

    # ── stress ───────────────────────────────────────────────────────────

    def _on_stress(self, score: float, au_row: dict):
        now = time.monotonic()

        if score < 33:
            level, colour = "Low", "#4caf50"
        elif score < 66:
            level, colour = "Moderate", "#ffa726"
        else:
            level, colour = "High", "#ef5350"

        active = [f"{au} {val:.1f}"
                  for au, val in sorted(au_row.items()) if val > 0.5]
        au_text = "  |  ".join(active) if active else "none"
        self._stress_label.setText(
            f"Stress: {score:.0f}% ({level})    AUs: {au_text}")
        self._stress_label.setStyleSheet(
            f"color: {colour}; font-size: 13px; font-weight: bold;"
            f"padding: 4px 10px; background-color: #1a1a1a;")

        # Blue glow — only if boredom glow is not active
        if score >= STRESS_THRESHOLD:
            if self._stress_start is None:
                self._stress_start = now
            elif (now - self._stress_start) >= STRESS_HOLD_SECS:
                if self._glow_source != "boredom":
                    if not self._glow.active:
                        self._glow.show_glow(QColor(30, 144, 255))
                        self._glow_source = "stress"
        else:
            self._stress_start = None
            if self._glow.active and self._glow_source == "stress":
                self._glow.hide_glow()

    # ── gaze / boredom ───────────────────────────────────────────────────

    def _on_gaze(self, looking: bool):
        now = time.monotonic()

        if looking:
            # Accumulate focus time (only between consecutive on-screen frames)
            if self._was_looking and self._gaze_last_time is not None:
                dt = now - self._gaze_last_time
                if dt < 1.0:                     # skip implausible gaps
                    self._lock_in.add_focus_time(dt)

            # Reset boredom state
            if self._gaze_away_start is not None:
                self._gaze_away_start = None
                self._boredom_fired = False
                self._lock_in.set_nudge(False)
        else:
            # Start / continue the away timer
            if self._gaze_away_start is None:
                self._gaze_away_start = now
                self._boredom_fired = False

            elapsed = now - self._gaze_away_start
            if elapsed >= GAZE_AWAY_SECS and not self._boredom_fired:
                # Fire red boredom glow (5 s)
                self._glow.show_glow(
                    QColor(220, 50, 50), duration_ms=BOREDOM_GLOW_MS)
                self._glow_source = "boredom"
                self._boredom_fired = True
                self._lock_in.set_nudge(True)

        self._gaze_last_time = now
        self._was_looking = looking

    def _on_glow_hidden(self):
        self._glow_source = None

    # ── confetti ─────────────────────────────────────────────────────────

    def _on_lockin_complete(self):
        self._confetti.show_confetti(duration_ms=8000)

    # ── cleanup ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._glow.hide_glow()
        self._confetti.hide_confetti()
        self._thread.stop()
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
