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
    QVBoxLayout, QPushButton, QHBoxLayout, QProgressBar,
    QTabWidget,
)

from dashboard import DashboardWidget, _C

# ── tunables ─────────────────────────────────────────────────────────────────
ANALYZE_EVERY_N_FRAMES = 8
AU_ANALYZE_EVERY_N_FRAMES = 3
AU_SMOOTH_ALPHA        = 0.6
STRESS_THRESHOLD       = 50.0
STRESS_HOLD_SECS       = 10
GAZE_AWAY_SECS         = 5
BOREDOM_GLOW_MS        = 5000
LOCKIN_GOAL_SECS       = 10 * 60
GLOW_DURATION_MS       = 2 * 60 * 1000
GLOW_DEPTH_PX          = 90
PULSE_INTERVAL_MS      = 40
BREATH_PHASE_INC       = 2 * math.pi / (8000 / PULSE_INTERVAL_MS)
GAZE_YAW_THRESHOLD     = 0.15
# ─────────────────────────────────────────────────────────────────────────────

# ── AU stress model ──────────────────────────────────────────────────────────
AU_STRESS_WEIGHTS = {
    "AU04": 0.35, "AU07": 0.20, "AU20": 0.15,
    "AU23": 0.15, "AU24": 0.10, "AU12": -0.25,
}
_POS_SUM = sum(w for w in AU_STRESS_WEIGHTS.values() if w > 0)
_NEG_SUM = abs(sum(w for w in AU_STRESS_WEIGHTS.values() if w < 0))

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
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_dist = abs(nose.x - left_eye.x)
    right_dist = abs(nose.x - right_eye.x)
    total = left_dist + right_dist
    if total < 0.01:
        return False
    yaw_ratio = left_dist / total
    return abs(yaw_ratio - 0.5) <= GAZE_YAW_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
#  Threads
# ═══════════════════════════════════════════════════════════════════════════

class EmotionThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    emotion_found = pyqtSignal(str, dict)
    stress_found  = pyqtSignal(float, dict)
    gaze_status   = pyqtSignal(bool)
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

            if frame_count % AU_ANALYZE_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if mp_result.face_landmarks:
                    lms = mp_result.face_landmarks[0]
                    self.gaze_status.emit(is_looking_at_screen(lms))
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
    glow_hidden = pyqtSignal()

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Window | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self._alpha = 0.0
        self._phase = 0.0
        self._active = False
        self._color = QColor(30, 144, 255)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_INTERVAL_MS)
        self._pulse_timer.timeout.connect(self._tick)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_glow)

    def show_glow(self, color=None, duration_ms=None):
        if color is not None:
            self._color = color
        dur = duration_ms if duration_ms is not None else GLOW_DURATION_MS
        self._hide_timer.setInterval(dur)
        if self._active:
            self._hide_timer.start()
            return
        self._active = True
        self._phase = 0.0
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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        r = self.rect()
        d = GLOW_DEPTH_PX
        a = int(self._alpha * 255)
        glow = QColor(self._color.red(), self._color.green(), self._color.blue(), a)
        clear = QColor(self._color.red(), self._color.green(), self._color.blue(), 0)

        def g(x1, y1, x2, y2):
            gr = QLinearGradient(x1, y1, x2, y2)
            gr.setColorAt(0.0, glow)
            gr.setColorAt(1.0, clear)
            return QBrush(gr)

        p.setBrush(g(0, r.top(), 0, r.top() + d))
        p.drawRect(r.left(), r.top(), r.width(), d)
        p.setBrush(g(0, r.bottom(), 0, r.bottom() - d))
        p.drawRect(r.left(), r.bottom() - d, r.width(), d)
        p.setBrush(g(r.left(), 0, r.left() + d, 0))
        p.drawRect(r.left(), r.top(), d, r.height())
        p.setBrush(g(r.right(), 0, r.right() - d, 0))
        p.drawRect(r.right() - d, r.top(), d, r.height())
        p.end()


class ConfettiOverlay(QWidget):
    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Window | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
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
        self._particles = [{
            "x": random.uniform(0, w), "y": random.uniform(-h * 0.6, 0),
            "dx": random.uniform(-3, 3), "dy": random.uniform(1, 6),
            "color": random.choice(colors),
            "w": random.randint(6, 12), "h": random.randint(3, 8),
            "rot": random.uniform(0, 360), "drot": random.uniform(-6, 6),
        } for _ in range(250)]
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
        for pt in self._particles:
            pt["x"] += pt["dx"]
            pt["y"] += pt["dy"]
            pt["dy"] += 0.12
            pt["dx"] += random.uniform(-0.08, 0.08)
            pt["rot"] += pt["drot"]
        self.update()

    def paintEvent(self, _):
        if not self._active:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        for pt in self._particles:
            c = pt["color"]
            p.setBrush(QColor(c[0], c[1], c[2], 210))
            p.save()
            p.translate(pt["x"], pt["y"])
            p.rotate(pt["rot"])
            p.drawRect(-pt["w"] // 2, -pt["h"] // 2, pt["w"], pt["h"])
            p.restore()
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
#  Lock-In widget
# ═══════════════════════════════════════════════════════════════════════════

class LockInWidget(QWidget):
    completed = pyqtSignal()

    _TITLE_NORMAL = (
        "color: #1e90ff; font-size: 13px; font-weight: bold; background: transparent;")
    _TITLE_ALERT = (
        "color: #ffffff; font-size: 13px; font-weight: bold; background: transparent;")
    _TITLE_DONE = (
        "color: #4caf50; font-size: 13px; font-weight: bold; background: transparent;")

    FADE_DURATION_MS = 5000
    _FADE_TICK_MS    = 40

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.Window | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(210, 86)
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.right() - self.width() - 18, screen.top() + 18)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        self._title = QLabel("Lock In! Focus for 10 min")
        self._title.setStyleSheet(self._TITLE_NORMAL)
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
        self._alert = False
        self._fading = False
        self._bg_r, self._bg_g, self._bg_b = 15, 15, 15
        self._bg_alpha = 80
        self._target_alpha = 80

        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(self._FADE_TICK_MS)
        self._fade_timer.timeout.connect(self._fade_tick)
        self._fade_delay = QTimer(self)
        self._fade_delay.setSingleShot(True)
        self._fade_delay.timeout.connect(self._start_fade_out)

    def paintEvent(self, _):
        if self._bg_alpha <= 0 and not self._alert and not self._completed:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._bg_r, self._bg_g, self._bg_b, int(self._bg_alpha)))
        p.drawRoundedRect(self.rect(), 10, 10)
        p.end()

    _IDLE_ALPHA = 80

    def _fade_tick(self):
        step = 255 * self._FADE_TICK_MS / self.FADE_DURATION_MS
        if self._bg_alpha < self._target_alpha:
            self._bg_alpha = min(self._bg_alpha + step * 4, self._target_alpha)
        elif self._bg_alpha > self._target_alpha:
            self._bg_alpha = max(self._bg_alpha - step, self._target_alpha)
        self.update()
        if abs(self._bg_alpha - self._target_alpha) < 1:
            self._bg_alpha = self._target_alpha
            self._fade_timer.stop()
            self._fading = False

    def _start_fade_out(self):
        self._bg_r, self._bg_g, self._bg_b = 15, 15, 15
        self._target_alpha = self._IDLE_ALPHA
        self._fading = True
        self._fade_timer.start()

    def add_focus_time(self, secs):
        if self._completed:
            return
        self._focus_secs = min(self._focus_secs + secs, LOCKIN_GOAL_SECS)
        mins = int(self._focus_secs) // 60
        s = int(self._focus_secs) % 60
        self._time_label.setText(f"{mins}:{s:02d} / 10:00")
        self._bar.setValue(int(self._focus_secs / LOCKIN_GOAL_SECS * 1000))
        if self._focus_secs >= LOCKIN_GOAL_SECS:
            self._completed = True
            self._fade_timer.stop()
            self._fade_delay.stop()
            self._title.setText("You did it!")
            self._title.setStyleSheet(self._TITLE_DONE)
            self._bg_r, self._bg_g, self._bg_b = 15, 15, 15
            self._bg_alpha = 220
            self._bar.setStyleSheet(
                "QProgressBar { background: #333; border-radius: 5px; }"
                "QProgressBar::chunk { background: #4caf50; border-radius: 5px; }")
            self.update()
            self.completed.emit()

    def set_nudge(self, active):
        if self._completed:
            return
        if active:
            self._alert = True
            self._fading = False
            self._fade_timer.stop()
            self._fade_delay.stop()
            self._bg_r, self._bg_g, self._bg_b = 180, 30, 30
            self._bg_alpha = 255
            self._target_alpha = 255
            self._title.setText("Hey — lock back in!")
            self._title.setStyleSheet(self._TITLE_ALERT)
            self.show()
            self.raise_()
            self.update()
        else:
            if self._alert:
                self._alert = False
                self._title.setText("Lock In! Focus for 10 min")
                self._title.setStyleSheet(self._TITLE_NORMAL)
                self._bg_r, self._bg_g, self._bg_b = 15, 15, 15
                self._bg_alpha = 220
                self.update()
                self._fade_delay.start(self.FADE_DURATION_MS)


# ═══════════════════════════════════════════════════════════════════════════
#  Camera widget
# ═══════════════════════════════════════════════════════════════════════════

class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Starting camera...")
        self.setStyleSheet("color: white; font-size: 16px;")
        self.setMinimumSize(640, 480)

    def update_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)


# ═══════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════

_TAB_STYLE = f"""
QTabWidget::pane {{
    border: none;
    background: {_C['bg']};
}}
QTabBar {{
    background: {_C['surface']};
}}
QTabBar::tab {{
    background: {_C['surface']};
    color: {_C['onSurfV']};
    padding: 10px 28px;
    border: none;
    font-family: 'Inter';
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
}}
QTabBar::tab:selected {{
    color: {_C['primary']};
    border-bottom: 2px solid {_C['primary']};
}}
QTabBar::tab:hover {{
    background: {_C['surface2']};
}}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodLens")
        self.setStyleSheet(f"background-color: {_C['bg']};")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Tab widget ───────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(_TAB_STYLE)
        root.addWidget(self._tabs)

        # Camera tab
        cam_page = QWidget()
        cam_page.setStyleSheet(f"background: {_C['bg']};")
        cam_lay = QVBoxLayout(cam_page)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        cam_lay.setSpacing(0)

        self._camera = CameraWidget()
        cam_lay.addWidget(self._camera)

        self._stress_label = QLabel("Stress: --")
        self._stress_label.setStyleSheet(
            f"color: #ccc; font-size: 13px; padding: 4px 10px;"
            f"background-color: {_C['surface']};")
        self._stress_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cam_lay.addWidget(self._stress_label)

        bar = QWidget()
        bar.setStyleSheet(f"background-color: {_C['surface']};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 4, 10, 4)

        self._debug_label = QLabel("Waiting for analysis...")
        self._debug_label.setStyleSheet("color: #aaa; font-size: 12px;")
        bar_layout.addWidget(self._debug_label, stretch=1)

        test_btn = QPushButton("Test Glow")
        test_btn.setStyleSheet(
            f"background:{_C['primaryC']}; color:white; border-radius:4px;"
            "padding: 4px 12px; font-size:12px;")
        test_btn.clicked.connect(self._test_glow)
        bar_layout.addWidget(test_btn)

        hide_btn = QPushButton("Hide Glow")
        hide_btn.setStyleSheet(
            "background:#444; color:white; border-radius:4px;"
            "padding: 4px 12px; font-size:12px;")
        bar_layout.addWidget(hide_btn)
        cam_lay.addWidget(bar)

        self._tabs.addTab(cam_page, "CAMERA")

        # Dashboard tab
        self._dashboard = DashboardWidget()
        self._tabs.addTab(self._dashboard, "DASHBOARD")

        self.resize(900, 680)

        # ── Overlays ─────────────────────────────────────────────────────
        self._glow = GlowOverlay()
        self._glow.glow_hidden.connect(self._on_glow_hidden)
        hide_btn.clicked.connect(self._glow.hide_glow)
        self._confetti = ConfettiOverlay()

        self._lock_in = LockInWidget()
        self._lock_in.completed.connect(self._on_lockin_complete)
        self._lock_in.show()

        # ── State ────────────────────────────────────────────────────────
        self._stress_start    = None
        self._glow_source     = None
        self._gaze_away_start = None
        self._boredom_fired   = False
        self._was_looking     = False
        self._gaze_last_time  = None
        self._total_focus     = 0.0   # tracked for dashboard

        # ── Thread ───────────────────────────────────────────────────────
        self._thread = EmotionThread()
        self._thread.frame_ready.connect(self._camera.update_frame)
        self._thread.emotion_found.connect(self._on_emotion)
        self._thread.stress_found.connect(self._on_stress)
        self._thread.gaze_status.connect(self._on_gaze)
        self._thread.log_message.connect(self._on_log)
        self._thread.start()

    # ── callbacks ────────────────────────────────────────────────────────

    def _test_glow(self):
        self._debug_label.setText("Manual test triggered")
        self._glow.show_glow()

    def _on_log(self, msg):
        print(msg)
        self._debug_label.setText(msg)

    def _on_emotion(self, dominant, scores):
        top = sorted(scores.items(), key=lambda x: -x[1])
        status = "  |  ".join(f"{e}: {s:.1f}%" for e, s in top[:4])
        self._debug_label.setText(f"Mood: {dominant.upper()}  --  {status}")
        self._dashboard.update_emotion(dominant, scores)

    # ── stress ───────────────────────────────────────────────────────────

    def _on_stress(self, score, au_row):
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
            f"padding: 4px 10px; background-color: {_C['surface']};")

        self._dashboard.update_stress(score)

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

    def _on_gaze(self, looking):
        now = time.monotonic()

        if looking:
            if self._was_looking and self._gaze_last_time is not None:
                dt = now - self._gaze_last_time
                if dt < 1.0:
                    self._lock_in.add_focus_time(dt)
                    self._total_focus += dt
                    self._dashboard.update_focus(self._total_focus)

            if self._gaze_away_start is not None:
                self._gaze_away_start = None
                self._boredom_fired = False
                self._lock_in.set_nudge(False)
        else:
            if self._gaze_away_start is None:
                self._gaze_away_start = now
                self._boredom_fired = False

            elapsed = now - self._gaze_away_start
            if elapsed >= GAZE_AWAY_SECS and not self._boredom_fired:
                self._glow.show_glow(
                    QColor(220, 50, 50), duration_ms=BOREDOM_GLOW_MS)
                self._glow_source = "boredom"
                self._boredom_fired = True
                self._lock_in.set_nudge(True)
                self._dashboard.add_distraction()

        self._gaze_last_time = now
        self._was_looking = looking

    def _on_glow_hidden(self):
        self._glow_source = None

    def _on_lockin_complete(self):
        self._lock_in.show()
        self._lock_in.raise_()
        self._confetti.show_confetti(duration_ms=8000)

    def closeEvent(self, event):
        self._glow.hide_glow()
        self._confetti.hide_confetti()
        self._lock_in.close()
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
