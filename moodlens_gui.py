import sys
import math
import cv2
import numpy as np
from deepface import DeepFace

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget,
    QVBoxLayout, QPushButton, QHBoxLayout
)

# ── tunables ─────────────────────────────────────────────────────────────────
ANALYZE_EVERY_N_FRAMES = 8
GLOW_DURATION_MS       = 2 * 60 * 1000   # 2 minutes
GLOW_LAYERS            = 32              # more layers = smoother gradient
GLOW_DEPTH_PX          = 90             # wider spread = softer edge
PULSE_INTERVAL_MS      = 40
# ─────────────────────────────────────────────────────────────────────────────


class EmotionThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    emotion_found = pyqtSignal(str, dict)
    log_message   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_message.emit("ERROR: Could not open webcam")
            return

        frame_count = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            self.frame_ready.emit(frame.copy())

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

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(PULSE_INTERVAL_MS)
        self._pulse_timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(GLOW_DURATION_MS)
        self._hide_timer.timeout.connect(self.hide_glow)

    # ── public API ────────────────────────────────────────────────────────────

    def show_glow(self):
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
        self._phase = (self._phase + 0.025) % (2 * math.pi)   # slower pulse
        self._alpha = 0.35 + 0.30 * (0.5 + 0.5 * math.sin(self._phase))  # gentler range
        self.update()

    def paintEvent(self, _):
        if not self._active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        for i in range(GLOW_LAYERS):
            t            = i / GLOW_LAYERS               # 0 outermost → 1 innermost
            layer_alpha  = int(self._alpha * 255 * (1.0 - t) ** 2.8)  # steep falloff = soft edge
            margin       = int(i * GLOW_DEPTH_PX / GLOW_LAYERS)
            stroke       = max(1, int(GLOW_DEPTH_PX / GLOW_LAYERS))

            color = QColor(30, 144, 255, layer_alpha)    # Dodger Blue
            pen   = QPen(color, stroke)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect.adjusted(margin, margin, -margin, -margin))

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

        # Emotion thread
        self._thread = EmotionThread()
        self._thread.frame_ready.connect(self._camera.update_frame)
        self._thread.emotion_found.connect(self._on_emotion)
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

        if scores.get("sad", 0) > 20 or scores.get("fear", 0) > 20:
            self._glow.show_glow()

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
