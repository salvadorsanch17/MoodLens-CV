from collections import defaultdict
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QFont
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QGridLayout, QSizePolicy, QPushButton, QComboBox,
)

# ── design tokens (from The Sentient Interface) ──────────────────────────────
_C = {
    "bg":       "#060B1C",   # near-black navy
    "surface":  "#0A1228",   # dark navy surface
    "surface2": "#0D1632",   # slightly lighter
    "surface3": "#12203E",   # medium navy
    "surfaceH": "#1C2A52",   # highlight surface
    "primary":  "#82C8E5",   # sky blue — bright accent on dark
    "primaryC": "#0047AB",   # cobalt blue — interactive / filled states
    "secondary":"#82C8E5",   # sky blue secondary
    "tertiary": "#6D8196",   # slate blue-grey — muted accent / high-stress bars
    "onSurf":   "#D4E8F8",   # light cool-white text
    "onSurfV":  "#6D8196",   # muted slate text
    "outline":  "#1C2A52",   # navy border
}

_EMOTION_STATES = {
    "happy":    ("Positive",  "Positive emotional patterns detected. Your environment is well-optimized."),
    "sad":      ("Low Mood",  "Low mood patterns detected. Consider a short break or a change of scenery."),
    "angry":    ("Elevated",  "Elevated arousal detected. Deep breathing may help restore your baseline."),
    "fear":     ("Anxious",   "Heightened anxiety patterns observed. Ambient adjustments recommended."),
    "surprise": ("Alert",     "Unexpected stimulus response detected. Monitoring engagement levels."),
    "disgust":  ("Aversion",  "Aversion response detected. Consider adjusting your current activity."),
    "neutral":  ("Focused",   "Baseline emotional state. Conditions are optimal for deep focus work."),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Toggle switch widget
# ═══════════════════════════════════════════════════════════════════════════

class ToggleSwitch(QWidget):
    """Animated pill-shaped toggle switch."""
    toggled = pyqtSignal(bool)

    _W, _H = 40, 22

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._offset = 1.0 if checked else 0.0
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(12)
        self._timer.timeout.connect(self._tick)

    def isChecked(self):
        return self._checked

    def setChecked(self, val):
        if val == self._checked:
            return
        self._checked = val
        self._timer.start()
        self.toggled.emit(self._checked)

    def mousePressEvent(self, _):
        self.setChecked(not self._checked)

    def _tick(self):
        target = 1.0 if self._checked else 0.0
        diff = target - self._offset
        if abs(diff) < 0.04:
            self._offset = target
            self._timer.stop()
        else:
            self._offset += diff * 0.3
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)

        # Track — interpolate from surfaceH (off) to primaryC (on)
        off_c, on_c = QColor(_C["surfaceH"]), QColor(_C["primaryC"])
        t = self._offset
        track = QColor(
            int(off_c.red()   + (on_c.red()   - off_c.red())   * t),
            int(off_c.green() + (on_c.green() - off_c.green()) * t),
            int(off_c.blue()  + (on_c.blue()  - off_c.blue())  * t),
        )
        p.setBrush(track)
        p.drawRoundedRect(0, 0, self._W, self._H, self._H // 2, self._H // 2)

        # Knob
        d = self._H - 4
        knob_x = int(2 + self._offset * (self._W - d - 4))
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(knob_x, 2, d, d)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
#  Bar chart helper
# ═══════════════════════════════════════════════════════════════════════════

class BarChartWidget(QWidget):
    """Bottom-aligned bar chart with rounded tops."""

    def __init__(self, count, labels=None, parent=None):
        super().__init__(parent)
        self._values = [0.0] * count   # 0.0 – 1.0
        self._colors = [QColor(_C["surfaceH"])] * count
        self._labels = labels or [""] * count
        self.setMinimumHeight(160)

    def set_data(self, values, colors=None):
        self._values = values
        if colors:
            self._colors = colors
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(self._values)
        if n == 0:
            p.end()
            return

        gap = 6
        label_h = 18
        chart_h = self.height() - label_h
        bar_w = max(8, (self.width() - gap * (n + 1)) // n)
        total_w = bar_w * n + gap * (n - 1)
        x0 = (self.width() - total_w) // 2

        p.setPen(Qt.NoPen)
        for i, val in enumerate(self._values):
            x = x0 + i * (bar_w + gap)
            bh = max(4, int(chart_h * max(0, min(val, 1.0))))
            p.setBrush(self._colors[i] if i < len(self._colors) else QColor(_C["surfaceH"]))
            p.drawRoundedRect(x, chart_h - bh, bar_w, bh, 4, 4)

        # Labels
        p.setPen(QColor(_C["onSurfV"]))
        f = QFont("Cabin", 8)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
        p.setFont(f)
        for i, lbl in enumerate(self._labels):
            x = x0 + i * (bar_w + gap)
            p.drawText(x, chart_h + 2, bar_w, label_h, Qt.AlignHCenter | Qt.AlignTop, lbl)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
#  Dashboard widget
# ═══════════════════════════════════════════════════════════════════════════

class DashboardWidget(QWidget):

    false_positive_clicked  = pyqtSignal()
    true_positive_clicked   = pyqtSignal()
    feedback_dismissed      = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {_C['bg']};")

        # ── data state ───────────────────────────────────────────────────
        self._hourly_stress = defaultdict(list)
        self._distraction_count = 0
        self._focus_secs = 0.0
        self._current_stress = 0.0
        self._stress_sum = 0.0
        self._stress_n = 0

        # emotion log: list of {emotion, stress, app, session_min, hour}
        self._emotion_log = []

        # ── build ────────────────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Feedback banner (hidden until a stress alert fires)
        self._feedback_banner = self._build_feedback_banner()
        self._feedback_banner.hide()
        outer.addWidget(self._feedback_banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {_C['bg']}; border: none; }}"
            f"QScrollBar:vertical {{ background: {_C['surface']}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {_C['outline']};"
            f"  border-radius: 3px; min-height: 30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}")

        content = QWidget()
        content.setStyleSheet(f"background: {_C['bg']};")
        self._lay = QVBoxLayout(content)
        self._lay.setContentsMargins(28, 20, 28, 28)
        self._lay.setSpacing(16)

        self._build_header()
        self._build_status_row()
        self._build_metrics_row()
        self._build_charts_row()
        self._build_correlations()
        self._build_insight()
        self._lay.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _card(bg=None):
        bg = bg or _C["surface"]
        w = QWidget()
        w.setAttribute(Qt.WA_StyledBackground, True)
        w.setStyleSheet(f"background: {bg}; border-radius: 12px;")
        return w

    @staticmethod
    def _lbl(text, color=None, size=14, bold=False, family="Cabin", upper=False):
        color = color or _C["onSurf"]
        l = QLabel(text)
        weight = "700" if bold else "400"
        extra = "text-transform: uppercase; letter-spacing: 2px;" if upper else ""
        l.setStyleSheet(
            f"color: {color}; font-size: {size}px; font-weight: {weight};"
            f"font-family: '{family}'; background: transparent; {extra}")
        return l

    # ── stress feedback banner ────────────────────────────────────────────

    def _build_feedback_banner(self) -> QWidget:
        banner = QWidget()
        banner.setAttribute(Qt.WA_StyledBackground, True)
        banner.setStyleSheet(
            "background: #2a1f0e; border-bottom: 1px solid #6b4c1e;")

        row = QHBoxLayout(banner)
        row.setContentsMargins(20, 10, 12, 10)
        row.setSpacing(12)

        icon = QLabel("⚠")
        icon.setStyleSheet(
            "color: #f0a030; font-size: 15px; background: transparent;")
        row.addWidget(icon)

        msg = QLabel("Stress was detected — were you actually stressed?")
        msg.setStyleSheet(
            "color: #e8c88a; font-size: 12px; font-family: 'Cabin';"
            " background: transparent;")
        row.addWidget(msg, stretch=1)

        yes_btn = QPushButton("Yes, stressed")
        yes_btn.setCursor(Qt.PointingHandCursor)
        yes_btn.setStyleSheet(
            "QPushButton { background: #0e2a1f; color: #4caf50;"
            " border: 1px solid #1e6b4c; border-radius: 6px;"
            " padding: 4px 14px; font-size: 11px; font-weight: 600;"
            " font-family: 'Cabin'; }"
            "QPushButton:hover { background: #12402e; }")
        yes_btn.clicked.connect(self.true_positive_clicked)
        yes_btn.clicked.connect(banner.hide)
        row.addWidget(yes_btn)

        not_stressed_btn = QPushButton("Not stressed")
        not_stressed_btn.setCursor(Qt.PointingHandCursor)
        not_stressed_btn.setStyleSheet(
            "QPushButton { background: #3d2a0a; color: #f0a030;"
            " border: 1px solid #6b4c1e; border-radius: 6px;"
            " padding: 4px 14px; font-size: 11px; font-weight: 600;"
            " font-family: 'Cabin'; }"
            "QPushButton:hover { background: #5a3e12; }")
        not_stressed_btn.clicked.connect(self.false_positive_clicked)
        not_stressed_btn.clicked.connect(banner.hide)
        row.addWidget(not_stressed_btn)

        dismiss_btn = QPushButton("×")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setCursor(Qt.PointingHandCursor)
        dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #888;"
            " border: none; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { color: #ccc; }")
        dismiss_btn.clicked.connect(self.feedback_dismissed)
        dismiss_btn.clicked.connect(banner.hide)
        row.addWidget(dismiss_btn)

        return banner

    def show_stress_banner(self):
        self._feedback_banner.show()

    def hide_stress_banner(self):
        self._feedback_banner.hide()

    # ── header ────────────────────────────────────────────────────────────

    def _build_header(self):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 8)
        icon = self._lbl("◆", _C["primary"], 20, True)
        title = self._lbl("MoodLens", _C["primary"], 18, True, "Manrope")
        row.addWidget(icon)
        row.addWidget(title)
        row.addStretch()

        # Sound toggle
        snd_icon = self._lbl("🔔", size=14)
        snd_icon.setStyleSheet("background: transparent;")
        row.addWidget(snd_icon)

        row.addWidget(self._lbl("Sound", _C["onSurfV"], 11))
        row.addSpacing(4)

        self._sound_btn = ToggleSwitch(checked=False)
        row.addWidget(self._sound_btn)

        row.addSpacing(8)

        self._sound_selector = QComboBox()
        self._sound_selector.addItem("Rain", "stress_alert.mp3")
        self._sound_selector.addItem("Brown Noise", "brown_noise.mp3")
        self._sound_selector.setStyleSheet(f"""
            QComboBox {{
                background: {_C['surface3']};
                color: {_C['onSurfV']};
                border: 1px solid {_C['outline']};
                border-radius: 10px;
                padding: 4px 10px 4px 12px;
                font-size: 11px;
                font-weight: 600;
                font-family: 'Cabin';
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
            QComboBox QAbstractItemView {{
                background: {_C['surface3']};
                color: {_C['onSurf']};
                selection-background-color: {_C['primaryC']};
                border: 1px solid {_C['outline']};
            }}
        """)
        row.addWidget(self._sound_selector)

        w = QWidget()
        w.setLayout(row)
        w.setStyleSheet("background: transparent;")
        self._lay.addWidget(w)

    # ── status row ────────────────────────────────────────────────────────

    def _build_status_row(self):
        row = QHBoxLayout()
        row.setSpacing(16)

        # -- State card (larger) --
        card = self._card()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setMinimumHeight(170)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(8)

        # Status dot + label
        dot_row = QHBoxLayout()
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f"background: {_C['secondary']}; border-radius: 5px;")
        dot_row.addWidget(dot)
        dot_row.addWidget(self._lbl("CURRENT STATE", _C["onSurfV"], 10, True, upper=True))
        dot_row.addStretch()
        lay.addLayout(dot_row)

        self._state_title = self._lbl("Focused", _C["onSurf"], 40, True, "Manrope")
        lay.addWidget(self._state_title)
        self._state_desc = self._lbl(
            _EMOTION_STATES["neutral"][1], _C["onSurfV"], 13)
        self._state_desc.setWordWrap(True)
        lay.addWidget(self._state_desc)
        lay.addStretch()
        row.addWidget(card, 2)

        # -- Pulse / stress card --
        pulse = self._card(_C["surface2"])
        pulse.setMinimumHeight(170)
        play = QVBoxLayout(pulse)
        play.setContentsMargins(20, 24, 20, 24)
        play.setAlignment(Qt.AlignCenter)
        play.setSpacing(8)

        self._pulse_value = self._lbl("0%", _C["primary"], 42, True, "Manrope")
        self._pulse_value.setAlignment(Qt.AlignCenter)
        play.addStretch()
        play.addWidget(self._pulse_value)
        play.addWidget(self._lbl("STRESS LEVEL", _C["onSurfV"], 9, True, upper=True))
        play.addStretch()
        row.addWidget(pulse, 1)

        w = QWidget()
        w.setLayout(row)
        w.setStyleSheet("background: transparent;")
        self._lay.addWidget(w)

    # ── metrics row ───────────────────────────────────────────────────────

    def _build_metrics_row(self):
        row = QHBoxLayout()
        row.setSpacing(12)

        def metric(icon_char, badge_text, badge_color, label_text, initial_value):
            card = self._card()
            card.setMinimumHeight(120)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(20, 18, 20, 18)
            lay.setSpacing(0)
            # Top: icon + badge
            top = QHBoxLayout()
            icon = self._lbl(icon_char, _C["onSurfV"], 18)
            top.addWidget(icon)
            top.addStretch()
            if badge_text:
                badge = self._lbl(badge_text, badge_color, 10, True)
                badge.setStyleSheet(
                    f"color: {badge_color}; font-size: 10px; font-weight: 600;"
                    f"background: {badge_color}22; padding: 2px 8px;"
                    f"border-radius: 8px;")
                top.addWidget(badge)
            lay.addLayout(top)
            lay.addSpacing(12)
            # Label
            lay.addWidget(self._lbl(label_text, _C["onSurfV"], 10, True, upper=True))
            lay.addSpacing(2)
            # Value
            val = self._lbl(initial_value, _C["onSurf"], 32, True, "Manrope")
            lay.addWidget(val)
            return card, val

        c1, self._avg_stress_val = metric(
            "~", "", "", "Average Stress", "—")
        c2, self._distractions_val = metric(
            "!", "", "", "Distractions", "0")
        c3, self._focus_val = metric(
            "T", "", "", "Focus Time", "0:00")

        row.addWidget(c1)
        row.addWidget(c2)
        row.addWidget(c3)

        w = QWidget()
        w.setLayout(row)
        w.setStyleSheet("background: transparent;")
        self._lay.addWidget(w)

    # ── charts row ────────────────────────────────────────────────────────

    def _build_charts_row(self):
        row = QHBoxLayout()
        row.setSpacing(16)

        # -- Heatmap --
        hcard = self._card()
        hcard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        hcard.setMinimumHeight(260)
        hlay = QVBoxLayout(hcard)
        hlay.setContentsMargins(24, 24, 24, 16)
        hlay.setSpacing(8)
        hlay.addWidget(self._lbl("Stress Heatmap", _C["onSurf"], 16, True, "Manrope"))
        hlay.addWidget(self._lbl("Hourly physiological load", _C["onSurfV"], 11))
        hlay.addSpacing(8)

        labels_h = ["8a", "9a", "10a", "11a", "12p",
                     "1p", "2p", "3p", "4p", "5p", "6p", "7p"]
        self._heatmap = BarChartWidget(12, labels_h)
        hlay.addWidget(self._heatmap, 1)
        row.addWidget(hcard, 3)

        # -- Weekly focus --
        wcard = self._card()
        wcard.setMinimumHeight(260)
        wlay = QVBoxLayout(wcard)
        wlay.setContentsMargins(24, 24, 24, 16)
        wlay.setSpacing(8)
        wlay.addWidget(self._lbl("Weekly Focus", _C["onSurf"], 16, True, "Manrope"))
        wlay.addWidget(self._lbl("Deep work consistency", _C["onSurfV"], 11))
        wlay.addSpacing(8)

        self._weekly = BarChartWidget(7, ["S", "M", "T", "W", "T", "F", "S"])
        wlay.addWidget(self._weekly, 1)
        row.addWidget(wcard, 2)

        w = QWidget()
        w.setLayout(row)
        w.setStyleSheet("background: transparent;")
        self._lay.addWidget(w)

        # Init weekly with placeholder + today
        self._daily_focus = [0.0] * 7
        self._refresh_weekly()

    # ── correlations card ─────────────────────────────────────────────────

    def _build_correlations(self):
        card = self._card()
        card.setMinimumHeight(160)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 24, 24, 20)
        lay.setSpacing(8)
        lay.addWidget(self._lbl(
            "App Correlations", _C["onSurf"], 16, True, "Manrope"))
        lay.addWidget(self._lbl(
            "Emotion patterns by application", _C["onSurfV"], 11))
        lay.addSpacing(8)

        # Container for dynamic correlation rows
        self._corr_container = QVBoxLayout()
        self._corr_container.setSpacing(6)

        placeholder = self._lbl(
            "Gathering data — correlations appear after a few minutes of use.",
            _C["onSurfV"], 12)
        placeholder.setWordWrap(True)
        self._corr_container.addWidget(placeholder)

        lay.addLayout(self._corr_container)
        lay.addStretch()
        self._lay.addWidget(card)

    # ── insight ───────────────────────────────────────────────────────────

    def _build_insight(self):
        card = QWidget()
        card.setAttribute(Qt.WA_StyledBackground, True)
        card.setStyleSheet(
            f"background: {_C['surface2']}88; border-radius: 16px;")
        card.setMinimumHeight(100)
        lay = QHBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(20)

        icon = self._lbl("*", _C["primary"], 28, True)
        icon.setFixedSize(48, 48)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(
            f"color: {_C['primary']}; font-size: 28px; font-weight: bold;"
            f"background: {_C['primaryC']}33; border-radius: 24px;")
        lay.addWidget(icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        self._insight_title = self._lbl(
            "Session Active", _C["onSurf"], 14, True, "Manrope")
        self._insight_desc = self._lbl(
            "MoodLens is monitoring your emotional and focus patterns. "
            "Insights will appear as data accumulates.",
            _C["onSurfV"], 12)
        self._insight_desc.setWordWrap(True)
        text_col.addWidget(self._insight_title)
        text_col.addWidget(self._insight_desc)
        lay.addLayout(text_col, 1)

        self._lay.addWidget(card)

    # ── sound toggle ─────────────────────────────────────────────────────

    @property
    def sound_enabled(self):
        return self._sound_btn.isChecked()

    @property
    def selected_sound(self) -> str:
        """Returns the filename of the currently selected sound."""
        return self._sound_selector.currentData()

    # ── public update API ─────────────────────────────────────────────────

    def update_emotion(self, dominant, scores):
        name, desc = _EMOTION_STATES.get(dominant, ("Unknown", ""))
        self._state_title.setText(name)
        self._state_desc.setText(desc)

    def update_stress(self, score):
        self._current_stress = score
        self._stress_sum += score
        self._stress_n += 1
        avg = self._stress_sum / self._stress_n

        self._pulse_value.setText(f"{score:.0f}%")
        self._avg_stress_val.setText(f"{avg:.0f}%")

        # Record hourly
        hour = datetime.now().hour
        self._hourly_stress[hour].append(score)
        self._refresh_heatmap()
        self._refresh_insight()

    def update_focus(self, secs):
        self._focus_secs = secs
        m = int(secs) // 60
        s = int(secs) % 60
        if m >= 60:
            self._focus_val.setText(f"{m // 60}h {m % 60:02d}m")
        else:
            self._focus_val.setText(f"{m}:{s:02d}")

        # Update today's bar in weekly chart
        dow = datetime.now().weekday()  # 0=Mon
        chart_idx = (dow + 1) % 7       # shift so 0=Sun
        self._daily_focus[chart_idx] = secs
        self._refresh_weekly()

    def add_distraction(self):
        self._distraction_count += 1
        self._distractions_val.setText(str(self._distraction_count))
        self._refresh_insight()

    def add_log_entry(self, emotion, stress, app, session_min):
        self._emotion_log.append({
            "emotion": emotion,
            "stress": stress,
            "app": app,
            "session_min": session_min,
            "hour": datetime.now().hour,
        })
        self._refresh_correlations()

    # ── internal refreshes ────────────────────────────────────────────────

    def _refresh_heatmap(self):
        values = []
        colors = []
        for h in range(8, 20):  # 8 AM – 7 PM
            samples = self._hourly_stress.get(h, [])
            if samples:
                avg = sum(samples) / len(samples)
            else:
                avg = 0
            values.append(max(0.05, avg / 100))  # min bar height for visibility
            if avg < 25:
                colors.append(QColor(_C["surfaceH"]))
            elif avg < 50:
                c = QColor(_C["primary"])
                c.setAlpha(int(80 + avg * 2))
                colors.append(c)
            elif avg < 70:
                c = QColor(_C["primary"])
                c.setAlpha(200)
                colors.append(c)
            else:
                colors.append(QColor(_C["tertiary"]))
        self._heatmap.set_data(values, colors)

    def _refresh_weekly(self):
        max_secs = max(max(self._daily_focus), 1)
        values = [s / max_secs * 0.9 + 0.05 if s > 0 else 0.05
                  for s in self._daily_focus]
        colors = []
        for v in values:
            c = QColor(_C["secondary"])
            c.setAlpha(int(40 + v * 200))
            colors.append(c)
        self._weekly.set_data(values, colors)

    _NEG_EMOTIONS = {"angry", "sad", "fear", "disgust"}

    def _refresh_correlations(self):
        # Need a minimum amount of data before showing correlations
        if len(self._emotion_log) < 12:
            return

        # Group entries by app
        app_data = defaultdict(list)
        for entry in self._emotion_log:
            app_data[entry["app"]].append(entry)

        # Build correlation insights: per-app avg stress, dominant emotion,
        # and time-in-session patterns
        insights = []
        for app, entries in app_data.items():
            if len(entries) < 3 or app == "Unknown":
                continue
            avg_stress = sum(e["stress"] for e in entries) / len(entries)
            neg_count = sum(1 for e in entries
                           if e["emotion"] in self._NEG_EMOTIONS)
            neg_pct = neg_count / len(entries) * 100

            # Check if stress rises with session duration while in this app
            late_entries = [e for e in entries if e["session_min"] >= 60]
            early_entries = [e for e in entries if e["session_min"] < 30]
            fatigue_note = ""
            if len(late_entries) >= 3 and len(early_entries) >= 3:
                late_stress = sum(e["stress"] for e in late_entries) / len(late_entries)
                early_stress = sum(e["stress"] for e in early_entries) / len(early_entries)
                if late_stress - early_stress > 10:
                    mins = int(min(e["session_min"] for e in late_entries
                                   if e["stress"] > avg_stress * 1.1))
                    fatigue_note = f"  Stress rises after ~{mins} min in session."

            # Determine the most common negative emotion for this app
            neg_emotions = [e["emotion"] for e in entries
                            if e["emotion"] in self._NEG_EMOTIONS]
            top_neg = ""
            if neg_emotions:
                from collections import Counter
                top_neg = Counter(neg_emotions).most_common(1)[0][0]

            insights.append({
                "app": app,
                "samples": len(entries),
                "avg_stress": avg_stress,
                "neg_pct": neg_pct,
                "top_neg": top_neg,
                "fatigue": fatigue_note,
            })

        # Sort by avg stress descending (most stressful apps first)
        insights.sort(key=lambda x: -x["avg_stress"])

        # Clear old rows
        while self._corr_container.count():
            item = self._corr_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not insights:
            lbl = self._lbl(
                "Not enough per-app data yet. Keep using MoodLens!",
                _C["onSurfV"], 12)
            lbl.setWordWrap(True)
            self._corr_container.addWidget(lbl)
            return

        for info in insights[:5]:  # top 5 apps
            row_w = QWidget()
            row_w.setAttribute(Qt.WA_StyledBackground, True)
            row_w.setStyleSheet(
                f"background: {_C['surface2']}; border-radius: 8px;")
            row_lay = QHBoxLayout(row_w)
            row_lay.setContentsMargins(14, 10, 14, 10)
            row_lay.setSpacing(12)

            # App name
            name = self._lbl(info["app"], _C["onSurf"], 13, True)
            name.setFixedWidth(140)
            row_lay.addWidget(name)

            # Stress bar
            stress_pct = min(info["avg_stress"] / 100, 1.0)
            bar_bg = QWidget()
            bar_bg.setFixedHeight(8)
            bar_bg.setMinimumWidth(80)
            bar_bg.setAttribute(Qt.WA_StyledBackground, True)
            bar_bg.setStyleSheet(
                f"background: {_C['surfaceH']}; border-radius: 4px;")

            bar_fill = QWidget(bar_bg)
            bar_fill.setFixedHeight(8)
            fill_w = max(4, int(stress_pct * 80))
            bar_fill.setFixedWidth(fill_w)
            if info["avg_stress"] < 33:
                fill_color = _C["primary"]
            elif info["avg_stress"] < 60:
                fill_color = _C["secondary"]
            else:
                fill_color = _C["tertiary"]
            bar_fill.setStyleSheet(
                f"background: {fill_color}; border-radius: 4px;")
            row_lay.addWidget(bar_bg)

            # Stress %
            pct_lbl = self._lbl(
                f"{info['avg_stress']:.0f}%", _C["onSurfV"], 11, True)
            pct_lbl.setFixedWidth(36)
            row_lay.addWidget(pct_lbl)

            # Description
            desc_parts = []
            if info["neg_pct"] > 30 and info["top_neg"]:
                desc_parts.append(
                    f"{info['neg_pct']:.0f}% {info['top_neg']}")
            if info["fatigue"]:
                desc_parts.append(info["fatigue"].strip())
            desc_text = "  ·  ".join(desc_parts) if desc_parts else "baseline"
            desc = self._lbl(desc_text, _C["onSurfV"], 11)
            desc.setWordWrap(True)
            row_lay.addWidget(desc, 1)

            self._corr_container.addWidget(row_w)

    def _get_top_correlation_insight(self):
        """Return (title, desc) if there's a noteworthy app correlation, else None."""
        if len(self._emotion_log) < 18:
            return None
        app_data = defaultdict(list)
        for entry in self._emotion_log:
            if entry["app"] != "Unknown":
                app_data[entry["app"]].append(entry)
        for app, entries in sorted(
                app_data.items(),
                key=lambda kv: -sum(e["stress"] for e in kv[1]) / len(kv[1])):
            if len(entries) < 5:
                continue
            avg = sum(e["stress"] for e in entries) / len(entries)
            neg = [e["emotion"] for e in entries
                   if e["emotion"] in self._NEG_EMOTIONS]
            if avg > 45 and neg:
                from collections import Counter
                top = Counter(neg).most_common(1)[0][0]
                return (
                    f"Pattern: {app}",
                    f"You're consistently {top} while using {app} "
                    f"(avg stress {avg:.0f}%). Consider taking breaks or "
                    f"adjusting your workflow in that app.")
        return None

    def _refresh_insight(self):
        corr = self._get_top_correlation_insight()
        if corr:
            self._insight_title.setText(corr[0])
            self._insight_desc.setText(corr[1])
            return
        if self._distraction_count >= 10:
            self._insight_title.setText("High Distraction Pattern")
            self._insight_desc.setText(
                "You've looked away from the screen frequently. Consider silencing "
                "notifications and closing unnecessary tabs to maintain flow state.")
        elif self._current_stress > 60:
            self._insight_title.setText("Elevated Stress Levels")
            self._insight_desc.setText(
                "Your stress has been consistently high. A short breathing exercise "
                "or a brief walk may help restore your baseline.")
        elif self._focus_secs > 30 * 60:
            self._insight_title.setText("Strong Focus Session")
            self._insight_desc.setText(
                "Excellent sustained attention today. Remember to take regular "
                "breaks to maintain performance over the long haul.")
        else:
            self._insight_title.setText("Session Active")
            self._insight_desc.setText(
                "MoodLens is monitoring your emotional and focus patterns. "
                "Insights will appear as data accumulates.")
