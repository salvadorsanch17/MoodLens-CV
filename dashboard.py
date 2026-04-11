from collections import defaultdict
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QFont, QPen
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QGridLayout, QSizePolicy,
)

# ── design tokens (from The Sentient Interface) ──────────────────────────────
_C = {
    "bg":       "#121416",
    "surface":  "#1a1c1e",
    "surface2": "#1e2022",
    "surface3": "#282a2c",
    "surfaceH": "#333537",
    "primary":  "#71d7cd",
    "primaryC": "#008178",
    "secondary":"#94ccff",
    "tertiary": "#ffb59e",
    "onSurf":   "#e2e2e5",
    "onSurfV":  "#bdc9c8",
    "outline":  "#3e4949",
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
        f = QFont("Inter", 8)
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

        # ── build ────────────────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

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
    def _lbl(text, color=None, size=14, bold=False, family="Inter", upper=False):
        color = color or _C["onSurf"]
        l = QLabel(text)
        weight = "700" if bold else "400"
        extra = "text-transform: uppercase; letter-spacing: 2px;" if upper else ""
        l.setStyleSheet(
            f"color: {color}; font-size: {size}px; font-weight: {weight};"
            f"font-family: '{family}'; background: transparent; {extra}")
        return l

    # ── header ────────────────────────────────────────────────────────────

    def _build_header(self):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 8)
        icon = self._lbl("◆", _C["primary"], 20, True)
        title = self._lbl("MoodLens", _C["primary"], 18, True, "Manrope")
        row.addWidget(icon)
        row.addWidget(title)
        row.addStretch()
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

    def _refresh_insight(self):
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
