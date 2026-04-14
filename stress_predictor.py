"""
stress_predictor.py – Predictive stress model for MoodLens.

Predicts whether the user is likely to become stressed in the next 2–5 minutes
using time-of-day, session length, recent stress patterns, and input behaviour
(mouse clicks / movement).  Starts with a heuristic fallback and automatically
switches to a trained GradientBoosting model once enough self-labelled data has
been collected from the user's own session.

Intervention levels (based on predicted probability):
    0  prob < 0.30   →  none
    1  0.30 – 0.50   →  subtle   (barely-perceptible warm shift)
    2  0.50 – 0.70   →  gentle   (light warm tint + dashboard hint)
    3  0.70+         →  proactive (warm tint + preemptive break suggestion)
"""

import math
import time
import pickle
import pathlib
from collections import deque

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from pynput import mouse as pynput_mouse
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False

# ── Configuration ────────────────────────────────────────────────────────────

FEATURE_INTERVAL_SECS    = 10          # how often to snap a feature vector
LOOKBACK_WINDOW_SECS     = 5 * 60     # rolling window for stress stats
PREDICTION_HORIZON_MIN_S = 2 * 60     # predict stress >= this far ahead …
PREDICTION_HORIZON_MAX_S = 5 * 60     # … up to this far ahead
MIN_TRAINING_SAMPLES     = 150        # need this many labelled rows to train
RETRAIN_INTERVAL_SECS    = 10 * 60   # retrain at most every 10 min
STRESS_THRESHOLD         = 50.0       # what counts as "stressed"
MODEL_SAVE_PATH          = pathlib.Path(__file__).parent / ".stress_model.pkl"

# Intervention bands  { level: (lo, hi, name) }
INTERVENTION_LEVELS = {
    0: (0.00, 0.30, "none"),
    1: (0.30, 0.50, "subtle"),
    2: (0.50, 0.70, "gentle"),
    3: (0.70, 1.01, "proactive"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Input-metrics tracker (mouse clicks & movement)
# ═══════════════════════════════════════════════════════════════════════════════

class InputMetrics:
    """Low-overhead rolling tracker for mouse behaviour."""

    def __init__(self):
        self._click_times: deque = deque(maxlen=600)
        self._mouse_pos: deque  = deque(maxlen=600)   # (t, x, y)

    # ── recording ────────────────────────────────────────────────────────

    def record_click(self):
        self._click_times.append(time.monotonic())

    def record_mouse_move(self, x: float, y: float):
        self._mouse_pos.append((time.monotonic(), x, y))

    # ── derived features ─────────────────────────────────────────────────

    def clicks_per_minute(self, window_secs: float = 60.0) -> float:
        cutoff = time.monotonic() - window_secs
        n = sum(1 for t in self._click_times if t > cutoff)
        return n * (60.0 / window_secs)

    def click_acceleration(self, window_secs: float = 60.0) -> float:
        """Δ click-rate between first and second half of the window."""
        now = time.monotonic()
        half = window_secs / 2
        first  = sum(1 for t in self._click_times
                     if now - window_secs < t <= now - half)
        second = sum(1 for t in self._click_times if t > now - half)
        r1 = first  * (60.0 / half)
        r2 = second * (60.0 / half)
        return r2 - r1                    # positive → accelerating

    def mouse_velocity(self, window_secs: float = 60.0) -> float:
        """Average speed in pixels/sec over the window."""
        cutoff = time.monotonic() - window_secs
        pts = [(t, x, y) for t, x, y in self._mouse_pos if t > cutoff]
        if len(pts) < 2:
            return 0.0
        total_dist = sum(
            math.hypot(pts[i][1] - pts[i-1][1], pts[i][2] - pts[i-1][2])
            for i in range(1, len(pts))
        )
        elapsed = pts[-1][0] - pts[0][0]
        return total_dist / elapsed if elapsed > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Predictive stress model
# ═══════════════════════════════════════════════════════════════════════════════

class StressPredictor:
    """
    Self-supervised predictive stress model.

    Feature vector (12-D):
        0  hour_sin            sin(2π · hour / 24)
        1  hour_cos            cos(2π · hour / 24)
        2  session_min         minutes since session started
        3  stress_mean_5m      mean stress (5-min window)
        4  stress_std_5m       std-dev of stress
        5  stress_max_5m       peak stress
        6  stress_trend_5m     linear slope of stress
        7  stress_spikes_30m   # of spikes above threshold (30 min)
        8  click_rate          clicks / minute (1-min window)
        9  click_accel         Δ click-rate between window halves
       10  mouse_velocity      avg px / sec (1-min window)
       11  stress_ratio_high   fraction of 5-min window above threshold

    Label (auto-generated):
        1 if stress > STRESS_THRESHOLD at any point in
        [T + 2 min, T + 5 min], else 0.
    """

    FEATURE_NAMES = [
        "hour_sin", "hour_cos", "session_min",
        "stress_mean", "stress_std", "stress_max", "stress_trend",
        "stress_spikes_30m",
        "click_rate", "click_accel", "mouse_velocity",
        "stress_ratio_high",
    ]

    def __init__(self, session_start: float | None = None):
        self._session_start = session_start or time.monotonic()
        self._stress_history: deque = deque(maxlen=7200)  # (t, score)
        self._input = InputMetrics()

        # Auto-label pipeline
        self._unlabeled: deque = deque(maxlen=3000)       # (t, features)
        self._X: list = []
        self._y: list = []

        # Model state
        self._model = None
        self._scaler = None
        self._last_train_time = 0.0
        self._is_trained = False

        # Heuristic weights (used before ML model is ready)
        self._hw = np.array([
            0.00, 0.00,         # time (neutral for heuristic)
            0.05,               # session length
            0.30,               # stress mean ★
            0.08,               # stress std
            0.12,               # stress max
            0.22,               # stress trend ★★
            0.05,               # spikes
            0.04,               # click rate
            0.03,               # click accel
            0.02,               # mouse velocity
            0.18,               # ratio above threshold ★
        ])

        self._try_load_model()
        self._mouse_listener = None
        if _HAS_PYNPUT:
            self._start_mouse_listener()

    # ── Recording API (called from the GUI) ──────────────────────────────

    def record_stress(self, score: float):
        """Feed a new stress score from the AU model."""
        self._stress_history.append((time.monotonic(), score))

    def record_click(self):
        self._input.record_click()

    def record_mouse_move(self, x: float, y: float):
        self._input.record_mouse_move(x, y)

    # ── Feature extraction ───────────────────────────────────────────────

    def _extract_features(self) -> np.ndarray:
        from datetime import datetime as _dt
        now = time.monotonic()
        dt_now = _dt.now()
        hour = dt_now.hour + dt_now.minute / 60.0

        # (0-1)  Time
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # (2)  Session length
        session_min = (now - self._session_start) / 60.0

        # (3-6, 11)  Stress rolling stats
        cutoff = now - LOOKBACK_WINDOW_SECS
        recent = np.array([s for t, s in self._stress_history if t > cutoff],
                          dtype=np.float64)

        if recent.size >= 1:
            stress_mean = float(recent.mean())
            stress_std  = float(recent.std())
            stress_max  = float(recent.max())
            ratio_high  = float((recent > STRESS_THRESHOLD).mean())
            if recent.size >= 3:
                x = np.arange(recent.size, dtype=np.float64)
                x -= x.mean()
                denom = (x ** 2).sum()
                stress_trend = (float((x * (recent - recent.mean())).sum()
                                      / denom) if denom > 0 else 0.0)
            else:
                stress_trend = 0.0
        else:
            stress_mean = stress_std = stress_max = 0.0
            stress_trend = ratio_high = 0.0

        # (7)  Spike count in last 30 min
        cutoff_30 = now - 30 * 60
        spikes_30 = 0
        prev_above = False
        for t, s in self._stress_history:
            if t < cutoff_30:
                continue
            above = s >= STRESS_THRESHOLD
            if above and not prev_above:
                spikes_30 += 1
            prev_above = above

        # (8-10)  Input behaviour
        click_rate  = self._input.clicks_per_minute()
        click_accel = self._input.click_acceleration()
        mouse_vel   = self._input.mouse_velocity()

        return np.array([
            hour_sin, hour_cos, session_min,
            stress_mean, stress_std, stress_max, stress_trend,
            float(spikes_30),
            click_rate, click_accel, mouse_vel,
            ratio_high,
        ], dtype=np.float64)

    # ── Core: collect → label → (re)train → predict ─────────────────────

    def collect_and_predict(self):
        """
        Snapshot features, auto-label old data, retrain if due,
        and return the current prediction.

        Returns
        -------
        prob : float
            Probability of stress in the next 2–5 minutes (0–1).
        level : int
            Intervention level (0 = none … 3 = proactive).
        level_name : str
            Human-readable intervention name.
        """
        features = self._extract_features()
        now = time.monotonic()

        self._unlabeled.append((now, features))
        self._auto_label(now)

        if (len(self._X) >= MIN_TRAINING_SAMPLES
                and now - self._last_train_time > RETRAIN_INTERVAL_SECS):
            self._train()

        prob = self._predict(features)
        level, name = self._intervention_level(prob)
        return prob, level, name

    # ── Prediction ───────────────────────────────────────────────────────

    def _predict(self, features: np.ndarray) -> float:
        if self._is_trained and self._model is not None:
            X = self._scaler.transform(features.reshape(1, -1))
            return float(self._model.predict_proba(X)[0, 1])
        return self._heuristic_predict(features)

    def _heuristic_predict(self, features: np.ndarray) -> float:
        """Rule-based fallback before enough data is collected."""
        norms = np.array([
            1.0, 1.0,                  # sin/cos already [-1, 1]
            180.0,                      # session up to 3 h
            100.0, 30.0, 100.0, 5.0,   # stress stats
            10.0,                       # spikes
            60.0, 20.0, 2000.0,         # input
            1.0,                        # ratio already [0, 1]
        ])
        normed = np.clip(np.abs(features) / (norms + 1e-9), 0, 1)
        return float(np.clip(np.dot(self._hw, normed), 0, 1))

    @staticmethod
    def _intervention_level(prob: float):
        for lvl, (lo, hi, name) in INTERVENTION_LEVELS.items():
            if lo <= prob < hi:
                return lvl, name
        return 0, "none"

    # ── Auto-labelling ───────────────────────────────────────────────────

    def _auto_label(self, now: float):
        """
        Retrospectively label old snapshots.
        A snapshot at time T is positive if stress exceeded the threshold
        at any point in [T + 2 min, T + 5 min].
        """
        remaining: deque = deque()
        for ts, feat in self._unlabeled:
            if now - ts < PREDICTION_HORIZON_MAX_S + 10:
                remaining.append((ts, feat))
                continue
            win_lo = ts + PREDICTION_HORIZON_MIN_S
            win_hi = ts + PREDICTION_HORIZON_MAX_S
            stressed = any(s >= STRESS_THRESHOLD
                           for t, s in self._stress_history
                           if win_lo <= t <= win_hi)
            self._X.append(feat)
            self._y.append(1 if stressed else 0)
        self._unlabeled = remaining

    # ── Training ─────────────────────────────────────────────────────────

    def _train(self):
        if not _HAS_SKLEARN or len(self._X) < MIN_TRAINING_SAMPLES:
            return

        X = np.array(self._X)
        y = np.array(self._y)
        if len(np.unique(y)) < 2:
            return  # need both classes

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self._model = GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.75, min_samples_leaf=10, random_state=42)
        self._model.fit(Xs, y)
        self._is_trained = True
        self._last_train_time = time.monotonic()
        self._save_model()

        imp = dict(zip(self.FEATURE_NAMES, self._model.feature_importances_))
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
        pos = int(y.sum())
        print(f"[StressPredictor] Trained on {len(X)} samples "
              f"({pos} pos / {len(X) - pos} neg).  "
              f"Top features: {top}")

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_model(self):
        try:
            data = {
                "model": self._model,
                "scaler": self._scaler,
                "X": self._X[-500:],
                "y": self._y[-500:],
            }
            with open(MODEL_SAVE_PATH, "wb") as f:
                pickle.dump(data, f)
        except Exception as exc:
            print(f"[StressPredictor] save failed: {exc}")

    def _try_load_model(self):
        if not _HAS_SKLEARN or not MODEL_SAVE_PATH.exists():
            return
        try:
            with open(MODEL_SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self._model  = data["model"]
            self._scaler = data["scaler"]
            self._X = list(data.get("X", []))
            self._y = list(data.get("y", []))
            self._is_trained = True
            print(f"[StressPredictor] Loaded model "
                  f"({len(self._X)} prior samples)")
        except Exception as exc:
            print(f"[StressPredictor] load failed: {exc}")

    # ── System-wide mouse listener (optional) ────────────────────────────

    def _start_mouse_listener(self):
        def _on_click(_x, _y, _btn, pressed):
            if pressed:
                self._input.record_click()

        def _on_move(x, y):
            self._input.record_mouse_move(x, y)

        self._mouse_listener = pynput_mouse.Listener(
            on_click=_on_click, on_move=_on_move)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

    # ── Shutdown ─────────────────────────────────────────────────────────

    def stop(self):
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._is_trained:
            self._save_model()

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        return {
            "is_trained":       self._is_trained,
            "training_samples": len(self._X),
            "pending_unlabeled": len(self._unlabeled),
            "stress_history":   len(self._stress_history),
            "positive_rate":    (sum(self._y) / len(self._y)
                                 if self._y else 0.0),
            "has_sklearn":      _HAS_SKLEARN,
            "has_pynput":       _HAS_PYNPUT,
        }
