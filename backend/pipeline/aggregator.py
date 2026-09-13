"""
Temporal aggregation module.

Takes per-frame confidence scores and:
1. Smooths them via a rolling window (kills single-frame noise)
2. Tracks consecutive over-threshold detections (streak logic)
3. Fires alerts when the streak crosses the required length

Phase 3 will swap the simple rolling window for an LSTM / Transformer
that models the *sequence* of frame embeddings, not just the average.
"""
from collections import deque
from datetime import datetime

import config


class TemporalAggregator:
    """
    Aggregates per-frame deepfake scores over time and issues alerts
    when sustained high-confidence detections occur.
    """

    def __init__(self,
                 window_size=None,
                 alert_threshold=None,
                 consecutive_needed=None):
        self.window_size = window_size or config.WINDOW_SIZE
        self.alert_threshold = alert_threshold or config.ALERT_THRESHOLD
        self.consecutive_needed = consecutive_needed or config.CONSECUTIVE_ALERTS_NEEDED

        self.score_history = deque(maxlen=self.window_size)
        self.streak = 0
        self.alerts = []  # list of (timestamp_str, smoothed_score) tuples

    def reset(self):
        """Clear all state — call at the start of a new session."""
        self.score_history.clear()
        self.streak = 0
        self.alerts = []

    def update(self, raw_score):
        """
        Feed a new per-frame score into the aggregator.

        Args:
            raw_score: float in [0.0, 1.0] from the frame-level detector

        Returns:
            dict with keys:
              - smoothed_score: rolling-window average
              - streak: current consecutive over-threshold count
              - alert_fired: True if this update triggered a new alert
        """
        self.score_history.append(raw_score)
        smoothed = sum(self.score_history) / len(self.score_history)

        alert_fired = False
        if smoothed >= self.alert_threshold:
            self.streak += 1
            if self.streak >= self.consecutive_needed:
                self._fire_alert(smoothed)
                alert_fired = True
                self.streak = 0  # reset after firing to avoid spam
        else:
            self.streak = 0

        return {
            "smoothed_score": smoothed,
            "streak": self.streak,
            "alert_fired": alert_fired,
        }

    def _fire_alert(self, smoothed_score):
        """Log a new alert with the current timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Avoid duplicate timestamps if fired multiple times in same second
        if not self.alerts or self.alerts[-1][0] != timestamp:
            self.alerts.append((timestamp, smoothed_score))

    def recent_alerts(self, n=5):
        """Return the N most recent alerts, newest first."""
        return self.alerts[-n:][::-1]

    def is_currently_alerting(self):
        """True if the smoothed score is currently above the alert threshold."""
        if not self.score_history:
            return False
        smoothed = sum(self.score_history) / len(self.score_history)
        return smoothed >= self.alert_threshold
    