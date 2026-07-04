"""
security/validator.py
Defence-in-depth validation for incoming "sensor" readings (simulated
vehicle-count updates sent to the controller).

Layer 1 - rule-based checks (fast, explainable, no training needed):
    - value out of physically possible range
    - implausible jump between consecutive readings from the same sensor
    - readings arriving faster than the sensor's real polling interval

Layer 2 - ML anomaly detection (Isolation Forest, scikit-learn):
    trained on a window of accepted "normal" readings, flags statistically
    unusual patterns that a fixed threshold wouldn't catch (e.g. subtly
    manipulated but individually-plausible values). This is the project's
    second AI/ML component, applied to the security/assessment criterion
    rather than the control criterion.
"""

import time
import numpy as np
from sklearn.ensemble import IsolationForest

MAX_PLAUSIBLE_COUNT = 40      # matches MAX_QUEUE in traffic_env
MAX_JUMP_PER_READING = 15     # a queue can't realistically jump by more than this in one tick
MIN_POLL_INTERVAL_SEC = 0.5   # readings faster than this from one sensor are suspicious


class ReadingValidator:
    def __init__(self, ml_train_window=200, contamination=0.05):
        self.last_reading = {}       # sensor_id -> (value, timestamp)
        self.history = {"ns": [], "ew": []}
        self.ml_train_window = ml_train_window
        self.contamination = contamination
        self.models = {"ns": None, "ew": None}
        self.log = []                # audit log of every decision, for the report/demo

    # ---- rule-based layer ----------------------------------------------------
    def _rule_check(self, sensor_id, value, now):
        if not (0 <= value <= MAX_PLAUSIBLE_COUNT):
            return False, f"out-of-range value ({value})"

        if sensor_id in self.last_reading:
            last_val, last_t = self.last_reading[sensor_id]
            if abs(value - last_val) > MAX_JUMP_PER_READING:
                return False, f"implausible jump ({last_val} -> {value})"
            if now - last_t < MIN_POLL_INTERVAL_SEC:
                return False, f"reading arrived too fast ({now - last_t:.2f}s since last)"

        return True, "ok"

    # ---- ML layer --------------------------------------------------------------
    def _maybe_train(self, sensor_id):
        hist = self.history[sensor_id]
        if len(hist) >= self.ml_train_window and (
            self.models[sensor_id] is None or len(hist) % 50 == 0
        ):
            X = np.array(hist[-self.ml_train_window:]).reshape(-1, 1)
            model = IsolationForest(
                n_estimators=100, contamination=self.contamination, random_state=42
            )
            model.fit(X)
            self.models[sensor_id] = model

    def _ml_check(self, sensor_id, value):
        model = self.models[sensor_id]
        if model is None:
            return True, "ml-not-trained-yet"
        pred = model.predict(np.array([[value]]))[0]  # 1 = normal, -1 = anomaly
        return (pred == 1), ("ok" if pred == 1 else "flagged as statistical anomaly")

    # ---- public API --------------------------------------------------------------
    def validate(self, sensor_id, value, now=None):
        """now: optional explicit timestamp (seconds). Pass this in for
        deterministic simulations/tests; omit it for live use, where the
        wall clock is what actually matters for rate-limiting."""
        if now is None:
            now = time.time()
        rule_ok, rule_reason = self._rule_check(sensor_id, value, now)

        ml_ok, ml_reason = True, "ml-skipped (rule already rejected)"
        if rule_ok:
            ml_ok, ml_reason = self._ml_check(sensor_id, value)

        accepted = rule_ok and ml_ok
        entry = {
            "t": now, "sensor_id": sensor_id, "value": value,
            "accepted": accepted, "rule_reason": rule_reason, "ml_reason": ml_reason,
        }
        self.log.append(entry)

        if accepted:
            self.last_reading[sensor_id] = (value, now)
            self.history[sensor_id].append(value)
            self._maybe_train(sensor_id)

        return accepted, entry
