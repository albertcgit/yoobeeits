"""
traffic_env.py
A simplified single-intersection traffic simulation.

Two conflicting traffic streams:
  - NS (North-South)
  - EW (East-West)

Cars arrive at each approach following a Poisson process. The
time-varying arrival RATE is not made up — its SHAPE comes from
analysis/generate_demand_profile.py, which used the project's trained
regression model (on the real UCI Metro Interstate Traffic Volume
dataset) to predict a realistic 24-hour traffic curve, separately for
weekday and weekend. The MAGNITUDE is rescaled to fit a single
intersection approach (see simulation/data/demand_profile.json for the
exact numbers and the documented reasoning).

The signal has 2 phases: NS_GREEN, EW_GREEN (all-red clearance implied
by a short fixed switch penalty).

This module is deliberately dependency-light (numpy + a small pre-baked
JSON file, no pandas/sklearn at runtime) so it can be reused unmodified by:
  - train.py               (offline RL training)
  - app.py                 (live dashboard simulation)
  - security/attack_simulator.py (feeds spoofed readings into the same env)
"""

import os
import json
import numpy as np

NS_GREEN = 0
EW_GREEN = 1

MAX_QUEUE = 40          # cars queued before we clip (for state discretisation)
SWITCH_PENALTY_SEC = 3   # lost seconds every time the light changes phase
MIN_GREEN_SEC = 10        # minimum hold time before a phase can switch again
                          # (standard real-world ITS constraint, also stops
                          # the agent from flickering the light every second)

_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "data", "demand_profile.json")
_profile_cache = None


def _load_demand_profile():
    """Loads the real-data-derived demand profile once and caches it.
    Falls back to a flat, unremarkable rate if the file is somehow missing,
    so the simulation never crashes — it just loses its real-data grounding."""
    global _profile_cache
    if _profile_cache is None:
        try:
            with open(_PROFILE_PATH) as f:
                data = json.load(f)
            _profile_cache = data["normalized_arrival_rate_cars_per_sec"]
        except (FileNotFoundError, KeyError):
            flat = [0.15] * 24
            _profile_cache = {"weekday": flat, "weekend": flat}
    return _profile_cache


class IntersectionEnv:
    def __init__(self, seed=None, is_weekend=None, start_hour=None):
        self.rng = np.random.default_rng(seed)
        self.profile = _load_demand_profile()
        # if not specified, randomly pick weekday or weekend for this episode
        # (roughly matching real-world proportions: 5/7 weekday, 2/7 weekend)
        if is_weekend is None:
            is_weekend = bool(self.rng.random() < (2 / 7))
        self.is_weekend = is_weekend
        self._start_hour = start_hour  # None = fully random (used by training); a number pins the start time (used by the dashboard for a livelier demo)
        self.reset()

    def reset(self):
        self.t = 0                      # simulated seconds
        # random offset into the 24-hour cycle, so training episodes (and
        # dashboard resets) don't always start at "midnight" — samples the
        # full real demand curve across many episodes/resets.
        # If _start_hour is set (dashboard use), pin to that hour instead of
        # a fully random one — otherwise a random low-traffic hour (e.g. 3am)
        # can make the dashboard look empty for a while right after loading,
        # which is honest to the real data but a poor first impression.
        if self._start_hour is not None:
            self._day_offset_sec = int(self._start_hour * 3600)
        else:
            self._day_offset_sec = int(self.rng.integers(0, 86400))
        self.queue_ns = 0
        self.queue_ew = 0
        self.phase = NS_GREEN
        self.time_in_phase = 0
        self.total_wait = 0.0
        self.total_cars = 0
        self._discharge_carry = 0.0  # fractional cars-per-second carried between steps
        # optional live override, set externally (e.g. by app.py) to a
        # callable(direction, hour_float, is_weekend) -> multiplier (float).
        # When set, the static profile rate is scaled by this multiplier
        # instead of being used as-is — this is what lets the dashboard's
        # live weather controls actually call the trained regression model
        # on every tick, rather than the simulation only ever using the
        # pre-baked static curve. None means "no live override" (default,
        # used by train.py / attack_simulator.py / anything offline).
        self.live_weather_multiplier_fn = getattr(self, "live_weather_multiplier_fn", None)
        return self._state()

    # ---- arrival rate model -------------------------------------------------
    def _arrival_rate(self, direction):
        """cars/sec. Interpolates the real-data-derived 24-hour demand curve
        (see analysis/generate_demand_profile.py) rather than a made-up
        formula — the shape (rush hour timing, weekday vs weekend) comes
        from the project's regression model trained on real traffic data.
        If live_weather_multiplier_fn is set, the rate is further scaled by
        a real-time call into that same trained model based on the
        dashboard's current weather selection."""
        curve = self.profile["weekend" if self.is_weekend else "weekday"]
        hour_float = ((self.t + self._day_offset_sec) / 3600.0) % 24.0
        h0 = int(hour_float) % 24
        h1 = (h0 + 1) % 24
        frac = hour_float - int(hour_float)
        rate = curve[h0] * (1 - frac) + curve[h1] * frac
        if direction == "ew":
            rate *= 0.75  # slightly less cross traffic, same assumption as before
        if self.live_weather_multiplier_fn is not None:
            multiplier = self.live_weather_multiplier_fn(hour_float, self.is_weekend)
            # clip to a sane range so a pathological model output can't break
            # the simulation (e.g. send queues instantly to max every tick)
            multiplier = max(0.2, min(3.0, multiplier))
            rate *= multiplier
        return rate

    # ---- step ----------------------------------------------------------------
    def step(self, action, dt=1):
        """
        action: 0 = keep NS green, 1 = keep EW green  (agent chooses desired
                phase every dt seconds; if it differs from current phase the
                light switches and pays SWITCH_PENALTY_SEC of lost throughput)
        """
        requested_switch = (action != self.phase)
        switched = requested_switch and (self.time_in_phase >= MIN_GREEN_SEC)
        if switched:
            self.time_in_phase = 0
            self.phase = action
        else:
            self.time_in_phase += dt

        # arrivals
        for _ in range(dt):
            if self.rng.random() < self._arrival_rate("ns"):
                self.queue_ns = min(self.queue_ns + 1, MAX_QUEUE)
                self.total_cars += 1
            if self.rng.random() < self._arrival_rate("ew"):
                self.queue_ew = min(self.queue_ew + 1, MAX_QUEUE)
                self.total_cars += 1

        # discharge (only the green direction clears cars), minus switch penalty
        discharge_capacity = max(0, dt - (SWITCH_PENALTY_SEC if switched else 0))
        SAT_FLOW = 0.65  # cars/sec that can cross on green — sized to comfortably
        # exceed real combined peak demand (~0.56 cars/sec at 7am/4pm, per the
        # real-data-driven profile). 0.45 was tuned for the old synthetic
        # curve's brief, narrow peaks; the real curve stays elevated across
        # ~6am-6pm continuously, so a lower capacity caused permanent,
        # unrecoverable gridlock for most of the simulated day regardless of
        # controller quality — a physical capacity deficit, not a policy bug.
        # accumulate fractional discharge so sub-1-car/sec rates aren't
        # truncated to zero every tick (that was clearing 0 cars/sec always)
        self._discharge_carry += discharge_capacity * SAT_FLOW
        cleared = int(self._discharge_carry)
        self._discharge_carry -= cleared
        if self.phase == NS_GREEN:
            cleared_ns = min(cleared, self.queue_ns)
            self.queue_ns -= cleared_ns
        else:
            cleared_ew = min(cleared, self.queue_ew)
            self.queue_ew -= cleared_ew

        # waiting time accrues for every car sitting in a queue this tick
        self.total_wait += (self.queue_ns + self.queue_ew) * dt

        self.t += dt
        reward = -(self.queue_ns + self.queue_ew)  # minimise total queue
        done = False
        return self._state(), reward, done, self._info()

    def set_queue(self, direction, value):
        """Directly overwrite a queue value — used when an external
        (possibly spoofed) sensor reading is applied to the live simulation,
        as opposed to the environment's own internal arrival process."""
        value = max(0, min(MAX_QUEUE, int(value)))
        if direction == "ns":
            self.queue_ns = value
        else:
            self.queue_ew = value

    def _state(self):
        return {
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "phase": self.phase,
            "time_in_phase": self.time_in_phase,
        }

    def _info(self):
        avg_wait = self.total_wait / max(1, self.total_cars)
        return {
            "t": self.t,
            "total_cars": self.total_cars,
            "avg_wait_per_car": avg_wait,
            "total_queue": self.queue_ns + self.queue_ew,
        }

    def discretized_state(self):
        """Bucket queues into 0-4 bins for the Q-table (5x5x2x3 = 150 states)."""
        def bucket(q):
            return min(4, q // 4)
        tip_bucket = min(2, self.time_in_phase // 15)
        return (bucket(self.queue_ns), bucket(self.queue_ew), self.phase, tip_bucket)
