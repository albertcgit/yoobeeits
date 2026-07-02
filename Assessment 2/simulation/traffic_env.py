"""
traffic_env.py
A simplified single-intersection traffic simulation.

Two conflicting traffic streams:
  - NS (North-South)
  - EW (East-West)

Cars arrive at each approach following a Poisson process (independent,
time-varying arrival rate so we get rush-hour-like patterns).

The signal has 2 phases: NS_GREEN, EW_GREEN (all-red clearance implied
by a short fixed switch penalty).

This module is deliberately dependency-light (numpy only) so it can be
reused unmodified by:
  - train.py               (offline RL training)
  - app.py                 (live dashboard simulation)
  - security/attack_simulator.py (feeds spoofed readings into the same env)
"""

import numpy as np

NS_GREEN = 0
EW_GREEN = 1

MAX_QUEUE = 40          # cars queued before we clip (for state discretisation)
SWITCH_PENALTY_SEC = 3   # lost seconds every time the light changes phase
MIN_GREEN_SEC = 10        # minimum hold time before a phase can switch again
                          # (standard real-world ITS constraint, also stops
                          # the agent from flickering the light every second)


class IntersectionEnv:
    def __init__(self, seed=None, arrival_profile="rush_hour"):
        self.rng = np.random.default_rng(seed)
        self.arrival_profile = arrival_profile
        self.reset()

    def reset(self):
        self.t = 0                      # simulated seconds
        self.queue_ns = 0
        self.queue_ew = 0
        self.phase = NS_GREEN
        self.time_in_phase = 0
        self.total_wait = 0.0
        self.total_cars = 0
        self._discharge_carry = 0.0  # fractional cars-per-second carried between steps
        return self._state()

    # ---- arrival rate model -------------------------------------------------
    def _arrival_rate(self, direction):
        """cars/sec, varies over the simulated day to mimic rush hour."""
        minute_of_day = (self.t // 60) % 120  # 2-hour cycle for demo purposes
        base = 0.06
        # two peaks to look like AM/PM rush
        peak = 0.22 * np.exp(-((minute_of_day - 30) ** 2) / (2 * 12 ** 2))
        peak += 0.22 * np.exp(-((minute_of_day - 90) ** 2) / (2 * 12 ** 2))
        rate = base + peak
        if direction == "ew":
            rate *= 0.75  # slightly less cross traffic
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
        SAT_FLOW = 0.45  # cars/sec that can cross on green
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
