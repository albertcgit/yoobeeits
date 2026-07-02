"""
fixed_timer_baseline.py
A conventional fixed-time signal controller (30s NS / 30s EW), used as the
efficiency baseline that the AI controller is measured against. This mirrors
what most real-world uncoordinated intersections still run today.
"""

NS_GREEN = 0
EW_GREEN = 1


class FixedTimerController:
    def __init__(self, ns_seconds=30, ew_seconds=30):
        self.ns_seconds = ns_seconds
        self.ew_seconds = ew_seconds

    def act(self, state, greedy=True):
        phase = state["phase"]
        tip = state["time_in_phase"]
        limit = self.ns_seconds if phase == NS_GREEN else self.ew_seconds
        if tip >= limit:
            return 1 - phase  # switch
        return phase  # keep current phase
