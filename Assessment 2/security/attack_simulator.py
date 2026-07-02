"""
security/attack_simulator.py
Standalone test/demo script: sends a mix of normal and spoofed/malicious
"sensor readings" through ReadingValidator and reports how many attacks
were caught, split by which layer caught them. This is what backs the
report's Security Assessment section and the live dashboard's attack log.

Run directly:  python3 -m security.attack_simulator
"""

import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from security.validator import ReadingValidator

random.seed(7)


def normal_reading(prev):
    # small random walk, like real traffic
    return max(0, min(40, prev + random.randint(-2, 2)))


def generate_attack(kind, prev):
    if kind == "spike":
        return random.choice([-999, 999, 5000])          # wildly out of range
    if kind == "jump":
        return max(0, min(40, prev + random.choice([-30, 30])))  # implausible jump, in-range
    if kind == "flood":
        return prev                                        # same value, sent too fast
    if kind == "stealth":
        # deliberately designed to slip past the rule layer: in-range,
        # within the max-jump limit, arrives at normal cadence — but sits
        # persistently at the edge of plausibility, which is exactly the
        # kind of pattern a fixed threshold can't see and Isolation Forest can.
        return max(0, min(40, prev + random.choice([-9, 9])))
    raise ValueError(kind)


def run_demo(n_normal=250, n_attacks=60):
    validator = ReadingValidator()
    prev_ns = 5
    caught, missed = 0, 0
    breakdown = {"rule": 0, "ml": 0, "missed": 0}
    sim_clock = 0.0
    POLL_INTERVAL = 1.0  # simulated seconds between sensor readings

    # warm up with normal traffic so the ML layer has something to learn from
    events = []
    for i in range(n_normal):
        prev_ns = normal_reading(prev_ns)
        sim_clock += POLL_INTERVAL
        events.append(("normal", prev_ns, sim_clock))

    attack_kinds = ["spike", "jump", "flood", "stealth"]
    for i in range(n_attacks):
        kind = random.choice(attack_kinds)
        val = generate_attack(kind, prev_ns)
        # flood attacks specifically arrive faster than the real poll interval
        sim_clock += 0.1 if kind == "flood" else POLL_INTERVAL
        events.append((f"attack:{kind}", val, sim_clock))

    tail = events[n_normal:]
    random.shuffle(tail)  # interleave attacks realistically
    events = events[:n_normal] + tail

    results = []
    for label, val, ts in events:
        accepted, entry = validator.validate("ns", val, now=ts)
        is_attack = label.startswith("attack")
        results.append((label, val, accepted, entry["rule_reason"], entry["ml_reason"]))
        if is_attack:
            if not accepted:
                caught += 1
                if "flagged as statistical anomaly" in entry["ml_reason"]:
                    breakdown["ml"] += 1
                else:
                    breakdown["rule"] += 1
            else:
                missed += 1
                breakdown["missed"] += 1

    print(f"Attacks attempted: {n_attacks}")
    print(f"Caught: {caught}  ({breakdown['rule']} by rule layer, {breakdown['ml']} by ML layer)")
    print(f"Missed (accepted despite being an attack): {missed}")
    print(f"Detection rate: {100*caught/n_attacks:.1f}%")

    print("\nSample of missed/stealth attacks (if any):")
    shown = 0
    for label, val, accepted, rr, mr in results:
        if label.startswith("attack") and accepted and shown < 5:
            print(f"  {label:16s} value={val:<6} accepted={accepted} rule='{rr}' ml='{mr}'")
            shown += 1
    if shown == 0:
        print("  (none)")

    return {
        "n_attacks": n_attacks, "caught": caught, "missed": missed,
        "breakdown": breakdown, "detection_rate_pct": 100 * caught / n_attacks,
    }


if __name__ == "__main__":
    run_demo()
