"""
app.py
Live dashboard for the AI Adaptive Traffic Light Optimization System.

- Runs the intersection simulation in a background thread, stepping once
  per (scaled) second, using either the trained Q-learning agent or the
  fixed-timer baseline (switchable live from the UI for side-by-side demo).
- Pushes state updates to the browser over WebSockets (Flask-SocketIO).
- Exposes a REST endpoint that lets the demo "attacker" inject spoofed
  sensor readings, which flow through the same ReadingValidator used in
  security/attack_simulator.py, with results shown live in the UI.

Run:  python3 app.py   then open http://localhost:5000
"""

import os, sys, time, threading, random
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from simulation.traffic_env import IntersectionEnv, NS_GREEN, EW_GREEN
from simulation.q_learning_agent import QLearningAgent
from simulation.fixed_timer_baseline import FixedTimerController
from security.validator import ReadingValidator

app = Flask(__name__)
app.config["SECRET_KEY"] = "its-assessment-2-demo"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.pkl")

# ---- shared simulation state (single demo intersection, single client focus) ----
state_lock = threading.Lock()
sim = {
    "env": IntersectionEnv(seed=random.randint(0, 100000)),
    "controller_name": "ai",   # "ai" or "fixed"
    "running": True,
    "speed": 1,                 # simulated seconds per tick
    "security_enabled": True,   # gates whether injected readings are validated before being applied
}
agent = QLearningAgent()
if os.path.exists(Q_TABLE_PATH):
    agent.load(Q_TABLE_PATH)
    print(f"Loaded trained Q-table from {Q_TABLE_PATH}")
else:
    print("WARNING: no trained q_table.pkl found — run train.py first. "
          "Falling back to an untrained (random-ish) agent.")
baseline_ctrl = FixedTimerController()

validator = ReadingValidator()
security_log = []  # most recent events, newest first, for the UI feed
SECURITY_LOG_MAX = 40


def get_controller():
    return agent if sim["controller_name"] == "ai" else baseline_ctrl


def sim_loop():
    """Background thread: advances the simulation and emits state every tick."""
    while True:
        with state_lock:
            if sim["running"]:
                env = sim["env"]
                controller = get_controller()
                if controller is agent:
                    d_state = env.discretized_state()
                    action = agent.act(d_state, greedy=True)
                else:
                    action = controller.act(env._state())
                state, reward, done, info = env.step(action, dt=sim["speed"])

                payload = {
                    "queue_ns": state["queue_ns"],
                    "queue_ew": state["queue_ew"],
                    "phase": "NS_GREEN" if state["phase"] == NS_GREEN else "EW_GREEN",
                    "time_in_phase": state["time_in_phase"],
                    "t": info["t"],
                    "avg_wait_per_car": round(info["avg_wait_per_car"], 1),
                    "total_cars": info["total_cars"],
                    "controller": sim["controller_name"],
                }
            else:
                payload = None
        if payload:
            socketio.emit("sim_update", payload)
        time.sleep(0.5)  # 2 UI updates/sec regardless of simulated speed


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/set_controller", methods=["POST"])
def set_controller():
    name = request.json.get("controller")
    if name not in ("ai", "fixed"):
        return jsonify({"error": "invalid controller"}), 400
    with state_lock:
        sim["controller_name"] = name
    return jsonify({"ok": True, "controller": name})


@app.route("/api/reset", methods=["POST"])
def reset_sim():
    with state_lock:
        sim["env"] = IntersectionEnv(seed=random.randint(0, 100000))
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def pause_sim():
    with state_lock:
        sim["running"] = not sim["running"]
        running = sim["running"]
    return jsonify({"ok": True, "running": running})


@app.route("/api/toggle_security", methods=["POST"])
def toggle_security():
    with state_lock:
        sim["security_enabled"] = not sim["security_enabled"]
        enabled = sim["security_enabled"]
    return jsonify({"ok": True, "security_enabled": enabled})


@app.route("/api/inject_reading", methods=["POST"])
def inject_reading():
    """Demo 'attacker' endpoint — sends a (possibly spoofed) sensor reading.

    If security is ON: the reading is validated first (rule layer + Isolation
    Forest). Only accepted readings are applied to the live simulation's
    queue state, so the controller only ever acts on trusted data.

    If security is OFF: the reading is applied directly to the live
    simulation, unvalidated — this is the "vulnerable" path, showing what an
    attacker could actually do to the traffic light if there were no
    validation layer in front of the sensor feed.
    """
    data = request.json
    sensor_id = data.get("sensor_id", "ns")
    value = data.get("value")
    label = data.get("label", "manual")
    if value is None:
        return jsonify({"error": "value required"}), 400

    with state_lock:
        security_enabled = sim["security_enabled"]

    if security_enabled:
        accepted, entry = validator.validate(sensor_id, value)
    else:
        accepted, entry = True, {
            "rule_reason": "security layer disabled — reading not checked",
            "ml_reason": "security layer disabled — reading not checked",
        }

    applied = False
    if accepted:
        with state_lock:
            sim["env"].set_queue(sensor_id, value)
            applied = True

    log_entry = {
        "label": label, "sensor_id": sensor_id, "value": value,
        "accepted": accepted, "applied": applied,
        "security_enabled": security_enabled,
        "rule_reason": entry["rule_reason"],
        "ml_reason": entry["ml_reason"], "t": time.time(),
    }
    security_log.insert(0, log_entry)
    del security_log[SECURITY_LOG_MAX:]
    socketio.emit("security_event", log_entry)
    return jsonify({"ok": True, "accepted": accepted, "applied": applied, "entry": entry})


@app.route("/api/security_log")
def get_security_log():
    return jsonify(security_log)


@app.route("/api/clear_security_log", methods=["POST"])
def clear_security_log():
    security_log.clear()
    socketio.emit("security_log_cleared", {})
    return jsonify({"ok": True})


@app.route("/api/metrics")
def get_metrics():
    import json
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "no metrics.json yet — run train.py"}), 404


if __name__ == "__main__":
    t = threading.Thread(target=sim_loop, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
