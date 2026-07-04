# AI Adaptive Traffic Light Optimization System
MSE806 Assessment 2 — quick start for the team.

## What this is
A simulated intersection where a Q-learning AI learns to switch the traffic
light based on live queue lengths, compared against a conventional
fixed-timer light. A security layer checks incoming sensor readings for
spoofing/tampering before they're allowed to affect the light.

## 1. Setup (one time)
```bash
pip install -r requirements.txt
```
The `results` folder (with `q_table.pkl`, `metrics.json`, and the
efficiency chart) is already included, so you don't need to run
`train.py` — go straight to step 2.

If `results` is ever missing or you want to retrain from scratch:
```bash
python train.py
```

## 2. Run it
```bash
python app.py
```
Open **http://localhost:5000** in your browser. Leave the terminal open —
that's the server running.

## 3. What you're looking at

| Panel | What it shows |
|---|---|
| Live Intersection | Small blue rectangles = cars waiting. Green light = right of way, red = stopped. |
| Security Layer | Buttons to send fake sensor readings ("attacks") and see whether the system catches them. |
| Efficiency Analysis | How much better the AI is than a fixed-timer light (currently ~38.5%, verified reproducible). |

## 4. Six things to try, in order

1. **Watch it run for 30 seconds.** Cars build up, the light switches, cars clear. No refresh needed.
2. **Click "Fixed-Timer Baseline"**, watch 30s, then click **"AI Controller"** to switch back. Notice the fixed-timer switches on a strict clock; the AI reacts to which side has more cars.
3. **Click "Attack: Spike" under NS sensor.** Log should say **BLOCKED**.
4. **Toggle "Security: ON" to OFF**, click the same attack again. Log now says **APPLIED TO TRAFFIC** and the NS queue jumps — this shows what an attacker could do without the security layer. Turn Security back ON afterward.
5. **Click "Attack: Stealth" 10-15 times in a row.** Early clicks get accepted; eventually some get blocked as a "statistical anomaly" once the ML layer has learned enough to catch it.
6. **Click "❄️ Snow" in the Live Weather Prediction panel**, then click back to "☁️ Clouds". Watch the "Model-Predicted Demand Multiplier" number change and the traffic visibly get lighter — this is the trained regression model making a real, live prediction, not a canned number.

## 5. If something looks broken

- **Numbers frozen, nothing updating:** hard refresh (Ctrl+F5, not a normal refresh).
- **Page won't load:** check the terminal — is `python app.py` still running?
- **"Security: ON" but a Stealth attack still shows APPLIED TO TRAFFIC:** expected, not a bug. Stealth attacks are built to pass the rule checks and can only be caught by the ML layer, which needs ~200 accepted readings before it's trained. Clicking the button a dozen times in a demo won't reach that — this is worth mentioning verbally if you present it live.
- **`XGBoostError: input stream corrupted` (or similar pickle/joblib error) on startup:** the `.pkl` model file got corrupted, almost always because Git mangled a binary file during commit/pull. Fix it by retraining locally (fast, no internet needed):
  ```bash
  python -m analysis.traffic_volume_prediction   # fixes analysis/results/best_model.pkl
  python train.py                                 # fixes results/q_table.pkl, if that one's affected instead
  ```
  **To stop this from recurring**, add a `.gitattributes` file to the repo root containing:
  ```
  *.pkl binary
  *.png binary
  ```
  and commit it — this tells Git never to touch these files' bytes.
- **Something else:** open DevTools (F12) → Console and screenshot any red errors.

## 6. Full test checklist
See `TEST_CASES.md` for the complete list of things to click through with pass/fail criteria — hand this to whoever is testing.

## Project structure
```
simulation/    - intersection simulation (arrival demand is REAL-DATA-DRIVEN, see analysis/), Q-learning AI, fixed-timer baseline
security/      - reading validator (rules + Isolation Forest) and attack test harness
analysis/      - real-dataset supervised ML (regression), generates the demand profile the simulation uses (see analysis/README.md)
train.py       - trains the AI, evaluates it, produces the efficiency numbers
app.py + templates/dashboard.html - the live dashboard (Flask + WebSockets)
```
The simulation's traffic demand isn't made up — its hourly pattern comes
from a regression model trained on a real public traffic dataset. See
`analysis/README.md` for the full explanation.

## Suggested team split
1. **Simulation & AI** — `simulation/`, `train.py`, tuning/retraining
2. **Dashboard** — `app.py`, `templates/dashboard.html`, deployment
3. **Security** — `security/`, running/extending the attack tests
