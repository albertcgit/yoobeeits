# User test cases — AI Adaptive Traffic Light Optimization System

All tests performed through the browser dashboard (`http://localhost:5000`). No code required. Tick off as you go.

## 1. Traffic simulation core

- [ ] **SIM-01 — Queue never goes negative.** Let the simulation run for a few minutes, click Reset a few times, click Pause then Resume, and use the NS/EW "Attack: Spike" buttons (which sometimes send -999) with Security OFF. **Pass if:** NS Queue and EW Queue are always 0 or positive — never negative, blank, or "NaN".
- [ ] **SIM-02 — Queue doesn't overflow visually.** With Security OFF, repeatedly click "Attack: Spike" on one sensor to push its value to the max. **Pass if:** the car stack caps at 12 visible cars with a "+N" label instead of spilling outside the intersection box.
- [ ] **SIM-03 — Green light actually clears cars.** Watch one direction build up a queue, then watch it get the green light. **Pass if:** that direction's queue count and car stack visibly shrink while its light is green.
- [ ] **SIM-04 — Light doesn't flicker.** Watch the light switch between NS and EW during normal running. **Pass if:** each phase holds for a noticeable stretch (several seconds) rather than switching every second.

## 2. AI controller vs. fixed-timer baseline

- [ ] **AI-01 — Controllers behave differently.** Click "AI Controller", watch for a minute, then click "Fixed-Timer Baseline" and watch for a minute. **Pass if:** the AI visibly favors whichever direction has more cars waiting, while the fixed-timer switches on a strict clock regardless of queue length.
- [ ] **AI-02 — Efficiency numbers are present.** Scroll to the "Efficiency Analysis" panel. **Pass if:** it shows real numbers (AI avg wait, fixed-timer avg wait, % improvement) rather than being stuck on "Loading...".
- [ ] **AI-03 — Switching controllers doesn't crash anything.** Toggle back and forth between AI Controller and Fixed-Timer Baseline several times quickly. **Pass if:** the dashboard keeps updating smoothly with no frozen numbers or console errors (F12 → Console).

## 3. Security — protected path (Security ON)

- [ ] **SEC-01 — Out-of-range spike is blocked.** With Security ON, click "Attack: Spike" on either sensor. **Pass if:** the log shows "BLOCKED" with a reason like "out-of-range value", and the live queue/car count does not change.
- [ ] **SEC-02 — Implausible jump is blocked.** With Security ON, click "Attack: Jump". **Pass if:** the log shows "BLOCKED" with a reason mentioning an implausible jump, and traffic is unaffected.
- [ ] **SEC-03 — Flood is blocked.** With Security ON, click "Attack: Flood" a couple of times quickly. **Pass if:** the log shows at least one "BLOCKED" entry mentioning timing/too fast.
- [ ] **SEC-04 — Stealth attack eventually caught.** With Security ON, click "Attack: Stealth" on one sensor repeatedly (10+ times). **Pass if:** early clicks may show "ACCEPTED", but after enough clicks some get "BLOCKED" with a reason mentioning "statistical anomaly" (this is the ML layer kicking in after it has enough data).
- [ ] **SEC-05 — Normal readings are accepted.** With Security ON, click "Send Normal Reading" a few times. **Pass if:** the log shows "ACCEPTED", not blocked.

## 4. Security — unprotected path (Security OFF)

- [ ] **INT-01 — Toggle changes to OFF state visibly.** Click the "Security: ON" button. **Pass if:** it visibly changes to "Security: OFF" (different color).
- [ ] **INT-02 — Unprotected attacks reach live traffic.** With Security OFF, click "Attack: Spike" on NS. **Pass if:** the log shows "APPLIED TO TRAFFIC" and the NS queue/car count visibly jumps to match the attack value.
- [ ] **INT-03 — Re-enabling security stops further attacks.** After INT-02, click the button again to turn Security back ON, then repeat the same attack. **Pass if:** the log now shows "BLOCKED" again instead of "APPLIED TO TRAFFIC".
- [ ] **INT-04 — NS and EW buttons are independent.** With Security OFF, click an attack under "NS sensor" only. **Pass if:** only the NS queue changes — EW queue is unaffected. Repeat with an "EW sensor" button and confirm the reverse.

## 5. Security log usability

- [ ] **LOG-01 — Block reason is readable without hovering.** Trigger any blocked attack. **Pass if:** you can read why it was blocked directly in the log text (e.g. "Blocked: out-of-range value (999)") without needing to hover over anything.
- [ ] **LOG-02 — Clear log works.** Generate a few log entries, then click "Clear log". **Pass if:** the log visibly empties immediately.
- [ ] **LOG-03 — Log doesn't grow forever.** Click various attack buttons 40+ times. **Pass if:** the log stays scrollable and responsive, doesn't visibly slow down the page.

## 6. General dashboard behaviour

- [ ] **UI-01 — No console errors.** Open DevTools (F12) → Console, hard refresh the page (Ctrl+F5). **Pass if:** there are no red error messages.
- [ ] **UI-02 — Live updates without refreshing.** Watch the dashboard for 30 seconds without touching anything. **Pass if:** queue counts, phase, wait time, and total cars processed all update on their own.
- [ ] **UI-03 — Pause actually pauses.** Click "Pause". **Pass if:** all numbers stop changing. Click again (should say "Resume") and confirm it continues.
- [ ] **UI-04 — Reset works.** Click "Reset". **Pass if:** queues return to 0 and traffic starts building up again from scratch.
- [ ] **UI-05 — Cars sit next to the light, not on top of it.** Let a queue build up to a few cars on both NS and EW. **Pass if:** the cars form a visible line starting right next to the stop light, with no visual overlap.
- [ ] **UI-06 — Light colors are correct.** Watch a phase change. **Pass if:** the direction with right-of-way shows green, the other shows red (not grey).

## 7. Live weather prediction

- [ ] **WX-01 — Weather buttons change the active state.** Click "❄️ Snow". **Pass if:** the button visibly highlights as active, and "Current Weather" updates to "Snow".
- [ ] **WX-02 — Multiplier updates.** After clicking a weather button, watch the "Model-Predicted Demand Multiplier" number. **Pass if:** it changes to a new value within a couple of seconds (not stuck at 1.00x).
- [ ] **WX-03 — Snow visibly reduces traffic.** Click "❄️ Snow" during a busy period (e.g. shortly after starting the dashboard). **Pass if:** the multiplier drops below 1.00x and queues/car counts noticeably lighten over the following seconds.
- [ ] **WX-04 — Clouds always shows exactly 1.00x.** Click "☁️ Clouds". **Pass if:** the multiplier reads exactly 1.00x (this is expected — Clouds is the baseline weather everything else is compared against).

## 8. Setup on a different machine

- [ ] **ENV-01 — Fresh setup works end-to-end.** On a teammate's machine (or a clean folder), run `pip install -r requirements.txt` then `python app.py` (the `results/` and `analysis/results/` folders already have trained models bundled in, so `train.py` isn't required unless those are missing). **Pass if:** it completes with no errors and the dashboard loads.
- [ ] **ENV-02 — Works without a data connection issue.** Confirm the dashboard loads fully (the browser needs one-time internet access to fetch the Socket.IO library from its CDN). **Pass if:** live updates work normally once the page has loaded.
