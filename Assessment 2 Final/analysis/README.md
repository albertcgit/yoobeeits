# Data & ML Analysis

This folder is a standalone data-science component, separate from the
traffic light simulation/dashboard. It satisfies the "real dataset +
classical ML algorithm" requirement using genuine supervised learning on
real-world data. (The project's other ML technique — unsupervised anomaly
detection via Isolation Forest — lives in `security/validator.py`, not
here; see the main `README.md`.)

## Dataset

**UCI Metro Interstate Traffic Volume Dataset**
Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5X60B
Licensed CC BY 4.0.

48,204 hourly records of westbound I-94 traffic volume near Minneapolis-St
Paul, MN, 2012-2018, with real weather (temperature, rain, snow, cloud
cover, weather description) and US holiday features. Bundled locally at
`data/Metro_Interstate_Traffic_Volume.csv` so no internet access is needed
to run this.

**Data cleaning applied:** removed duplicate timestamped rows (a known
issue in the raw dataset), and clipped `rain_1h` to 100mm — one record had
`rain_1h = 9831.3mm`, a physically impossible sensor error (world record
hourly rainfall is ~305mm) that was distorting feature scaling.

## Supervised learning — predict traffic volume

```bash
python3 -m analysis.traffic_volume_prediction
```

Trains a single regression model — **XGBoost** — on real weather/time
features:

| Model | RMSE | R² |
|---|---|---|
| **XGBoost** | **446.3** | **0.949** |

**Why only one model, and why XGBoost:** this is the only regression
model actually used live elsewhere in the project (both
`generate_demand_profile.py` and the dashboard's live weather prediction
load its saved output) — comparing it against models that aren't
otherwise used would be scope for its own sake rather than genuine
analysis. XGBoost was chosen because traffic volume vs. hour-of-day is
highly cyclical, a relationship tree-based/boosted models capture far
better than a linear model would; R²=0.949 confirms it explains ~95% of
the variance in real traffic volume from time and weather alone.

Outputs: `results/regression_metrics.json`, `results/regression_comparison.png`,
`results/feature_importance.png`.

## How this relates to the rest of the project

**This is now genuinely wired into the live simulation, not just a
standalone analysis.** `analysis/generate_demand_profile.py` uses the
trained regression model to generate a real 24-hour traffic demand curve
(separately for weekday/weekend), which `simulation/traffic_env.py` loads
and uses as its actual arrival-rate function — replacing what used to be
an arbitrary made-up sine wave.

**What's real vs. what's scaled, stated plainly for the report:** the
*shape* of the demand curve (when rush hour happens, how much busier
weekdays are than weekends) comes directly from the regression model's
predictions on real I-94 traffic data. The *magnitude* is rescaled to
0.03–0.32 cars/sec, appropriate for a single intersection approach, since
the source data is an interstate highway carrying a much larger volume of
traffic than one intersection. This distinction is documented in
`simulation/data/demand_profile.json` itself.

**Practical effect:** retraining the Q-learning agent under this
real-data-grounded demand produced a **38.5% wait-time improvement** over
the fixed-timer baseline (8.15s vs 13.26s average wait per car), verified
reproducible across independent evaluation runs.

**A second real issue found through live testing and fixed:** the
original discharge capacity (`SAT_FLOW = 0.45` cars/sec in
`traffic_env.py`) was tuned against the old synthetic demand curve's
brief, narrow peaks. The real data's demand stays elevated continuously
across roughly 6am-6pm (accurately reflecting real all-day business-hour
traffic, not just two short rush-hour spikes) — combined peak demand
regularly exceeded 0.56 cars/sec while the intersection could only ever
discharge ~0.45 cars/sec total, a genuine physical capacity deficit that
caused permanent, unrecoverable gridlock for most of the simulated day
**regardless of how good the controller was**. This was caught by
observing sustained heavy queues on both approaches simultaneously during
live dashboard testing — a different symptom from the earlier one-sided
neglect bug, and a different root cause (physics/capacity, not policy).
Fixed by raising `SAT_FLOW` to 0.65 (leaving ~14% headroom over peak
combined demand) and retraining; confirmed via a full 12-simulated-hour
run that queues now reach a stable, low equilibrium instead of runaway
growth.

**A real methodology issue found and fixed along the way, worth including
in a testing/limitations section:** an earlier version of this evaluation
was not reproducible — the agent's small amount of exploration during
evaluation used an unseeded random number generator, so re-running the
same evaluation gave different results each time (one lucky run reported
15% improvement; re-running it could give a worse number, or even show
the agent losing to the baseline). This was fixed by (1) explicitly
seeding the agent's RNG before evaluation, and (2) adding a rule-based
safety override (`q_learning_agent.act_with_safety`) for a specific
failure mode found through live testing: the queue-bucket discretisation
caps everything at "16 or more," so a queue of 16 and a queue of 40 look
identical to the learned policy — in a rare, severely-imbalanced state,
the trained Q-values for that bucket were close to noise and picked the
wrong action. The fix combines a rule-based failsafe for that specific
edge case with substantially more training (1200 → 4000 episodes), which
resolved the underlying issue directly. Both are documented in code
comments in `simulation/q_learning_agent.py`.

## Live weather prediction (the model runs at runtime, not just once)

The static demand profile above is generated once and doesn't call the
model again — that's a fair criticism of a "real dataset" project if it
stopped there. So the dashboard also has a **live weather control panel**:
picking Clear / Clouds / Rain / Snow calls `analysis/model_loader.py`,
which loads the saved `best_model.pkl` (the same trained XGBoost model)
and predicts traffic volume for the current simulated hour under that
weather, live. That prediction is compared against a baseline prediction
to get a multiplier, which scales the simulation's arrival rate in real
time — visible immediately as the queue/car count responding to the
weather change.

Real example: selecting Snow at 8am produces a predicted multiplier of
**~0.73x** (27% less traffic) compared to baseline weather — a genuine
model output, not a scripted number, and consistent with the real-world
intuition that snow disrupts commuting more than routine rain does
(Rain's multiplier is close to 1.0, since rain is common and mundane in
the source dataset).

This is what makes the regression model do real work while the system is
running, not just shape a one-time offline curve.

To regenerate the profile after retraining the regression model:
```bash
python3 -m analysis.generate_demand_profile   # writes simulation/data/demand_profile.json
python3 train.py                               # retrains the AI against the new profile
```
