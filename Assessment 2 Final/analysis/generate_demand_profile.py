"""
analysis/generate_demand_profile.py

Bridges the real-dataset regression model to the traffic light simulation.

Uses the trained regression model (traffic_volume_prediction.py) to
generate a 24-hour predicted traffic volume curve, separately for a
typical weekday and a typical weekend day, holding weather at
representative "average" conditions (median temp/clouds, no rain/snow,
'Clouds' — the single most common weather_main in the real data).

The output is the *shape* of real demand by hour, not the literal I-94
volume — an interstate highway carries a different order of magnitude of
traffic than a single intersection approach. simulation/traffic_env.py
normalizes this shape onto its own arrival-rate scale. This is documented
explicitly so the distinction between "real pattern" and "scaled magnitude"
is honest and clear.

Run:  python3 -m analysis.generate_demand_profile
(Only needs to be run once — its output is committed to
 simulation/data/demand_profile.json and loaded from there at runtime,
 so the live simulation doesn't need pandas/sklearn/xgboost as a dependency.)
"""

import os
import json
import pandas as pd
import numpy as np

from analysis.traffic_volume_prediction import load_and_engineer_features, build_pipeline
from analysis.model_loader import BASELINE_WEATHER
import xgboost as xgb

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(os.path.dirname(HERE), "simulation", "data", "demand_profile.json")


def main():
    X, y, df = load_and_engineer_features()

    # train the best model (XGBoost, per traffic_volume_prediction.py results) on ALL data
    # — this profile generator wants the best possible curve, not a held-out test split
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42)
    pipeline = build_pipeline(model)
    pipeline.fit(X, y)

    # representative "average" weather conditions, held constant across all 24 hours
    # so the ONLY thing varying is time — isolating the real hour-of-day / weekday effect.
    # Uses the SAME preset as analysis/model_loader.py's BASELINE_WEATHER, so the live
    # dashboard's weather multiplier (predicted / baseline) is mathematically consistent
    # with how this static profile was generated in the first place.
    representative_weather = BASELINE_WEATHER

    profiles = {}
    for label, day_of_week, is_weekend, month in [
        ("weekday", 2, 0, 6),   # a typical Wednesday in June
        ("weekend", 5, 1, 6),   # a typical Saturday in June
    ]:
        rows = []
        for hour in range(24):
            rows.append({
                "hour": hour, "day_of_week": day_of_week, "month": month,
                "is_weekend": is_weekend, "is_holiday": 0,
                **representative_weather,
            })
        X_pred = pd.DataFrame(rows)
        predicted_volume = pipeline.predict(X_pred)
        profiles[label] = [float(v) for v in predicted_volume]
        print(f"{label}: " + ", ".join(f"{h}h={v:.0f}" for h, v in enumerate(predicted_volume)))

    # normalize the SHAPE onto the intersection's arrival-rate scale.
    # This is a deliberate, documented choice: I-94 carries thousands of
    # cars/hour, a single intersection approach carries a small fraction of
    # that — so we preserve the relative pattern (peaks, troughs, weekday
    # vs weekend difference) and rescale the magnitude to values the
    # existing intersection simulation was tuned around (see
    # simulation/traffic_env.py comments for the target range).
    all_vals = profiles["weekday"] + profiles["weekend"]
    lo, hi = min(all_vals), max(all_vals)
    TARGET_MIN, TARGET_MAX = 0.03, 0.32  # cars/sec, matches the simulation's original tuned range

    def normalize(v):
        return TARGET_MIN + (v - lo) / (hi - lo) * (TARGET_MAX - TARGET_MIN)

    normalized = {
        label: [round(normalize(v), 4) for v in values]
        for label, values in profiles.items()
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "24-hour arrival-rate profile derived from the trained XGBoost "
                "regression model on the real UCI Metro Interstate Traffic Volume "
                "dataset. SHAPE is real (relative hour-of-day and weekday/weekend "
                "pattern from real traffic behaviour); MAGNITUDE is rescaled to "
                "0.03-0.32 cars/sec to fit a single intersection approach, since "
                "the source data is an interstate highway at a much larger scale."
            ),
            "raw_predicted_volume": profiles,
            "normalized_arrival_rate_cars_per_sec": normalized,
        }, f, indent=2)

    print(f"\nSaved normalized demand profile to {OUT_PATH}")
    return normalized


if __name__ == "__main__":
    main()
