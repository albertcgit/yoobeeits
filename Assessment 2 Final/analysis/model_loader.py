"""
analysis/model_loader.py

Loads the trained regression model (saved by traffic_volume_prediction.py)
for reuse elsewhere, without needing to retrain it every time:
  - generate_demand_profile.py uses it to build the static 24-hour curve
  - app.py uses it for LIVE prediction, driven by the weather controls
    on the dashboard — this is what makes the ML component actually do
    something at runtime, not just once offline.
"""

import os
import joblib
import pandas as pd

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "results", "best_model.pkl")

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict_volume(hour, day_of_week, month, is_weekend, is_holiday,
                    temp, rain_1h, snow_1h, clouds_all, weather_main):
    """Predict traffic volume for a single set of conditions. Returns a
    plain float. This is a real call into the trained XGBoost model —
    not a lookup table."""
    model = load_model()
    row = pd.DataFrame([{
        "hour": hour, "day_of_week": day_of_week, "month": month,
        "is_weekend": is_weekend, "is_holiday": is_holiday,
        "temp": temp, "rain_1h": rain_1h, "snow_1h": snow_1h,
        "clouds_all": clouds_all, "weather_main": weather_main,
    }])
    return float(model.predict(row)[0])


# Representative weather presets a user can pick on the dashboard.
# temp is in Kelvin (matches the original dataset's units).
WEATHER_PRESETS = {
    "clear":  {"weather_main": "Clear",  "temp": 288.0, "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 5},
    "clouds": {"weather_main": "Clouds", "temp": 283.0, "rain_1h": 0.0, "snow_1h": 0.0, "clouds_all": 75},
    "rain":   {"weather_main": "Rain",   "temp": 280.0, "rain_1h": 4.0, "snow_1h": 0.0, "clouds_all": 90},
    "snow":   {"weather_main": "Snow",   "temp": 267.0, "rain_1h": 0.0, "snow_1h": 1.5, "clouds_all": 90},
}

# The "baseline" weather used when generate_demand_profile.py built the
# static curve — used as the reference point for computing a live
# weather multiplier (see app.py).
BASELINE_WEATHER = WEATHER_PRESETS["clouds"]
