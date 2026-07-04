"""
analysis/traffic_volume_prediction.py

SUPERVISED LEARNING component using a real public dataset:
UCI Metro Interstate Traffic Volume Dataset (Hogue, 2019, CC-BY 4.0)
https://doi.org/10.24432/C5X60B

48,204 hourly records of westbound I-94 traffic volume near Minneapolis-St
Paul, MN (2012-2018), with real weather and US holiday features.

Task: predict traffic_volume (a continuous target) from time-of-day,
day-of-week, month, holiday, and weather features. This is genuine
supervised regression on real-world data — distinct from the Q-learning
controller in simulation/, which learns from a simulated reward signal
rather than a labelled dataset.

Single model: XGBoost (gradient boosting). Kept deliberately to one model
rather than a multi-model comparison — this is the only regression model
actually used live elsewhere in the project (analysis/generate_demand_profile.py
and analysis/model_loader.py both load its saved output), so a comparison
against models that aren't otherwise used would be scope for its own sake.
XGBoost was chosen because tree-based/boosted models handle the highly
cyclical, non-linear relationship between hour-of-day and traffic volume
far better than a linear model would.

Run:  python3 -m analysis.traffic_volume_prediction
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "Metro_Interstate_Traffic_Volume.csv")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_engineer_features():
    df = pd.read_csv(DATA_PATH)

    # the raw dataset has a handful of duplicate timestamped rows
    # (documented data quality issue in the original UCI dataset) — drop them
    df = df.drop_duplicates(subset="date_time")

    # data quality issue found during EDA: one record has rain_1h = 9831.3mm,
    # which is physically impossible (the real-world record for rainfall in
    # one hour is ~305mm). This is a bad sensor/logging value, not a real
    # extreme weather event — clip rain_1h to a generous but physically
    # plausible ceiling so this single row doesn't distort feature scaling.
    df["rain_1h"] = df["rain_1h"].clip(upper=100)

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Monday
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    features = [
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
        "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main",
    ]
    X = df[features]
    y = df["traffic_volume"]
    return X, y, df


def build_pipeline(model):
    categorical = ["weather_main"]
    numeric = ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
               "temp", "rain_1h", "snow_1h", "clouds_all"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def main():
    X, y, df = load_and_engineer_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset: {len(df)} rows after de-duplication")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42)
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"XGBoost  RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.3f}")

    results = {"XGBoost": {"rmse": rmse, "mae": mae, "r2": r2}}
    with open(os.path.join(RESULTS_DIR, "regression_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # retrain on ALL data (not just the 80% train split) before saving —
    # for a deployed/live-use model you want it to have seen everything
    # available, not leave 20% on the table
    final_pipeline = build_pipeline(
        xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42)
    )
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, os.path.join(RESULTS_DIR, "best_model.pkl"))
    print(f"Saved trained model to {RESULTS_DIR}/best_model.pkl")

    # --- chart: predicted vs actual ---
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, preds, s=4, alpha=0.25, color="#2E86AB")
    lims = [0, y_test.max()]
    plt.plot(lims, lims, color="#A23B72", linewidth=1)
    plt.xlabel("Actual traffic volume")
    plt.ylabel("Predicted traffic volume")
    plt.title(f"XGBoost: predicted vs actual (R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "regression_comparison.png"), dpi=150)
    print(f"Chart saved to {RESULTS_DIR}/regression_comparison.png")

    # --- chart: feature importance ---
    ohe = pipeline.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["weather_main"]))
    numeric_names = ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
                      "temp", "rain_1h", "snow_1h", "clouds_all"]
    all_names = cat_names + numeric_names
    importances = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(7, 4.5))
    plt.barh([all_names[i] for i in order][::-1], [importances[i] for i in order][::-1],
              color="#2E86AB")
    plt.xlabel("Importance")
    plt.title("Top 10 features — XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    print(f"Chart saved to {RESULTS_DIR}/feature_importance.png")

    return results


if __name__ == "__main__":
    main()
