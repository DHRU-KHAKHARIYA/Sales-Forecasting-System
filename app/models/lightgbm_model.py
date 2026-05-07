import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMRegressor

from app.features.pipeline import run_feature_engineering


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_4", "rolling_std_4",
    "rolling_mean_8", "rolling_std_8",
    "rolling_mean_12", "rolling_std_12",
    "day_of_week", "month", "quarter",
    "week_of_year", "is_month_start",
    "is_month_end", "is_holiday_week",
]


def train(df: pd.DataFrame, state: str) -> dict:
    state_df = df[df["State"] == state].dropna(subset=FEATURE_COLS).copy()

    X = state_df[FEATURE_COLS]
    y = state_df["Total"]

    model = LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)

    return {"model": model, "state": state}


def forecast(
    trained: dict,
    last_known: pd.DataFrame,
    n_weeks: int = 8,
) -> pd.DataFrame:
    model = trained["model"]
    state = trained["state"]
    history = last_known[last_known["State"] == state].sort_values("Date").copy()

    predictions = []
    for _ in range(n_weeks):
        featured = run_feature_engineering(history)
        last_row = featured.iloc[-1:]

        if last_row[FEATURE_COLS].isnull().any(axis=1).iloc[0]:
            last_row = last_row.fillna(0)

        pred_value = model.predict(last_row[FEATURE_COLS])[0]
        next_date = history["Date"].max() + pd.Timedelta(weeks=1)

        predictions.append({
            "Date": next_date,
            "State": state,
            "Predicted": pred_value,
        })

        new_row = pd.DataFrame([{
            "State": state,
            "Date": next_date,
            "Total": pred_value,
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def save_model(trained: dict, path: str | Path) -> None:
    joblib.dump(trained, Path(path))


def load_model(path: str | Path) -> dict:
    return joblib.load(Path(path))
