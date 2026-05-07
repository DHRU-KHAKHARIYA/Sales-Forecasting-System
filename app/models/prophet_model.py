import pandas as pd
import joblib
from pathlib import Path
from prophet import Prophet


def train(df: pd.DataFrame, state: str) -> dict:
    state_df = (
        df[df["State"] == state]
        .sort_values("Date")[["Date", "Total"]]
        .rename(columns={"Date": "ds", "Total": "y"})
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(state_df)

    return {"model": model, "state": state}


def forecast(trained: dict, n_weeks: int = 8) -> pd.DataFrame:
    model = trained["model"]
    future = model.make_future_dataframe(periods=n_weeks, freq="W-SUN")
    pred = model.predict(future)

    forecast_df = pred.tail(n_weeks)[["ds", "yhat"]].rename(
        columns={"ds": "Date", "yhat": "Predicted"}
    )
    forecast_df["State"] = trained["state"]

    return forecast_df[["Date", "State", "Predicted"]].reset_index(drop=True)


def save_model(trained: dict, path: str | Path) -> None:
    joblib.dump(trained, Path(path))


def load_model(path: str | Path) -> dict:
    return joblib.load(Path(path))
