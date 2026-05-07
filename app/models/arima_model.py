import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train(df: pd.DataFrame, state: str) -> dict:
    series = (
        df[df["State"] == state]
        .sort_values("Date")
        .set_index("Date")["Total"]
    )
    series.index = pd.DatetimeIndex(series.index, freq="W-SUN")

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)

    return {"model": fitted, "state": state}


def forecast(trained: dict, n_weeks: int = 8) -> pd.DataFrame:
    fitted = trained["model"]
    pred = fitted.forecast(steps=n_weeks)

    return pd.DataFrame({
        "Date": pred.index,
        "State": trained["state"],
        "Predicted": pred.values,
    })


def save_model(trained: dict, path: str | Path) -> None:
    joblib.dump(trained, Path(path))


def load_model(path: str | Path) -> dict:
    return joblib.load(Path(path))
