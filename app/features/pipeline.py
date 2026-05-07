import pandas as pd

from app.features.lag_features import add_lag_features
from app.features.rolling_features import add_rolling_features
from app.features.calendar_features import add_calendar_features


def run_feature_engineering(
    df: pd.DataFrame,
    lags: list[int] = [1, 7, 30],
    rolling_windows: list[int] = [4, 8, 12],
) -> pd.DataFrame:
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_calendar_features(df)
    return df
