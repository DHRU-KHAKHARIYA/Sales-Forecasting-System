import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Total",
    lags: list[int] = [1, 7, 30],
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("State")[target_col].shift(lag)
    return df
