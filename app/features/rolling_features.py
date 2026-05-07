import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Total",
    windows: list[int] = [4, 8, 12],
) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        grouped = df.groupby("State")[target_col]
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    return df
