import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    test_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["Date"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["Date"] <= cutoff].copy()
    test = df[df["Date"] > cutoff].copy()
    return train, test
