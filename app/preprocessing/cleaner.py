import pandas as pd


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df


def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def resample_to_weekly(
    df: pd.DataFrame,
    freq: str = "W-SUN",
) -> pd.DataFrame:
    resampled_frames = []

    for state in df["State"].unique():
        state_df = df[df["State"] == state].sort_values("Date")

        state_df = state_df.set_index("Date")
        weekly = state_df[["Total"]].resample(freq).last()

        weekly["State"] = state
        resampled_frames.append(weekly)

    result = pd.concat(resampled_frames).reset_index()
    return result[["State", "Date", "Total"]]


def drop_before_year(df: pd.DataFrame, year: int = 2020) -> pd.DataFrame:
    return df[df["Date"].dt.year >= year].reset_index(drop=True)


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    filled_frames = []

    for state in df["State"].unique():
        state_df = df[df["State"] == state].sort_values("Date").copy()
        state_df["Total"] = state_df["Total"].ffill()
        filled_frames.append(state_df)

    return pd.concat(filled_frames).reset_index(drop=True)
