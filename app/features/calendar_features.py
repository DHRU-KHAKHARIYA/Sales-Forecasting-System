import pandas as pd


US_HOLIDAYS = {
    (1, 1),    # New Year
    (7, 4),    # Independence Day
    (12, 25),  # Christmas
    (11, 24),  # ~Thanksgiving (approximate)
    (11, 25),
    (11, 26),
    (11, 27),
    (11, 28),
    (11, 29),
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_holiday_week"] = df["Date"].apply(_is_holiday_week).astype(int)
    return df


def _is_holiday_week(date: pd.Timestamp) -> bool:
    for day_offset in range(7):
        check = date + pd.Timedelta(days=day_offset)
        if (check.month, check.day) in US_HOLIDAYS:
            return True
    return False
