import pandas as pd
from pathlib import Path

from app.preprocessing.data_loader import load_data
from app.preprocessing.cleaner import (
    parse_dates,
    drop_single_value_columns,
    resample_to_weekly,
    drop_before_year,
    fill_missing_weeks,
)


def run_preprocessing(file_path: str | Path, freq: str = "W-SUN") -> pd.DataFrame:
    df = load_data(file_path)
    df = parse_dates(df)
    df = drop_single_value_columns(df)
    df = resample_to_weekly(df, freq=freq)
    df = drop_before_year(df, year=2020)
    df = fill_missing_weeks(df)
    return df
