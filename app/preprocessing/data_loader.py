import pandas as pd
from pathlib import Path


def load_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if file_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    required_cols = {"State", "Date", "Total"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
