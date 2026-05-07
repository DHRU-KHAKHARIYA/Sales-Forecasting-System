import pandas as pd
import numpy as np
from app.evaluation.metrics import mae, rmse, mape


def evaluate_model(
    actual_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    state: str,
) -> dict:
    actual = actual_df[actual_df["State"] == state].sort_values("Date")["Total"].values
    predicted = predicted_df[predicted_df["State"] == state].sort_values("Date")["Predicted"].values

    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    return {
        "state": state,
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
    }


def compare_models(
    actual_df: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    states: list[str],
) -> pd.DataFrame:
    rows = []
    for model_name, pred_df in predictions.items():
        for state in states:
            result = evaluate_model(actual_df, pred_df, state)
            result["model"] = model_name
            rows.append(result)

    return pd.DataFrame(rows)[["model", "state", "mae", "rmse", "mape"]]


def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "mape",
) -> pd.DataFrame:
    best = comparison_df.loc[
        comparison_df.groupby("state")[metric].idxmin()
    ][["state", "model", metric]].reset_index(drop=True)

    best.columns = ["state", "best_model", f"best_{metric}"]
    return best
