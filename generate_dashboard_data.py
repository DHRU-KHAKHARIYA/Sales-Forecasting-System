"""Generate prediction data for dashboard: 8-week eval + 8-week forecast per state, all models."""
import sys
import warnings
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from app.preprocessing.pipeline import run_preprocessing
from app.features.pipeline import run_feature_engineering
from app.utils.split import time_based_split
from app.models import arima_model, prophet_model, xgboost_model, lightgbm_model, lstm_model
from app.evaluation.metrics import mae, rmse, mape

MODEL_DIR = Path("saved_models")
DATA_PATH = Path("data/Forecasting Case- Study.xlsx")

MODEL_LOADERS = {
    "SARIMA": arima_model,
    "Prophet": prophet_model,
    "XGBoost": xgboost_model,
    "LightGBM": lightgbm_model,
    "LSTM": lstm_model,
}
MODEL_PREFIX = {
    "SARIMA": "sarima",
    "Prophet": "prophet",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "LSTM": "lstm",
}

df = run_preprocessing(DATA_PATH)
best_df = pd.read_csv(MODEL_DIR / "best_models.csv")
best_map = dict(zip(best_df["state"], best_df["best_model"]))
states = sorted(best_map.keys())

train_raw, test_raw = time_based_split(df, test_weeks=8)

all_eval = []
all_forecast = []
all_metrics = []

for state in states:
    print(f"{state}:", end="")
    for model_name, loader in MODEL_LOADERS.items():
        prefix = MODEL_PREFIX[model_name]
        model_path = MODEL_DIR / f"{prefix}_{state}.joblib"

        try:
            trained = loader.load_model(model_path)
            print(f" {model_name}", end="")

            if model_name in ("XGBoost", "LightGBM"):
                eval_pred = loader.forecast(trained, train_raw, n_weeks=8)
            else:
                eval_pred = loader.forecast(trained, n_weeks=8)
            eval_pred["model"] = model_name
            eval_pred["type"] = "evaluation"
            all_eval.append(eval_pred)

            if model_name in ("XGBoost", "LightGBM"):
                full_pred = loader.forecast(trained, train_raw, n_weeks=16)
            else:
                full_pred = loader.forecast(trained, n_weeks=16)
            forecast_pred = full_pred.tail(8).copy()
            forecast_pred["model"] = model_name
            forecast_pred["type"] = "forecast"
            all_forecast.append(forecast_pred)

            actual = test_raw[test_raw["State"] == state].sort_values("Date")["Total"].values
            predicted = eval_pred.sort_values("Date")["Predicted"].values
            min_len = min(len(actual), len(predicted))
            a, p = actual[:min_len], predicted[:min_len]

            all_metrics.append({
                "state": state,
                "model": model_name,
                "mae": mae(a, p),
                "rmse": rmse(a, p),
                "mape": mape(a, p),
            })
        except Exception as e:
            print(f" {model_name}(FAIL)", end="")

    print(" done")

pd.concat(all_eval, ignore_index=True).to_csv("saved_models/eval_predictions.csv", index=False)
pd.concat(all_forecast, ignore_index=True).to_csv("saved_models/forecast_predictions.csv", index=False)
pd.DataFrame(all_metrics).to_csv("saved_models/state_metrics.csv", index=False)

print(f"\nSaved: eval_predictions.csv, forecast_predictions.csv, state_metrics.csv")
print(f"Total: {len(all_metrics)} model-state combinations")
