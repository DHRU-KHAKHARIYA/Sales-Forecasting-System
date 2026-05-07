import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.api.schemas import (
    PredictRequest,
    PredictResponse,
    WeeklyForecast,
    ModelInfo,
    HealthResponse,
)
from app.preprocessing.pipeline import run_preprocessing
from app.models import arima_model, prophet_model, xgboost_model, lightgbm_model, lstm_model

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "saved_models"
DATA_PATH = PROJECT_ROOT / "data" / "Forecasting Case- Study.xlsx"

app = FastAPI(
    title="Sales Forecasting API",
    description="Forecast next N weeks of sales per US state using the best auto-selected model.",
    version="1.0.0",
)

MODELS = {}
BEST_MODELS = {}
PREPROCESSED_DATA = None

MODEL_LOADERS = {
    "SARIMA": arima_model,
    "Prophet": prophet_model,
    "XGBoost": xgboost_model,
    "LightGBM": lightgbm_model,
    "LSTM": lstm_model,
}

MODEL_FILE_PREFIX = {
    "SARIMA": "sarima",
    "Prophet": "prophet",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "LSTM": "lstm",
}


@app.on_event("startup")
def load_models():
    global MODELS, BEST_MODELS, PREPROCESSED_DATA

    best_df = pd.read_csv(MODEL_DIR / "best_models.csv")
    BEST_MODELS = dict(zip(best_df["state"], best_df["best_model"]))

    for state, model_name in BEST_MODELS.items():
        prefix = MODEL_FILE_PREFIX[model_name]
        model_path = MODEL_DIR / f"{prefix}_{state}.joblib"
        loader = MODEL_LOADERS[model_name]
        MODELS[state] = {
            "trained": loader.load_model(model_path),
            "model_name": model_name,
            "module": loader,
        }

    PREPROCESSED_DATA = run_preprocessing(DATA_PATH)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=len(MODELS),
        states_available=len(BEST_MODELS),
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    best_df = pd.read_csv(MODEL_DIR / "best_models.csv")
    return [
        ModelInfo(
            state=row["state"],
            best_model=row["best_model"],
            best_mape=round(row["best_mape"], 4),
        )
        for _, row in best_df.iterrows()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    state = request.state
    n_weeks = request.weeks

    if state not in MODELS:
        available = sorted(BEST_MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Available: {available}",
        )

    model_entry = MODELS[state]
    trained = model_entry["trained"]
    model_name = model_entry["model_name"]
    module = model_entry["module"]

    if model_name in ("XGBoost", "LightGBM"):
        pred_df = module.forecast(trained, PREPROCESSED_DATA, n_weeks=n_weeks)
    else:
        pred_df = module.forecast(trained, n_weeks=n_weeks)

    forecast = [
        WeeklyForecast(
            date=str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]),
            predicted_sales=round(float(row["Predicted"]), 2),
        )
        for _, row in pred_df.iterrows()
    ]

    return PredictResponse(
        state=state,
        model_used=model_name,
        forecast=forecast,
    )
