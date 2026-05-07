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
from app.evaluation.comparator import compare_models, select_best_model

DATA_PATH = Path("data/Forecasting Case- Study.xlsx")
MODEL_DIR = Path("saved_models")
N_WEEKS = 8


def main():
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    df = run_preprocessing(DATA_PATH)
    print(f"Preprocessed: {df.shape[0]} rows, {df['State'].nunique()} states")

    print("\nSTEP 2: Feature Engineering")
    print("=" * 60)
    df_featured = run_feature_engineering(df)
    print(f"Features added: {df_featured.shape[1]} columns")

    print("\nSTEP 3: Train/Test Split")
    print("=" * 60)
    train_raw, test_raw = time_based_split(df, test_weeks=N_WEEKS)
    train_feat, test_feat = time_based_split(df_featured.dropna(), test_weeks=N_WEEKS)
    print(f"Train: {train_raw['Date'].min()} to {train_raw['Date'].max()}")
    print(f"Test:  {test_raw['Date'].min()} to {test_raw['Date'].max()}")

    states = sorted(df["State"].unique())
    all_predictions = {"SARIMA": [], "Prophet": [], "XGBoost": [], "LightGBM": [], "LSTM": []}

    print(f"\nSTEP 4: Training all models for {len(states)} states")
    print("=" * 60)

    for i, state in enumerate(states):
        print(f"\n[{i+1}/{len(states)}] {state}")

        # SARIMA
        try:
            print("  Training SARIMA...", end=" ")
            sarima = arima_model.train(train_raw, state)
            pred = arima_model.forecast(sarima, n_weeks=N_WEEKS)
            all_predictions["SARIMA"].append(pred)
            arima_model.save_model(sarima, MODEL_DIR / f"sarima_{state}.joblib")
            print("done")
        except Exception as e:
            print(f"failed: {e}")

        # Prophet
        try:
            print("  Training Prophet...", end=" ")
            prop = prophet_model.train(train_raw, state)
            pred = prophet_model.forecast(prop, n_weeks=N_WEEKS)
            all_predictions["Prophet"].append(pred)
            prophet_model.save_model(prop, MODEL_DIR / f"prophet_{state}.joblib")
            print("done")
        except Exception as e:
            print(f"failed: {e}")

        # XGBoost
        try:
            print("  Training XGBoost...", end=" ")
            xgb = xgboost_model.train(train_feat, state)
            pred = xgboost_model.forecast(xgb, train_raw, n_weeks=N_WEEKS)
            all_predictions["XGBoost"].append(pred)
            xgboost_model.save_model(xgb, MODEL_DIR / f"xgboost_{state}.joblib")
            print("done")
        except Exception as e:
            print(f"failed: {e}")

        # LightGBM
        try:
            print("  Training LightGBM...", end=" ")
            lgb = lightgbm_model.train(train_feat, state)
            pred = lightgbm_model.forecast(lgb, train_raw, n_weeks=N_WEEKS)
            all_predictions["LightGBM"].append(pred)
            lightgbm_model.save_model(lgb, MODEL_DIR / f"lightgbm_{state}.joblib")
            print("done")
        except Exception as e:
            print(f"failed: {e}")

        # LSTM
        try:
            print("  Training LSTM...", end=" ")
            lstm = lstm_model.train(train_raw, state, epochs=50)
            pred = lstm_model.forecast(lstm, n_weeks=N_WEEKS)
            all_predictions["LSTM"].append(pred)
            lstm_model.save_model(lstm, MODEL_DIR / f"lstm_{state}.joblib")
            print("done")
        except Exception as e:
            print(f"failed: {e}")

    # Combine predictions
    pred_dfs = {}
    for model_name, preds in all_predictions.items():
        if preds:
            pred_dfs[model_name] = pd.concat(preds, ignore_index=True)

    print(f"\nSTEP 5: Evaluation")
    print("=" * 60)

    comparison = compare_models(test_raw, pred_dfs, states)
    print("\n--- Average Metrics Across All States ---")
    print(comparison.groupby("model")[["mae", "rmse", "mape"]].mean().round(2).to_string())

    best = select_best_model(comparison)
    print("\n--- Best Model Per State ---")
    print(best.to_string(index=False))

    # Save results
    comparison.to_csv("saved_models/model_comparison.csv", index=False)
    best.to_csv("saved_models/best_models.csv", index=False)
    print("\nResults saved to saved_models/")


if __name__ == "__main__":
    main()
