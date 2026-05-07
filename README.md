# Sales Forecasting System

End-to-end time-series forecasting system that predicts next 8 weeks of beverage sales across 43 US states. Trains 5 models per state, auto-selects the best based on MAPE, and serves predictions via FastAPI + Streamlit dashboard.

## Results

| Model | Avg MAPE | States Won |
|-------|----------|------------|
| SARIMA | 2.81% | 20 |
| XGBoost | 3.85% | 13 |
| LightGBM | 4.45% | 0 |
| Prophet | 4.80% | 0 |
| LSTM | 5.44% | 10 |

Best single-state result: Maine (XGBoost) at 0.52% MAPE.

## Project Structure

```
forecasting-system/
├── data/                  # Raw dataset (.xlsx)
├── notebooks/             # EDA notebook
├── app/
│   ├── preprocessing/     # Data cleaning, weekly resampling, gap handling
│   ├── features/          # Lag, rolling, calendar, holiday features
│   ├── models/            # SARIMA, Prophet, XGBoost, LightGBM, LSTM
│   ├── evaluation/        # MAE, RMSE, MAPE + auto model selection
│   ├── api/               # FastAPI inference endpoints
│   └── utils/             # Time-based train/test split
├── saved_models/          # Serialized models + prediction CSVs
├── train.py               # Training pipeline (all 5 models x 43 states)
├── generate_dashboard_data.py  # Generate eval/forecast data for dashboard
├── dashboard.py           # Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Usage

### 1. Train Models

```bash
python train.py
```

Trains 5 models (SARIMA, Prophet, XGBoost, LightGBM, LSTM) for each of 43 states. Saves models to `saved_models/` and outputs evaluation metrics. Takes ~20 minutes.

### 2. Generate Dashboard Data

```bash
python generate_dashboard_data.py
```

Loads all trained models and generates `eval_predictions.csv`, `forecast_predictions.csv`, and `state_metrics.csv` for the dashboard.

### 3. Run Dashboard

```bash
streamlit run dashboard.py
```

Interactive dashboard with:
- State selector (43 states)
- Model selector (switch between all 5 models, defaults to best)
- Metric cards (MAPE, MAE, RMSE)
- Actual vs Predicted validation chart
- 8-week forecast chart
- Forecast details table

### 4. Run API

```bash
uvicorn app.api.main:app --reload
```

Endpoints:
- `GET /health` — health check
- `GET /models` — list best model per state with MAPE
- `POST /predict` — forecast next N weeks for a state

```json
POST /predict
{
  "state": "California",
  "weeks": 8
}
```

## Key Design Decisions

- **Dropped 2019 data**: 2019 had 58-91 day gaps creating flat forward-filled noise. Removing it improved SARIMA from 4.79% to 2.81% MAPE.
- **Weekly resampling (W-SUN)**: Raw data is irregular snapshots mostly on Sundays. Resampled using `.resample().last()` — takes last record per week, not sum.
- **Per-state model training**: Each state trained independently on all 5 models. Different states have different best models.
- **MAPE for model selection**: Scale-independent metric — fair comparison across states of vastly different sizes (California $850M vs Wyoming $17M).
- **8-week eval + 8-week forecast**: Train up to Oct 8 2023, first 8 weeks = evaluation (actuals available), next 8 = true forecast.
- **Forward-fill for missing weeks**: More honest than interpolation — assumes values stayed the same until next report.

## Dataset

Forecasting Case Study — 8,084 rows, 43 US states, weekly beverage sales (Jan 2019 - Dec 2023). Training uses 2020-2023 only.

## Tech Stack

- **ML**: statsmodels, prophet, xgboost, lightgbm, pytorch
- **Data**: pandas, numpy, scikit-learn
- **API**: FastAPI, uvicorn
- **Dashboard**: Streamlit, Plotly
- **Visualization**: matplotlib, seaborn
