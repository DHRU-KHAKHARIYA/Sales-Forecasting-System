# System Design — Sales Forecasting

## Objective

Forecast next 8 weeks of beverage sales for each US state using historical data (2020-2023).
Train 5 models per state, auto-select the best performer based on MAPE, and serve predictions via REST API + interactive dashboard.

## Pipeline Flow

```
Raw Data → Preprocessing → Feature Engineering → Train 5 Models → Evaluate All → Select Best
                                                                                      ↓
                                                                 API (FastAPI) ← Load Best Model
                                                                       ↓
                                                                  Dashboard (Streamlit)
```

## Preprocessing Strategy

**Location:** `app/preprocessing/`

1. Parse dates from mixed formats
2. Drop single-value columns
3. Resample to weekly frequency (W-SUN) using `.resample().last()` — takes last record per week
4. Drop pre-2020 data — 2019 had 58-91 day gaps creating flat forward-filled noise
5. Forward-fill remaining missing weeks (~29 per state)

Key rule: same preprocessing pipeline used at training and inference time.

## Feature Engineering Strategy

**Location:** `app/features/`

- **Lag features:** t-1, t-7, t-30 (used by XGBoost, LightGBM)
- **Rolling statistics:** rolling mean and std (4, 8, 12 week windows)
- **Calendar features:** day of week, month, quarter, week of year, is_month_start, is_month_end
- **Holiday flag:** is_holiday_week

All lag/rolling features respect temporal ordering — no future values leak into features.

## Models

**Location:** `app/models/`

### 1. SARIMA (`arima_model.py`)
- Statistical model with seasonal order (1,1,1,52) for yearly seasonality
- Uses raw sales series directly
- Library: `statsmodels`

### 2. Prophet (`prophet_model.py`)
- Handles trend + yearly seasonality out of the box
- Minimal feature engineering — feed date + sales
- Library: `prophet`

### 3. XGBoost (`xgboost_model.py`)
- Gradient boosting on full engineered feature set (17 features)
- Recursive multi-step forecasting
- Library: `xgboost`

### 4. LightGBM (`lightgbm_model.py`)
- Same feature set and recursive forecasting as XGBoost
- Faster training, comparable accuracy
- Library: `lightgbm`

### 5. LSTM (`lstm_model.py`)
- 2-layer LSTM (hidden_size=64, dropout=0.2)
- Sequence length: 30 weeks
- MinMaxScaler normalization
- Library: `torch`

## Model Selection

- All 5 models evaluated on same 8-week held-out test period (Oct-Dec 2023)
- MAPE used for auto-selection — scale-independent, fair across states of different sizes
- Best model stored per state in `best_models.csv`

## Results

| Model | Avg MAPE | States Won |
|-------|----------|------------|
| SARIMA | 2.81% | 20 |
| XGBoost | 3.85% | 13 |
| LightGBM | 4.45% | 0 |
| Prophet | 4.80% | 0 |
| LSTM | 5.44% | 10 |

## API

**Location:** `app/api/`

- Framework: FastAPI
- Purpose: inference only — no training through the API
- Loads best model per state at startup from `saved_models/`
- Endpoints:
  - `POST /predict` — accept state + weeks, return forecast
  - `GET /health` — service health check
  - `GET /models` — best model per state with MAPE

## Dashboard

**Location:** `dashboard.py`

- Framework: Streamlit + Plotly
- State selector (43 states)
- Model selector (switch between all 5 models, defaults to winner)
- Metric cards (MAPE, MAE, RMSE)
- Actual vs Predicted validation chart (last 8 weeks)
- 8-week forecast chart with forecast zone
- Forecast details table

## Engineering Priorities

1. **Correctness first.** No leakage, proper temporal splits, reproducible results.
2. **Modularity.** Each pipeline stage independently testable.
3. **Simplicity.** Clean code over clever code.
4. **Ship it.** Get all 5 models working end-to-end before polishing any one.
