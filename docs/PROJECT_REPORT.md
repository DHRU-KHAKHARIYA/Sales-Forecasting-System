# Project Report — Sales Forecasting System

## 1. Problem Statement

Given historical beverage sales data across 43 US states (Jan 2019 – Dec 2023), build a system that:
- Forecasts the next 8 weeks of sales per state
- Trains multiple models and auto-selects the best one per state
- Serves predictions through an API and interactive dashboard

The dataset contains 8,084 rows with columns: Date, State, Category (always "Beverages"), and Total (sales amount in dollars).

---

## 2. Exploratory Data Analysis

### 2.1 Initial Observations
- 43 states, 188 records per state
- Single category ("Beverages") — column is useless, dropped immediately
- No null values in the raw data
- Sales range from ~$17M (Wyoming) to ~$850M (California) — huge scale differences across states

### 2.2 Date Irregularities
The data is not clean weekly data. Key findings:
- Mixed date formats in the raw Excel file (required `format='mixed'` parsing)
- Records mostly fall on Sundays, but not consistently
- Gaps between consecutive records vary: most are ~7 days, but many are 14, 21, or even longer

### 2.3 The 2019 Problem (Critical Discovery)
During EDA, we noticed that 2019 data had massive gaps — **58 to 91 days** between consecutive records for most states. This means for a given state, you might have a record in January and then nothing until March or April.

When we resampled to weekly frequency and forward-filled (which is the standard approach for time-series gaps), these huge 2019 gaps created **8-13 weeks of flat, identical values**. This is artificial noise — the model sees "sales were exactly $X for 13 weeks straight" which never actually happened.

**Example — California 2019:**
```
Jan 6, 2019:  $812,450,000
Jan 13, 2019: $812,450,000  ← forward-filled (no real data)
Jan 20, 2019: $812,450,000  ← forward-filled
...
Mar 10, 2019: $812,450,000  ← forward-filled (still the Jan 6 value!)
Mar 17, 2019: $798,200,000  ← finally a real data point
```

This flat-line pattern is especially harmful for SARIMA, which tries to learn seasonal patterns from the data. Feeding it months of fake-constant values destroys its ability to capture real seasonality.

### 2.4 2020-2023 Data Quality
After 2019, the data quality improves significantly:
- Missing weeks drop from ~67 per state (with 2019) to ~29 per state (without 2019)
- Coverage improves from 73.7% to 85.7%
- Remaining gaps are small (1-2 weeks), where forward-fill is a reasonable assumption

### 2.5 Seasonality Patterns
- Clear monthly seasonality visible — sales tend to peak in summer months
- Year-over-year trends present across most states
- States are highly correlated with each other (similar macro patterns)
- Some outliers per state but not enough to warrant removal

---

## 3. Preprocessing Pipeline

Based on EDA findings, the preprocessing pipeline was designed as:

1. **Load data** from Excel
2. **Parse dates** with mixed format handling
3. **Drop single-value columns** (Category = "Beverages" always)
4. **Resample to weekly** (W-SUN) using `.resample().last()` — takes the last recorded value per week, not sum, because records are already aggregated totals
5. **Drop pre-2020 data** — removes the noisy 2019 data
6. **Forward-fill missing weeks** — for the remaining small gaps (1-2 weeks), forward-fill is appropriate: "assume sales stayed the same until the next report"

**Why `.resample().last()` instead of `.resample().sum()`?**
Each row in the dataset represents a cumulative total at that point in time, not a transaction. Summing would double-count. Taking the last value per week gives us the most recent snapshot for that week.

**Why forward-fill instead of interpolation?**
Interpolation assumes a smooth trend between two points, which may not be true for weekly sales (a holiday week could spike). Forward-fill is more conservative — it says "we don't know what happened, so we assume nothing changed." This is more honest and less likely to introduce artificial patterns.

---

## 4. Feature Engineering

Two types of models need different inputs:

**Time-series models (SARIMA, Prophet):** Use raw date + sales values directly. They have built-in mechanisms for trend and seasonality.

**ML models (XGBoost, LightGBM):** Need engineered features because they don't understand time natively. We created 17 features:

| Feature Group | Features | Purpose |
|---------------|----------|---------|
| Lag features | lag_1, lag_7, lag_30 | "What happened 1/7/30 weeks ago?" |
| Rolling statistics | rolling_mean_4, rolling_std_4, rolling_mean_8, rolling_std_8, rolling_mean_12, rolling_std_12 | Smoothed trends and volatility at different windows |
| Calendar | day_of_week, month, quarter, week_of_year | Seasonal patterns |
| Binary flags | is_month_start, is_month_end, is_holiday_week | Special period indicators |

**Why these specific lags?**
- `lag_1`: Most recent data point — strongest predictor
- `lag_7`: Same week last month (approximation) — captures monthly cycle
- `lag_30`: ~7 months ago — captures longer-term patterns

**What's missing (intentionally)?**
- `lag_52` (same week last year) — would give direct year-over-year comparison. We didn't add it because SARIMA and Prophet already handle yearly seasonality natively, and they're winning most states. If XGBoost were underperforming, this would be the first feature to add.
- `lag_13` (same week last quarter) — similar reasoning

**LSTM:** Uses a different approach — feeds raw sequences of 30 consecutive weeks and learns patterns from the sequence itself. No manual feature engineering needed.

---

## 5. Model Training

### 5.1 Train/Test Split

Time-based split (never random for time series):
- **Train:** Jan 2020 – Oct 8, 2023
- **Test:** Oct 15 – Dec 3, 2023 (last 8 weeks)

Same split used for all 5 models — non-negotiable for fair comparison.

### 5.2 Per-State Training
Each state is trained independently on all 5 models. Why?
- States have vastly different scales ($17M Wyoming vs $850M California)
- Different states may have different seasonal patterns
- A global model would be dominated by large states
- Per-state training lets each state have its own best model

This means **215 total models** (5 models × 43 states).

### 5.3 The Five Models

#### SARIMA (Statistical)
```
SARIMAX(order=(1,1,1), seasonal_order=(1,1,1,52))
```
- Seasonal ARIMA with 52-week (yearly) seasonality
- Uses raw sales series directly
- `enforce_stationarity=False` because some states have strong trends
- Strongest model overall — benefits most from clean data (hence the 2019 removal)

#### Prophet (Facebook/Meta)
```
Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
```
- Handles trend + yearly seasonality automatically
- Weekly/daily seasonality disabled because our data is already weekly frequency
- Good baseline with minimal tuning

#### XGBoost (Gradient Boosting)
```
XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
```
- Trains on 17 engineered features
- **Recursive multi-step forecasting:** predicts week 1, appends it to history, re-engineers features, predicts week 2, etc.
- This is important — XGBoost can't natively predict 8 steps ahead, so we chain single-step predictions

#### LightGBM (Gradient Boosting)
```
LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31)
```
- Same architecture and features as XGBoost
- Same recursive forecasting approach
- Added to compare against XGBoost — LightGBM is often faster and sometimes more accurate
- Result: XGBoost won every head-to-head comparison on this dataset

#### LSTM (Deep Learning)
```
2-layer LSTM, hidden_size=64, dropout=0.2, sequence_length=30
```
- Trained with Adam optimizer, MSE loss, 50 epochs
- MinMaxScaler normalization (fit on train only)
- Sequence length of 30 weeks — can see ~7 months of history per prediction
- Kept simple intentionally — deep learning needs more data than we have to truly shine

---

## 6. Evaluation & Model Selection

### 6.1 Metrics
Three metrics computed for each model-state combination:
- **MAE (Mean Absolute Error):** Average dollar amount off — interpretable but scale-dependent
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily
- **MAPE (Mean Absolute Percentage Error):** Percentage-based — scale-independent

### 6.2 Why MAPE for Model Selection?
California's sales are ~$850M/week. Wyoming's are ~$17M/week. An MAE of $5M means very different things for these two states:
- California: 0.6% error — excellent
- Wyoming: 29% error — terrible

MAPE normalizes this — 5% MAPE means "5% off" regardless of state size. This is the only fair way to compare across states and pick the best model per state.

### 6.3 Results — Before Data Cleaning (2019-2023)

| Model | Avg MAPE | States Won |
|-------|----------|------------|
| XGBoost | 3.92% | 21 |
| Prophet | 4.77% | 3 |
| SARIMA | 4.79% | 6 |
| LSTM | 5.00% | 13 |

XGBoost was the clear winner. SARIMA performed poorly — its seasonal patterns were corrupted by the flat forward-filled 2019 data.

### 6.4 Results — After Data Cleaning (2020-2023)

| Model | Avg MAPE | States Won | Change |
|-------|----------|------------|--------|
| **SARIMA** | **2.81%** | **20** | Improved 41% (from 4.79%) |
| XGBoost | 3.85% | 13 | Improved 2% (from 3.92%) |
| LightGBM | 4.45% | 0 | New model |
| Prophet | 4.80% | 0 | Roughly same |
| LSTM | 5.44% | 10 | Got slightly worse |

### 6.5 Key Insights from Results

**SARIMA's dramatic improvement (4.79% → 2.81%):**
Removing 2019's flat forward-filled data let SARIMA properly learn the 52-week seasonal cycle. With clean data, it became the best model overall — winning 20 of 43 states. This validates the decision to drop 2019 data.

**XGBoost still strong (3.85%):**
Slight improvement from cleaner data. Won 13 states — tends to win where sales patterns are less seasonal and more driven by recent trends (lag features).

**LightGBM vs XGBoost:**
LightGBM (4.45%) lost to XGBoost (3.85%) on every single state. This is common on smaller datasets — LightGBM's advantages (speed, memory efficiency) matter more with large datasets. On ~200 rows per state, XGBoost's approach works fine. This is itself a useful insight: "I tried LightGBM, benchmarked it, and XGBoost won — here's why."

**Prophet's consistent mediocrity (4.80%):**
Prophet didn't improve much because it's robust to data quality issues by design. But that robustness also means it can't take advantage of the cleaner data as much as SARIMA can. It won 0 states.

**LSTM got slightly worse (5.00% → 5.44%):**
Counterintuitive — less data hurt LSTM. With 2019 removed, each state has ~195 training weeks instead of ~250. LSTM needs more data than statistical models, so losing 50 weeks hurt it. Still won 10 states where patterns are complex enough that its sequence learning helps.

**Best single result:** Maine — XGBoost at 0.52% MAPE (half a percent error over 8 weeks).

---

## 7. Serving: API & Dashboard

### 7.1 FastAPI Inference API
- Loads the best model per state at startup
- Three endpoints:
  - `GET /health` — service health check
  - `GET /models` — list best model per state with MAPE scores
  - `POST /predict` — forecast next N weeks for a given state
- Handles the XGBoost/LightGBM difference (needs historical data for recursive forecasting) vs SARIMA/Prophet/LSTM (forecast directly)

### 7.2 Streamlit Dashboard
Interactive dashboard with:
- **State selector:** Pick any of 43 states
- **Model selector:** Defaults to winner, but can switch to any of the 5 models to compare
- **Metric cards:** MAPE, MAE, RMSE for the selected model
- **Winner indicator:** Badge shows which model is the best, and when viewing a non-winner, shows the MAPE gap
- **Actual vs Predicted chart:** Last 30 weeks of historical + 8-week evaluation period with both actual and predicted overlaid
- **Forecast chart:** Historical trend + 8-week future forecast with forecast zone
- **Forecast table:** Week-by-week predicted sales amounts

---

## 8. Project Timeline & Decisions Log

### Phase 1: Setup & EDA
- Project scaffolding with modular `app/` structure
- Loaded and explored the dataset
- Discovered date irregularities, 2019 data quality issues, single-value Category column
- Decided on weekly resampling with forward-fill

### Phase 2: Preprocessing & Features
- Built preprocessing pipeline: parse dates → drop columns → resample → forward-fill
- Built feature engineering: lags, rolling stats, calendar features, holiday flags
- Created time-based train/test split (last 8 weeks = test)

### Phase 3: Model Training (v1 — with 2019 data)
- Trained SARIMA, Prophet, XGBoost, LSTM across 43 states (172 models)
- XGBoost won 21 states, average MAPE 3.92%
- SARIMA underperformed at 4.79% — suspected data quality issue

### Phase 4: Data Cleaning & LightGBM
- Analyzed 2019 gaps quantitatively — confirmed 58-91 day gaps creating flat noise
- Dropped 2019 data, retrained all models on 2020-2023
- Added LightGBM as 5th model for comparison
- SARIMA jumped to 2.81% MAPE (41% improvement), became the best model
- LightGBM benchmarked at 4.45% — consistently behind XGBoost

### Phase 5: API & Dashboard
- Built FastAPI inference API with 3 endpoints
- Built Streamlit dashboard with state selector, model selector, charts, and metrics
- Generated prediction data for all 5 models × 43 states (215 combinations)

---

## 9. What I Would Do With More Time

1. **Add lag_52 and lag_13 features** — Give XGBoost/LightGBM direct year-over-year and quarter-over-quarter comparisons. SARIMA and Prophet handle this natively, but the tree models don't have it yet.

2. **Hyperparameter tuning** — Grid search for XGBoost/LightGBM (n_estimators, max_depth, learning_rate). Currently using reasonable defaults but not optimized per state.

3. **Increase LSTM epochs and sequence length** — 50 epochs is conservative. Sequence length of 30 misses yearly patterns (need 52+). Could also try GRU as a lighter alternative.

4. **Ensemble top 2 models** — For states where SARIMA and XGBoost are close, a weighted average might beat both.

5. **Confidence intervals** — SARIMA and Prophet can provide prediction intervals natively. XGBoost could use quantile regression. This gives the business a range, not just a point estimate.

6. **Model comparison dashboard page** — Side-by-side comparison of all 5 models across states, leaderboard, US heatmap colored by winning model.

7. **Automated retraining pipeline** — Scheduled retraining as new data comes in, with drift detection to flag when models degrade.

---

## 10. Conclusion

The final system achieves an average MAPE of 2.81% (SARIMA) across 43 states — meaning predictions are off by less than 3% on average. The key insight was that **data quality matters more than model complexity**: simply removing noisy 2019 data improved SARIMA by 41%, making it the best model after being one of the worst.

The system is modular (each component independently testable), serves predictions via API, and provides an interactive dashboard for exploration. All 5 models are available for comparison, with the best automatically selected per state.
