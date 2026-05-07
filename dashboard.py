import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 5px 0 0 0;
        font-size: 28px;
    }
    .model-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        color: white;
        transition: transform 0.2s ease;
    }
    .model-badge:hover {
        transform: scale(1.05);
    }
    .badge-xgboost { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .badge-lstm { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .badge-prophet { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
    .badge-sarima { background: linear-gradient(135deg, #cc2b5e, #753a88); }
    .badge-lightgbm { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .header-container {
        text-align: center;
        padding: 10px 0 25px 0;
    }
    .forecast-table {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = Path("saved_models")
ALL_MODELS = ["SARIMA", "Prophet", "XGBoost", "LightGBM", "LSTM"]


@st.cache_data
def load_data():
    historical = pd.read_csv(MODEL_DIR.parent / "data" / "preprocessed.csv", parse_dates=["Date"]) \
        if (MODEL_DIR.parent / "data" / "preprocessed.csv").exists() else None
    eval_pred = pd.read_csv(MODEL_DIR / "eval_predictions.csv", parse_dates=["Date"])
    forecast_pred = pd.read_csv(MODEL_DIR / "forecast_predictions.csv", parse_dates=["Date"])
    metrics = pd.read_csv(MODEL_DIR / "state_metrics.csv")
    best = pd.read_csv(MODEL_DIR / "best_models.csv")
    return historical, eval_pred, forecast_pred, metrics, best


@st.cache_data
def load_preprocessed():
    import sys
    sys.path.insert(0, ".")
    from app.preprocessing.pipeline import run_preprocessing
    return run_preprocessing(Path("data/Forecasting Case- Study.xlsx"))


historical_raw, eval_pred, forecast_pred, metrics_df, best_df = load_data()
if historical_raw is None:
    historical_raw = load_preprocessed()

best_map = dict(zip(best_df["state"], best_df["best_model"]))
states = sorted(best_map.keys())

# --- Header ---
st.markdown("""
<div class="header-container">
    <h1>📈 Sales Forecasting Dashboard</h1>
    <p style="font-size: 16px; opacity: 0.7;">
        Automated model selection &middot; 43 US States &middot; 8-Week Forecast
    </p>
</div>
""", unsafe_allow_html=True)

# --- State & Model Selector ---
col_select, col_model, col_badge = st.columns([3, 2, 1])
with col_select:
    selected_state = st.selectbox(
        "Select a State",
        states,
        index=states.index("California"),
    )

winner = best_map[selected_state]

with col_model:
    selected_model = st.selectbox(
        "Select a Model",
        ALL_MODELS,
        index=ALL_MODELS.index(winner),
    )

with col_badge:
    is_winner = selected_model == winner
    badge_class = f"badge-{selected_model.lower()}"
    badge_label = f"🏆 {selected_model} (Best)" if is_winner else selected_model
    st.markdown(f"""
    <div style="padding-top: 28px;">
        <span class="model-badge {badge_class}">{badge_label}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Metrics ---
state_model_metrics = metrics_df[
    (metrics_df["state"] == selected_state) & (metrics_df["model"] == selected_model)
]

if not state_model_metrics.empty:
    state_metrics = state_model_metrics.iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>MAPE</h3>
            <h1>{state_metrics['mape']:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>MAE</h3>
            <h1>${state_metrics['mae']/1e6:.1f}M</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>RMSE</h3>
            <h1>${state_metrics['rmse']/1e6:.1f}M</h1>
        </div>
        """, unsafe_allow_html=True)

    if not is_winner:
        winner_metrics = metrics_df[
            (metrics_df["state"] == selected_state) & (metrics_df["model"] == winner)
        ].iloc[0]
        mape_diff = state_metrics["mape"] - winner_metrics["mape"]
        st.caption(
            f"Best model for {selected_state} is **{winner}** "
            f"(MAPE {winner_metrics['mape']:.2f}%). "
            f"Current selection is {mape_diff:+.2f}% higher."
        )

st.markdown("<br>", unsafe_allow_html=True)

# --- Chart 1: Actual vs Predicted (Last 8 weeks evaluation) ---
st.subheader("Actual vs Predicted — Model Validation (Last 8 Weeks)")

hist_state = historical_raw[historical_raw["State"] == selected_state].sort_values("Date")
eval_state = eval_pred[
    (eval_pred["State"] == selected_state) & (eval_pred["model"] == selected_model)
].sort_values("Date")

cutoff = hist_state["Date"].max() - pd.Timedelta(weeks=30)
hist_recent = hist_state[hist_state["Date"] >= cutoff]

test_cutoff = hist_state["Date"].max() - pd.Timedelta(weeks=8)
test_actual = hist_state[hist_state["Date"] > test_cutoff]

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=hist_recent["Date"], y=hist_recent["Total"],
    mode="lines",
    name="Historical",
    line=dict(color="#667eea", width=2.5),
))

fig1.add_trace(go.Scatter(
    x=test_actual["Date"], y=test_actual["Total"],
    mode="lines+markers",
    name="Actual (Test)",
    line=dict(color="#2ecc71", width=2.5),
    marker=dict(size=7),
))

fig1.add_trace(go.Scatter(
    x=eval_state["Date"], y=eval_state["Predicted"],
    mode="lines+markers",
    name=f"Predicted ({selected_model})",
    line=dict(color="#f5576c", width=2.5, dash="dash"),
    marker=dict(size=7, symbol="diamond"),
))

fig1.add_vrect(
    x0=test_actual["Date"].min(), x1=test_actual["Date"].max(),
    fillcolor="rgba(245,87,108,0.08)", line_width=0,
    annotation_text="Evaluation Period", annotation_position="top left",
)

fig1.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="", yaxis_title="Sales ($)",
    hovermode="x unified",
    yaxis=dict(tickformat=",.0f"),
)

st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Future 8-Week Forecast ---
st.subheader("Sales Forecast — Next 8 Weeks")

forecast_state = forecast_pred[
    (forecast_pred["State"] == selected_state) & (forecast_pred["model"] == selected_model)
].sort_values("Date")

last_hist_point = hist_state.iloc[-1]

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=hist_recent["Date"], y=hist_recent["Total"],
    mode="lines",
    name="Historical",
    line=dict(color="#667eea", width=2.5),
))

bridge_dates = [last_hist_point["Date"], forecast_state["Date"].iloc[0]]
bridge_values = [last_hist_point["Total"], forecast_state["Predicted"].iloc[0]]
fig2.add_trace(go.Scatter(
    x=bridge_dates, y=bridge_values,
    mode="lines",
    name="",
    line=dict(color="#38ef7d", width=2, dash="dot"),
    showlegend=False,
))

fig2.add_trace(go.Scatter(
    x=forecast_state["Date"], y=forecast_state["Predicted"],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#38ef7d", width=3),
    marker=dict(size=9, symbol="circle",
                line=dict(width=2, color="white")),
    fill="tozeroy",
    fillcolor="rgba(56,239,125,0.08)",
))

fig2.add_vrect(
    x0=forecast_state["Date"].min(), x1=forecast_state["Date"].max(),
    fillcolor="rgba(56,239,125,0.05)", line_width=0,
    annotation_text="Forecast Zone", annotation_position="top left",
)

fig2.update_layout(
    template="plotly_dark",
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="", yaxis_title="Sales ($)",
    hovermode="x unified",
    yaxis=dict(tickformat=",.0f"),
)

st.plotly_chart(fig2, use_container_width=True)

# --- Forecast Table ---
st.subheader("Forecast Details")

display_df = forecast_state[["Date", "Predicted"]].copy()
display_df["Date"] = display_df["Date"].dt.strftime("%b %d, %Y")
display_df["Predicted"] = display_df["Predicted"].apply(lambda x: f"${x:,.0f}")
display_df.columns = ["Week", "Forecasted Sales"]
display_df = display_df.reset_index(drop=True)
display_df.index = display_df.index + 1

st.dataframe(display_df, use_container_width=True, height=340)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; opacity: 0.5;'>"
    "Built with Streamlit &middot; Models: SARIMA, Prophet, XGBoost, LightGBM, LSTM &middot; Auto-selected best per state"
    "</p>",
    unsafe_allow_html=True,
)
