from pydantic import BaseModel


class PredictRequest(BaseModel):
    state: str
    weeks: int = 8


class WeeklyForecast(BaseModel):
    date: str
    predicted_sales: float


class PredictResponse(BaseModel):
    state: str
    model_used: str
    forecast: list[WeeklyForecast]


class ModelInfo(BaseModel):
    state: str
    best_model: str
    best_mape: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    states_available: int
