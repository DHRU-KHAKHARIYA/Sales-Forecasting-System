import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


SEQUENCE_LENGTH = 30


class LSTMNet(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def _create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train(df: pd.DataFrame, state: str, epochs: int = 50) -> dict:
    series = df[df["State"] == state].sort_values("Date")["Total"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = _create_sequences(scaled, SEQUENCE_LENGTH)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    model = LSTMNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    return {"model": model, "scaler": scaler, "state": state, "last_sequence": scaled[-SEQUENCE_LENGTH:]}


def forecast(trained: dict, n_weeks: int = 8) -> pd.DataFrame:
    model = trained["model"]
    scaler = trained["scaler"]
    state = trained["state"]
    current_seq = trained["last_sequence"].copy()

    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(n_weeks):
            x = torch.FloatTensor(current_seq.reshape(1, SEQUENCE_LENGTH, 1))
            pred_scaled = model(x).numpy()[0, 0]
            pred_value = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_value)
            current_seq = np.append(current_seq[1:], [[pred_scaled]], axis=0)

    last_date = pd.Timestamp("2023-10-08")
    dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(n_weeks)]

    return pd.DataFrame({
        "Date": dates,
        "State": state,
        "Predicted": predictions,
    })


def save_model(trained: dict, path: str | Path) -> None:
    path = Path(path)
    save_data = {
        "model_state_dict": trained["model"].state_dict(),
        "scaler": trained["scaler"],
        "state": trained["state"],
        "last_sequence": trained["last_sequence"],
    }
    joblib.dump(save_data, path)


def load_model(path: str | Path) -> dict:
    data = joblib.load(Path(path))
    model = LSTMNet()
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": data["scaler"],
        "state": data["state"],
        "last_sequence": data["last_sequence"],
    }
