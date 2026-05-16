import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from BDT.LLM.models.trend_lstm import TrendPredictorLSTM
from BDT.LLM.models.prediction_lstm import PriceForecasterMultiHorizon
from BDT.LLM.global_ import *

def predict_for_symbol(symbol):
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=len(HORIZONS)).to(DEVICE)

    trend_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "trend_model_final.pth"), map_location=DEVICE))
    price_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "price_model_final.pth"), map_location=DEVICE))

    trend_model.eval()
    price_model.eval()

    file_path = None
    for root, dirs, files in os.walk(DATA_DIR):
        if f"{symbol}.csv" in files:
            file_path = os.path.join(root, f"{symbol}.csv")
            break

    if not file_path:
        print(f"Eroare: Simbolul {symbol} nu a fost găsit în {DATA_DIR}")
        return

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.dropna(subset=feature_cols)
    
    if len(df) < SEQ_LENGTH:
        print(f"Eroare: Date insuficiente pentru {symbol}.")
        return

    historical_data = df.tail(SEQ_LENGTH * 3).copy()
    recent_data = df.tail(SEQ_LENGTH).copy()
    current_price = recent_data['Close'].iloc[-1]
    
    scaler = StandardScaler()
    scaler.fit(historical_data[feature_cols])
    scaled_data = scaler.transform(recent_data[feature_cols])
    
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        trend_logits = trend_model(input_tensor)
        trend_prob = torch.sigmoid(trend_logits).item()
        price_pct_preds = price_model(input_tensor).cpu().numpy()[0]

    print(f"\nRezultate pentru {symbol}:")
    print(f"Preț curent: {current_price:.2f}")
    print("-" * 30)
    
    trend_str = "CREȘTERE" if trend_prob > 0.5 else "SCĂDERE"
    print(f"Predicție Trend (mâine): {trend_str} (Încredere: {trend_prob*100:.2f}%)")
    
    print("\nPredicții Preț:")
    labels = ["1 Zi", "1 Săpt", "1 Lună", "1 An", "5 Ani", "10 Ani"]
    for i, label in enumerate(labels):
        if i < len(price_pct_preds):
            predicted_price = current_price * (1 + price_pct_preds[i])
            print(f"  - Peste {label}: {predicted_price:.2f} ({price_pct_preds[i]*100:+.2f}%)")

if __name__ == "__main__":
    target = input("Introdu simbolul dorit (ex: AAPL, TSLA, A): ").upper()
    predict_for_symbol(target)