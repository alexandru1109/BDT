import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.trend_lstm import TrendPredictorLSTM
from models.prediction_lstm import PriceForecasterMultiHorizon
from global_ import *

def run_batch_inference(output_csv_path="rezultate_predictii_piata.csv"):
    print(f"--- Începere inferență în masă pe {DEVICE} ---")
    
    # 1. Încărcare modele o singură dată (pentru eficiență)
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=len(HORIZONS)).to(DEVICE)

    # Am adăugat weights_only=True pentru a evita avertismentul de securitate PyTorch
    trend_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "trend_model_final.pth"), map_location=DEVICE, weights_only=True))
    price_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "price_model_final.pth"), map_location=DEVICE, weights_only=True))

    trend_model.eval()
    price_model.eval()

    # 2. Căutăm toate fișierele CSV din dataset
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            # Excludem fișierul de metadate și luăm doar CSV-urile valide
            if file.endswith('.csv') and file != 'symbols_valid_meta.csv':
                symbol = file.replace('.csv', '')
                all_files.append((os.path.join(root, file), symbol))

    if not all_files:
        print(f"Eroare: Nu s-au găsit fișiere CSV în {DATA_DIR}")
        return

    print(f"S-au găsit {len(all_files)} simboluri pentru analiză.")
    
    results = []
    labels = ["1_Zi", "1_Sapt", "1_Luna", "1_An", "5_Ani", "10_Ani"]

    # 3. Bucla de inferență cu progress bar
    progress_bar = tqdm(all_files, desc="Procesare Simboluri", unit="simbol")
    
    for file_path, symbol in progress_bar:
        try:
            df = pd.read_csv(file_path)
            
            if 'Date' not in df.columns:
                continue
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.dropna(subset=feature_cols)
            
            if len(df) < SEQ_LENGTH:
                continue # Sărim peste simbolurile cu prea puține date

            # Pregătire date
            historical_data = df.tail(SEQ_LENGTH * 3).copy()
            recent_data = df.tail(SEQ_LENGTH).copy()
            current_price = recent_data['Close'].iloc[-1]
            
            scaler = StandardScaler()
            scaler.fit(historical_data[feature_cols])
            scaled_data = scaler.transform(recent_data[feature_cols])
            
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Predicții
            with torch.no_grad():
                trend_logits = trend_model(input_tensor)
                trend_prob = torch.sigmoid(trend_logits).item()
                price_pct_preds = price_model(input_tensor).cpu().numpy()[0]

            trend_str = "CREȘTERE" if trend_prob > 0.5 else "SCĂDERE"
            
            # 4. Creare rând de date pentru CSV
            row_data = {
                "Simbol": symbol,
                "Pret_Curent": round(current_price, 4),
                "Predictie_Trend": trend_str,
                "Incredere_Trend_%": round(trend_prob * 100, 2)
            }
            
            # Adăugăm predicțiile de preț și randament pentru fiecare orizont
            for i, label in enumerate(labels):
                if i < len(price_pct_preds):
                    predicted_price = current_price * (1 + price_pct_preds[i])
                    # Salvăm atât prețul estimat, cât și randamentul estimat (procentual)
                    row_data[f"Pret_{label}"] = round(predicted_price, 4)
                    row_data[f"Randament_{label}_%"] = round(price_pct_preds[i] * 100, 2)
            
            results.append(row_data)
            
        except Exception as e:
            # Opțional: Poți da un print(f"Eroare la {symbol}: {e}") dacă vrei să faci debug
            continue

    # 5. Salvarea datelor în CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        print(f"\n[ Succes ] Au fost generate predicții pentru {len(results)} simboluri.")
        print(f"Rezultatele au fost salvate în: {output_csv_path}")
    else:
        print("\n[ Avertisment ] Nu s-au putut genera predicții pentru niciun simbol (verifică integritatea datelor).")

if __name__ == "__main__":
    run_batch_inference()