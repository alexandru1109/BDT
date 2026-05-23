import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.trend_lstm import TrendPredictorLSTM
from models.prediction_lstm import PriceForecasterMultiHorizon
from global_ import *

def calculate_accuracy_score(pred_price, actual_price):
    """
    Calculează scorul de acuratețe.
    Marjă de 10% pentru acuratețe maximă (100%).
    Dacă eroarea e mai mare de 10%, scorul scade treptat.
    """
    error_pct = abs(pred_price - actual_price) / actual_price * 100
    
    if error_pct <= 10.0:  # Marja de 10% (poți modifica la 5.0 dacă vrei să fie mai strict)
        return 100.0, error_pct
    else:
        # Scade 5 puncte procentuale din scor pentru fiecare 1% eroare peste marjă
        score = max(0.0, 100.0 - (error_pct - 10.0) * 5.0)
        return score, error_pct

def evaluate_historical_accuracy(target_date_str='2020-04-01', output_csv="acuratete_istorica.csv"):
    print(f"--- Începere evaluare istorică (Target: {target_date_str}) ---")
    
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=len(HORIZONS)).to(DEVICE)

    trend_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "trend_model_final.pth"), map_location=DEVICE, weights_only=True))
    price_model.load_state_dict(torch.load(os.path.join(FINAL_MODELS_DIR, "price_model_final.pth"), map_location=DEVICE, weights_only=True))

    trend_model.eval()
    price_model.eval()

    # Orizonturile noastre (zile de trading) - ignorăm 5/10 ani deoarece e prea recent
    horizons_map = {
        "1_Zi": 1,
        "1_Sapt": 5,
        "1_Luna": 21,
        "1_An": 252
    }
    
    target_date = pd.to_datetime(target_date_str)
    
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.csv') and file != 'symbols_valid_meta.csv':
                symbol = file.replace('.csv', '')
                all_files.append((os.path.join(root, file), symbol))

    results = []
    progress_bar = tqdm(all_files, desc="Evaluare istorică", unit="simbol")
    
    for file_path, symbol in progress_bar:
        try:
            df = pd.read_csv(file_path)
            
            if 'Date' not in df.columns:
                continue
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Găsim indexul pentru data țintă (cel mai apropiat pe sau înainte de 01.04.2020)
            past_df = df[df['Date'] <= target_date]
            if past_df.empty:
                continue
                
            idx_t = past_df.index[-1]
            actual_date = df.loc[idx_t, 'Date']
            
            # Verificăm dacă avem destule date în spate
            if idx_t < SEQ_LENGTH * 3:
                continue
                
            # Curățăm datele (NaNs)
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if df.loc[idx_t - SEQ_LENGTH*3 : idx_t, feature_cols].isnull().values.any():
                continue

            current_price = df.loc[idx_t, 'Close']
            
            # Scalare Date
            historical_data = df.loc[idx_t - SEQ_LENGTH*3 + 1 : idx_t].copy()
            recent_data = df.loc[idx_t - SEQ_LENGTH + 1 : idx_t].copy()
            
            scaler = StandardScaler()
            scaler.fit(historical_data[feature_cols])
            scaled_data = scaler.transform(recent_data[feature_cols])
            
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # INFERENȚĂ
            with torch.no_grad():
                trend_logits = trend_model(input_tensor)
                trend_prob = torch.sigmoid(trend_logits).item()
                price_pct_preds = price_model(input_tensor).cpu().numpy()[0]

            predicted_trend = "CREȘTERE" if trend_prob > 0.5 else "SCĂDERE"
            
            row_data = {
                "Simbol": symbol,
                "Data_Start": actual_date.strftime('%Y-%m-%d'),
                "Pret_Start": round(current_price, 4),
                "Predictie_Trend": predicted_trend,
                "Incredere_Trend_%": round(trend_prob * 100, 2)
            }
            
            # EVALUARE TREND REAL (Comparam ziua T+1)
            if idx_t + 1 < len(df):
                actual_t1_price = df.loc[idx_t + 1, 'Close']
                actual_trend = "CREȘTERE" if actual_t1_price > current_price else "SCĂDERE"
                row_data["Trend_Real"] = actual_trend
                row_data["Trend_Corect"] = "DA" if predicted_trend == actual_trend else "NU"
            else:
                row_data["Trend_Real"] = "N/A"
                row_data["Trend_Corect"] = "N/A"

            # EVALUARE PREȚURI PE ORIZONTURI
            for i, (label, offset) in enumerate(horizons_map.items()):
                # i este indexul în array-ul prezis [1zi, 1sapt, 1luna, 1an, ...]
                pred_price = current_price * (1 + price_pct_preds[i])
                row_data[f"Pred_{label}"] = round(pred_price, 4)
                
                # Avem date reale în viitor?
                if idx_t + offset < len(df):
                    actual_price = df.loc[idx_t + offset, 'Close']
                    row_data[f"Real_{label}"] = round(actual_price, 4)
                    
                    score, error_pct = calculate_accuracy_score(pred_price, actual_price)
                    
                    row_data[f"Eroare_{label}_%"] = round(error_pct, 2)
                    row_data[f"Scor_{label}_%"] = round(score, 2)
                else:
                    row_data[f"Real_{label}"] = "N/A"
                    row_data[f"Eroare_{label}_%"] = "N/A"
                    row_data[f"Scor_{label}_%"] = "N/A"

            results.append(row_data)
            
        except Exception as e:
            continue

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n[ Succes ] Evaluarea istorică pentru {len(results)} simboluri finalizată.")
        print(f"Datele au fost salvate în: {output_csv}")
        
        # Scurt raport în consolă (Media Scorului pe An, dacă există date valide)
        try:
            valid_year_scores = [r["Scor_1_An_%"] for r in results if isinstance(r.get("Scor_1_An_%"), (int, float))]
            if valid_year_scores:
                avg_year_score = sum(valid_year_scores) / len(valid_year_scores)
                print(f"==> SCOR MEDIU ACURATEȚE PE 1 AN (pe tot portofoliul): {avg_year_score:.2f}%")
        except:
            pass

    else:
        print("\n[ Eroare ] Nu s-au putut procesa datele pentru evaluare.")

if __name__ == "__main__":
    # Poți modifica target_date_str dacă vrei să testezi și pe alte date
    evaluate_historical_accuracy(target_date_str='2020-04-01', output_csv='acuratete_istorica_2020.csv')