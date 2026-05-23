import pandas as pd
import yfinance as yf
from datetime import timedelta
import os
from tqdm import tqdm
import warnings
import logging

# 1. Oprim avertismentele generale (Pandas/yfinance)
warnings.filterwarnings("ignore")

# 2. Oprim logger-ul intern yfinance care dă mesaje de eroare în consolă pentru companii delistate
logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

def calculate_accuracy_score(pred_price, actual_price):
    """Calculează scorul de acuratețe cu o marjă de 10% pentru 100 de puncte."""
    if pd.isna(actual_price) or actual_price == 0:
        return None, None
        
    error_pct = abs(pred_price - actual_price) / actual_price * 100
    
    if error_pct <= 10.0:
        return 100.0, error_pct
    else:
        # Scade 5 puncte procentuale din scor pentru fiecare 1% eroare peste marja de 10%
        score = max(0.0, 100.0 - (error_pct - 10.0) * 5.0)
        return score, error_pct

def get_actual_price(ticker_data, target_date):
    """
    Găsește prețul real ('Close') pentru o anumită dată.
    Dacă data pică în weekend sau sărbătoare, caută prima zi lucrătoare de după.
    """
    if ticker_data.empty:
        return None
    
    # Căutăm data exactă sau o marjă de până la 5 zile viitoare
    for i in range(5):
        check_date = str((target_date + timedelta(days=i)).date())
        if check_date in ticker_data.index:
            return ticker_data.loc[check_date, 'Close']
    return None

def evaluate_with_yfinance(input_csv="rezultate_predictii_piata.csv", output_csv="evaluare_yfinance_2020.csv"):
    print("--- Începere evaluare folosind date reale din Yahoo Finance ---")
    
    try:
        df_preds = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Eroare: Nu găsesc fișierul {input_csv}.")
        return

    # Data de referință la care s-a oprit setul de date
    base_date = pd.to_datetime('2020-04-01')
    
    # Maparea orizonturilor în timp real (zile calendaristice)
    horizons = {
        "1_Zi": base_date + timedelta(days=1),
        "1_Sapt": base_date + timedelta(days=7),
        "1_Luna": base_date + timedelta(days=30),
        "1_An": base_date + timedelta(days=365),
        "5_Ani": base_date + timedelta(days=365 * 5),
        "10_Ani": base_date + timedelta(days=365 * 10) # Acesta va returna N/A
    }

    results = []
    delisted_count = 0
    
    # Procesăm fiecare simbol din CSV
    progress_bar = tqdm(df_preds.iterrows(), total=df_preds.shape[0], desc="Preluare & Evaluare API")
    for index, row in progress_bar:
        symbol = str(row['Simbol']).strip()
        
        try:
            # Descărcăm istoricul real de la Yahoo Finance (acum complet silențios)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start="2020-03-25", end="2026-12-31") 
            
            if hist.empty:
                delisted_count += 1
                continue
                
            # Eliminăm timezone-ul pentru a putea face comparații clare de date
            hist.index = hist.index.tz_localize(None)
            
            eval_row = {
                "Simbol": symbol,
                "Pret_CSV_2020": row.get('Pret_Curent', 'N/A')
            }
            
            # Evaluare Trend (Ziua curentă vs Ziua următoare)
            actual_t0 = get_actual_price(hist, base_date)
            actual_t1 = get_actual_price(hist, horizons["1_Zi"])
            
            if actual_t0 is not None and actual_t1 is not None:
                eval_row["Pret_Real_API_2020"] = round(actual_t0, 4)
                actual_trend = "CREȘTERE" if actual_t1 > actual_t0 else "SCĂDERE"
                eval_row["Trend_Real"] = actual_trend
                eval_row["Trend_Corect"] = "DA" if row.get("Predictie_Trend") == actual_trend else "NU"
            else:
                eval_row["Pret_Real_API_2020"] = "N/A"
                eval_row["Trend_Real"] = "N/A"
                eval_row["Trend_Corect"] = "N/A"

            # Evaluare prețuri pe toate orizonturile de timp
            for label, target_date in horizons.items():
                pred_col = f"Pret_{label}"
                if pred_col not in row:
                    continue
                
                pred_price = row[pred_col]
                actual_price = get_actual_price(hist, target_date)
                
                eval_row[f"Pred_{label}"] = pred_price
                
                if actual_price is not None:
                    eval_row[f"Real_{label}"] = round(actual_price, 4)
                    score, error_pct = calculate_accuracy_score(pred_price, actual_price)
                    eval_row[f"Eroare_{label}_%"] = round(error_pct, 2)
                    eval_row[f"Scor_{label}"] = round(score, 2)
                else:
                    eval_row[f"Real_{label}"] = "N/A"
                    eval_row[f"Eroare_{label}_%"] = "N/A"
                    eval_row[f"Scor_{label}"] = "N/A"
                    
            results.append(eval_row)
            
        except Exception as e:
            delisted_count += 1
            continue

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"\n[ Succes ] Evaluarea cu date reale s-a încheiat.")
        print(f"-> Au fost evaluate cu succes {len(results)} simboluri.")
        print(f"-> Au fost ignorate {delisted_count} simboluri (delistate, tickers schimbați sau fără date).")
        print(f"Rezultatele au fost salvate în: {output_csv}")
    else:
        print("\n[ Avertisment ] Nu s-au putut prelua sau evalua date.")

if __name__ == "__main__":
    evaluate_with_yfinance()