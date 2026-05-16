import torch
import os
import logging
from tqdm import tqdm

from BDT.LLM.dataset.data_loader import get_dataloaders
from BDT.LLM.models.trend_lstm import TrendPredictorLSTM
from BDT.LLM.models.prediction_lstm import PriceForecasterMultiHorizon

# --- CONFIGURAȚII ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
SEQ_LENGTH = 60
INPUT_FEATURES = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
CSV_PATH = 'raw_data/stock.csv' 

# --- SETUP LOGGER ---
def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(message)s') # Format mai simplu pentru rezultate
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger

inference_logger = setup_logger('inference_logger', 'logs/inference_results.log')

def run_inference():
    print(f"--- Începere Inferență pe {DEVICE} ---")
    
    # 1. Încărcare Modele
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=6).to(DEVICE)
    
    try:
        trend_model.load_state_dict(torch.load("models/final_models/trend_model_final.pth", map_location=DEVICE))
        price_model.load_state_dict(torch.load("models/final_models/price_model_final.pth", map_location=DEVICE))
    except FileNotFoundError:
        print("Eroare: Nu s-au găsit modelele salvate.")
        return

    trend_model.eval()
    price_model.eval()

    # 2. Încărcare Date (folosim DOAR inference_loader)
    _, _, inference_loader, _ = get_dataloaders(CSV_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    
    if len(inference_loader) == 0:
        print("Setul de inferență este gol.")
        return

    print("\n[ Rulare Inferență pe date noi ]\n")
    inference_logger.info("--- NOUA SESIUNE DE INFERENTA ---")

    horizons_labels = ["1 Zi", "1 Săpt", "1 Lună", "1 An", "5 Ani", "10 Ani"]

    with torch.no_grad():
        for batch_idx, (data, trend_targets, price_targets) in enumerate(inference_loader):
            data = data.to(DEVICE)
            
            # Predicții
            raw_trend_preds = trend_model(data)
            trend_probs = torch.sigmoid(raw_trend_preds) # <-- Adaugă asta pentru probabilități
            price_preds = price_model(data)
            
            # Trecem datele înapoi pe CPU pentru afișare și logging
            trend_preds = trend_probs.cpu().numpy() # <-- Folosim trend_probs aici
            trend_targets = trend_targets.cpu().numpy()
            price_preds = price_preds.cpu().numpy()
            price_targets = price_targets.cpu().numpy()

            # Afișăm detaliat primele 3 exemple din primul batch
            if batch_idx == 0:
                for i in range(min(3, len(data))):
                    print(f"--- Exemplul {i+1} ---")
                    
                    # Logica Trend
                    predicted_trend_prob = trend_preds[i][0]
                    predicted_trend_str = "CREȘTE" if predicted_trend_prob > 0.5 else "SCADE/STAȚIONEAZĂ"
                    actual_trend_str = "CREȘTE" if trend_targets[i][0] == 1.0 else "SCADE/STAȚIONEAZĂ"
                    
                    trend_msg = f"Trend: Prezicere = {predicted_trend_str} ({predicted_trend_prob*100:.1f}%) | Realitate = {actual_trend_str}"
                    print(trend_msg)
                    inference_logger.info(trend_msg)
                    
                    # Logica Preț
                    print("Prețuri:")
                    for h_idx, label in enumerate(horizons_labels):
                        pred_p = price_preds[i][h_idx]
                        real_p = price_targets[i][h_idx]
                        eroare_procentuala = abs(pred_p - real_p) / real_p * 100
                        
                        price_msg = f"  - Orizont {label}: Prezicere = {pred_p:.2f} | Realitate = {real_p:.2f} (Eroare: {eroare_procentuala:.2f}%)"
                        print(price_msg)
                        inference_logger.info(price_msg)
                    print("\n")
            
            # Aici poți adăuga logică adițională dacă vrei să salvezi toate predicțiile din tot loader-ul
            # într-un fișier .csv separat pentru analiză ulterioară.

    print("Inferență completă. Detaliile au fost salvate în logs/inference_results.log")

if __name__ == "__main__":
    run_inference()