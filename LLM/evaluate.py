import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging

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
CSV_PATH = 'raw_data/stock.csv' # Modifică cu calea corectă către fisierul tău

# --- SETUP LOGGER ---
def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger

eval_logger = setup_logger('eval_logger', 'logs/eval.log')

def evaluate_models():
    print(f"--- Începere Evaluare pe {DEVICE} ---")
    
    # 1. Încărcare Modele
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=6).to(DEVICE)
    
    # Încărcăm ponderile salvate (asigură-te că antrenamentul s-a finalizat cel puțin o dată)
    try:
        trend_model.load_state_dict(torch.load("models/final_models/trend_model_final.pth", map_location=DEVICE))
        price_model.load_state_dict(torch.load("models/final_models/price_model_final.pth", map_location=DEVICE))
        print("Modele încărcate cu succes.")
    except FileNotFoundError:
        print("Eroare: Nu s-au găsit modelele salvate. Rulează main.py (antrenarea) mai întâi.")
        return

    trend_model.eval()
    price_model.eval()

    # 2. Funcții de Loss
    trend_criterion = nn.BCEWithLogitsLoss() # <-- SCHIMBĂ AICI
    price_criterion = nn.HuberLoss()

    # 3. Încărcare Date
    _, eval_loader, _, _ = get_dataloaders(CSV_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    
    if len(eval_loader) == 0:
        print("Setul de evaluare este gol. Verifică lungimea dataset-ului și orizonturile cerute.")
        return

    running_trend_loss = 0.0
    running_price_loss = 0.0
    correct_trend_predictions = 0
    total_trend_predictions = 0

    print("\n[ Rulare Set de Evaluare ]")
    with torch.no_grad(): # DEZACTIVĂM CALCULELE PENTRU GRADIENTI
        progress_bar = tqdm(eval_loader, desc='Evaluating', unit='batch')
        
        for data, trend_targets, price_targets in progress_bar:
            data = data.to(DEVICE)
            trend_targets = trend_targets.to(DEVICE)
            price_targets = price_targets.to(DEVICE)
            
            # Predictii
            trend_preds = trend_model(data)
            price_preds = price_model(data)
            
            # Calcul Loss
            trend_loss = trend_criterion(trend_preds, trend_targets)
            price_loss = price_criterion(price_preds, price_targets)
                  
            running_trend_loss += trend_loss.item()
            running_price_loss += price_loss.item()
            
            # Calcul Acuratețe Trend (Dacă probabilitatea > 0.5, considerăm 1, altfel 0)
            trend_probs = torch.sigmoid(trend_preds)
            predicted_classes = (trend_probs > 0.5).float()
            
            correct_trend_predictions += (predicted_classes == trend_targets).sum().item()
            total_trend_predictions += trend_targets.size(0)
            
            progress_bar.set_postfix({'T_Loss': f'{trend_loss.item():.4f}', 'P_Loss': f'{price_loss.item():.4f}'})

    # Centralizare rezultate
    avg_t_loss = running_trend_loss / len(eval_loader)
    avg_p_loss = running_price_loss / len(eval_loader)
    trend_accuracy = (correct_trend_predictions / total_trend_predictions) * 100

    # Afișare și Salvare in Log
    result_str = (f"Evaluare Finalizată | "
                  f"Trend Loss: {avg_t_loss:.4f} | Trend Accuracy: {trend_accuracy:.2f}% | "
                  f"Price Loss: {avg_p_loss:.4f}")
    print(f"\n{result_str}")
    eval_logger.info(result_str)

if __name__ == "__main__":
    evaluate_models()