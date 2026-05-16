import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import logging

# Importăm modelele noastre
from BDT.LLM.models.trend_lstm import TrendPredictorLSTM
from BDT.LLM.models.prediction_lstm import PriceForecasterMultiHorizon
from BDT.LLM.global_ import * 
# --- CONFIGURAȚII GLOBALE ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- OPTIMIZĂRI CUDA (RTX 40-Series) ---
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high') # Activează TF32

# --- SETUP LOGGERS ---
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

train_logger = setup_logger('train_logger', 'logs/train.log')
eval_logger = setup_logger('eval_logger', 'logs/eval.log')

# --- FUNCȚIA PRINCIPALĂ ---
def main():
    print(f"--- Începere proces. Device activ: {DEVICE} ---")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Inițializare Modele
    trend_model = TrendPredictorLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    price_model = PriceForecasterMultiHorizon(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, num_horizons=len(HORIZONS)).to(DEVICE)

    # 2. Funcții de Loss și Optimizatori
    # Trend = Clasificare Binară -> Binary Cross Entropy
    trend_criterion = nn.BCEWithLogitsLoss() 
    price_criterion = nn.HuberLoss() 

    trend_optimizer = optim.AdamW(trend_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    price_optimizer = optim.AdamW(price_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    trend_scheduler = torch.optim.lr_scheduler.StepLR(trend_optimizer, step_size=3, gamma=0.5)
    price_scheduler = torch.optim.lr_scheduler.StepLR(price_optimizer, step_size=3, gamma=0.5)

    scaler = GradScaler('cuda')

    from BDT.LLM.dataset.data_loader import get_dataloaders
    from BDT.LLM.global_ import DATA_DIR, SEQ_LENGTH, BATCH_SIZE

    print(f"Încărcare set de date masiv din folderul: {DATA_DIR}...")
    train_loader, eval_loader, inference_loader, scalers_dict = get_dataloaders(
        folder_path=DATA_DIR,
        seq_length=SEQ_LENGTH, 
        batch_size=BATCH_SIZE
    )

    os.makedirs('models/checkpoints', exist_ok=True)

    # 4. Bucla de Antrenare
    print("\n[ Începere Antrenare ]")
    for epoch in range(EPOCHS):
        trend_model.train()
        price_model.train()
        
        running_trend_loss = 0.0
        running_price_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')
        
        for batch_idx, (data, trend_targets, price_targets) in enumerate(progress_bar):
            data = data.to(DEVICE, non_blocking=True)
            trend_targets = trend_targets.to(DEVICE, non_blocking=True)
            price_targets = price_targets.to(DEVICE, non_blocking=True)
            
            # --- Antrenare Model Trend ---
            trend_optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'): # Context pentru Mixed Precision
                trend_preds = trend_model(data)
                trend_loss = trend_criterion(trend_preds, trend_targets)
            
            scaler.scale(trend_loss).backward()
            scaler.step(trend_optimizer)
            
            # --- Antrenare Model Preț ---
            price_optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                price_preds = price_model(data)
                price_loss = price_criterion(price_preds, price_targets)
            
            scaler.scale(price_loss).backward()
            scaler.step(price_optimizer)
            
            # Update la scaler pentru următorul batch
            scaler.update()
            
            running_trend_loss += trend_loss.item()
            running_price_loss += price_loss.item()
            
            # Update consolă
            progress_bar.set_postfix({
                'T_Loss': f'{trend_loss.item():.4f}', 
                'P_Loss': f'{price_loss.item():.4f}'
            })

        trend_scheduler.step()
        price_scheduler.step()
            
        # Logare la final de epoch
        avg_t_loss = running_trend_loss / len(train_loader)
        avg_p_loss = running_price_loss / len(train_loader)
        train_logger.info(f"Epoch {epoch+1}/{EPOCHS} | Trend Loss: {avg_t_loss:.6f} | Price Loss: {avg_p_loss:.6f}")

        # Salvare Checkpoints
        if (epoch + 1) % 5 == 0: # Salvăm o dată la 5 epoci
            torch.save({
                'epoch': epoch + 1,
                'trend_state_dict': trend_model.state_dict(),
                'price_state_dict': price_model.state_dict(),
                'trend_optimizer': trend_optimizer.state_dict(),
                'price_optimizer': price_optimizer.state_dict(),
            }, f"models/checkpoints/checkpoint_epoch_{epoch+1}.pth")
            train_logger.info(f"Checkpoint salvat pentru epoch-ul {epoch+1}")

    print("\n[ Antrenare Finalizată ]")
    # Salvare finală în folderul dedicat
    os.makedirs('models/final_models', exist_ok=True)
    torch.save(trend_model.state_dict(), "models/final_models/trend_model_final.pth")
    torch.save(price_model.state_dict(), "models/final_models/price_model_final.pth")

if __name__ == "__main__":
    main()