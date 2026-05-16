# train.py
import torch
from tqdm import tqdm
from global_ import DEVICE, BATCH_SIZE, EPOCHS
from utils.logger import setup_logger
from utils.model_utils import save_checkpoint

train_logger = setup_logger('train', 'logs/train.log')

def train_model(model, train_loader, optimizer, criterion, start_epoch=0):
    model.train()
    
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        
        # Bara de progres pentru epoch-ul curent
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # Folosim Mixed Precision pentru performanță
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Actualizăm interfața din consolă în timp real
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        train_logger.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.6f}")
        
        # Salvare Checkpoint la fiecare epoch
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        save_checkpoint(checkpoint_state, f"models/checkpoints/checkpoint_epoch_{epoch+1}.pth")