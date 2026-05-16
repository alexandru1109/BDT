import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class GlobalStockDataset(Dataset):
    def __init__(self, data_list, seq_length, horizons):
        self.data_list = data_list
        self.seq_length = seq_length
        self.horizons = horizons
        self.max_horizon = max(horizons)
        
        self.valid_lengths = []
        self.cumulative_lengths = [0]
        
        for data in self.data_list:
            v_len = max(0, len(data['features']) - self.seq_length - self.max_horizon)
            self.valid_lengths.append(v_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + v_len)
            
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        stock_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        
        local_idx = idx - self.cumulative_lengths[stock_idx]
        
        stock_data = self.data_list[stock_idx]
        features = stock_data['features']
        raw_close = stock_data['raw_close']
        
        x = features[local_idx : local_idx + self.seq_length]
        
        current_day_idx = local_idx + self.seq_length - 1
        current_price = raw_close[current_day_idx]
        
        next_day_price = raw_close[current_day_idx + 1]
        trend_target = 1.0 if next_day_price > current_price else 0.0
        
        price_targets = []
        for h in self.horizons:
            target_price = raw_close[current_day_idx + h]
            if current_price > 1e-4:
                pct_change = (target_price - current_price) / current_price
            else:
                pct_change = 0.0
            price_targets.append(pct_change)
            
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor([trend_target], dtype=torch.float32), \
               torch.tensor(price_targets, dtype=torch.float32)

def get_dataloaders(folder_path, seq_length=60, batch_size=64):
    from BDT.LLM.global_ import HORIZONS
    
    horizons = HORIZONS
    max_horizon = max(horizons)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    all_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    
    print(f"S-au găsit {len(all_files)} fișiere CSV în {folder_path} și subfoldere. Începe procesarea...")

    if len(all_files) == 0:
        raise FileNotFoundError(f"EROARE: Nu s-a găsit niciun fișier .csv în calea '{folder_path}' sau subfoldere. Verifică variabila DATA_DIR!")

    train_data_list = []
    eval_data_list = []
    inference_data_list = []
    
    scalers_dict = {} 

    for file_path in tqdm(all_files, desc="Procesare Companii"):
        try:
            df = pd.read_csv(file_path)
            
            if 'Date' not in df.columns or len(df) < (seq_length + max_horizon + 100):
                continue
                
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            df = df.dropna(subset=feature_cols)
            
            if len(df) < (seq_length + max_horizon + 100):
                continue 
                
            raw_close_prices = df['Close'].values
            
            total_len = len(df)
            train_idx = int(total_len * 0.95)
            eval_idx = int(total_len * 0.98)
            
            train_df = df.iloc[:train_idx]
            eval_df = df.iloc[train_idx:eval_idx]
            inference_df = df.iloc[eval_idx:]
            
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_df[feature_cols])
            eval_features = scaler.transform(eval_df[feature_cols])
            inference_features = scaler.transform(inference_df[feature_cols])
            
            symbol = os.path.basename(file_path).replace('.csv', '')
            scalers_dict[symbol] = scaler
            
            train_data_list.append({'features': train_features, 'raw_close': raw_close_prices[:train_idx]})
            eval_data_list.append({'features': eval_features, 'raw_close': raw_close_prices[train_idx:eval_idx]})
            inference_data_list.append({'features': inference_features, 'raw_close': raw_close_prices[eval_idx:]})
            
        except Exception as e:
            continue

    print("\nConstruire Tensiuni (Creare Dataset)...")
    train_dataset = GlobalStockDataset(train_data_list, seq_length, horizons)
    eval_dataset = GlobalStockDataset(eval_data_list, seq_length, horizons)
    inference_dataset = GlobalStockDataset(inference_data_list, seq_length, horizons)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, prefetch_factor=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2)

    print(f"Gata! Total secvențe generate - Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}, Inference: {len(inference_dataset):,}")

    return train_loader, eval_loader, inference_loader, scalers_dict