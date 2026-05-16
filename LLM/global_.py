# global.py
import torch

# Setări Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimizări CUDA pentru RTX 40-Series (Ada Lovelace)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

# Căi fișiere
DATA_DIR = '/home/mi3/dataset/raw_data'
CHECKPOINT_DIR = 'models/checkpoints'
FINAL_MODELS_DIR = 'models/final_models'

# Hiperparametri Model
INPUT_FEATURES = 5   # OHLCV
HIDDEN_SIZE = 512
NUM_LAYERS = 4
SEQ_LENGTH = 60      # Fereastra de timp istoric
HORIZONS = [1, 5, 21, 252] # Zilele viitoare prezise

# Hiperparametri Antrenare
BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 1e-3