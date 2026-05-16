import torch
import torch.nn as nn

class PriceForecasterMultiHorizon(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_horizons=6, dropout_rate=0.2):
        super(PriceForecasterMultiHorizon, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Folosim GRU pentru eficiență pe regresie
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output-ul are o dimensiune egală cu numărul de orizonturi de timp
        # (ex: 6 neuroni pentru 1zi, 1sapt, 1luna, 1an, 5ani, 10ani)
        self.fc_out = nn.Linear(hidden_size, num_horizons)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] # Ultimul pas de timp
        
        # Conexiune reziduală pentru stabilitate pe orizonturi lungi
        residual = out
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual 
        
        # Fără funcție de activare finală, deoarece prezicem valori absolute/continue (preț/randament)
        out = self.fc_out(out) 
        return out