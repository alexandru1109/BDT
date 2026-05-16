# utils/model_utils.py
import torch
import os

def save_checkpoint(state, filename="models/checkpoints/checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")