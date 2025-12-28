import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import json
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def save_experiment_results(history, save_dir, save_name):
    """
    Save training history to JSON and plot learning curves.
    history: dict with keys 'train_loss', 'valid_loss', 'valid_bleu'
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Save raw data
    json_path = os.path.join(save_dir, f"{save_name}.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Plotting
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss (Epoch-based)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss (Epoch)')
    plt.plot(epochs, history['valid_loss'], label='Val Loss (Epoch)')
    plt.title(f'Loss Curve (Epoch) - {save_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{save_name}_loss_epoch.png"))
    plt.close()
    
    # Plot Step Loss
    if 'step_loss' in history and history['step_loss']:
        plt.figure(figsize=(10, 5))
        # Optional: Smoothening
        steps = range(1, len(history['step_loss']) + 1)
        plt.plot(steps, history['step_loss'], label='Train Loss (Step)', alpha=0.6)
        
        # Moving average
        if len(history['step_loss']) > 100:
            window = 50
            moving_avg = np.convolve(history['step_loss'], np.ones(window)/window, mode='valid')
            plt.plot(range(window, len(history['step_loss']) + 1), moving_avg, label='Moving Avg', color='red')
            
        plt.title(f'Loss Curve (Step) - {save_name}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{save_name}_loss_step.png"))
        plt.close()
    
    # Plot BLEU if available
    if 'valid_bleu' in history and history['valid_bleu']:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['valid_bleu'], label='Val BLEU', color='green')
        plt.title(f'BLEU Curve - {save_name}')
        plt.xlabel('Epochs')
        plt.ylabel('BLEU Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{save_name}_bleu.png"))
        plt.close()
