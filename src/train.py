import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import os
from src.utils import epoch_time
from src.config import Config
from src.evaluate import calculate_bleu

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device, scheduler=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.config = config

    def train_epoch(self, clip, update_history_callback=None):
        self.model.train()
        epoch_loss = 0
        
        for i, (src, tgt) in enumerate(tqdm(self.train_loader, desc="Training")):
            # Debug: stop after 5 batches
            if hasattr(self.config, 'DEBUG') and self.config.DEBUG and i >= 5:
                break
                
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            if isinstance(self.model, nn.Transformer) or hasattr(self.model, 'encoder_layers'): # Transformer
                tgt_input = tgt[:, :-1]
                output = self.model(src, tgt_input)
                output_dim = output.shape[-1]
                
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(output, tgt_output)
            else: # RNN
                output = self.model(src, tgt, teacher_forcing_ratio=getattr(self.config, 'TEACHER_FORCING_RATIO', 0.5))
                output_dim = output.shape[-1]
                
                output = output[:, 1:].contiguous().view(-1, output_dim)
                tgt_output = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(output, tgt_output)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            
            # Record step-level loss
            if update_history_callback:
                update_history_callback(loss_val)
            
        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, (src, tgt) in enumerate(tqdm(self.valid_loader, desc="Validating")):
                # Debug: stop after 2 batches
                if hasattr(self.config, 'DEBUG') and self.config.DEBUG and i >= 2:
                    break
                    
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                if isinstance(self.model, nn.Transformer) or hasattr(self.model, 'encoder_layers'):
                    tgt_input = tgt[:, :-1]
                    output = self.model(src, tgt_input)
                    output_dim = output.shape[-1]
                    
                    output = output.contiguous().view(-1, output_dim)
                    tgt_output = tgt[:, 1:].contiguous().view(-1)
                    
                    loss = self.criterion(output, tgt_output)
                else:
                    output = self.model(src, tgt, teacher_forcing_ratio=0) # Turn off teacher forcing
                    output_dim = output.shape[-1]
                    
                    output = output[:, 1:].contiguous().view(-1, output_dim)
                    tgt_output = tgt[:, 1:].contiguous().view(-1)
                    
                    loss = self.criterion(output, tgt_output)
                
                epoch_loss += loss.item()
                
        return epoch_loss / len(self.valid_loader)

    def train(self, n_epochs, clip, save_path):
        best_valid_loss = float('inf')
        
        history = {
            'train_loss': [], # Epoch-level
            'valid_loss': [],
            'valid_bleu': [],
            'step_loss': []   # New: Step-level
        }
        
        # Try to resume from history if exists
        import json
        history_path = os.path.splitext(save_path)[0] + ".json"
        
        start_epoch = 0
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    old_history = json.load(f)
                    start_epoch = len(old_history['train_loss'])
                    history = old_history
                    # Initialize step_loss if resuming from old format
                    if 'step_loss' not in history:
                         history['step_loss'] = []
                    
                    print(f"Resuming from epoch {start_epoch+1}")
                    
                    if os.path.exists(save_path):
                        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
                        print("Loaded model checkpoint")
            except Exception as e:
                print(f"Failed to resume history: {e}")
        
        # Callback to update step loss
        def update_step_loss(loss_val):
            history['step_loss'].append(loss_val)

        for epoch in range(start_epoch, n_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(clip, update_history_callback=update_step_loss)
            valid_loss = self.evaluate()
            
            # Calculate BLEU on validation set (can be slow, but useful)
            # Use a smaller subset or just do it? Dataset is small enough (500 valid)
            bleu = calculate_bleu(self.valid_loader, self.model, self.device)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), save_path)
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['valid_bleu'].append(bleu)
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Val. BLEU: {bleu:.2f}')
            
            # Save intermediate history
            from src.utils import save_experiment_results
            save_base_name = os.path.basename(os.path.splitext(save_path)[0])
            save_dir = os.path.dirname(save_path)
            # Use separate results dir if save_path is in checkpoints
            if 'checkpoints' in save_dir:
                results_dir = save_dir.replace('checkpoints', 'results')
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                save_experiment_results(history, results_dir, save_base_name)
            else:
                save_experiment_results(history, 'results', save_base_name)
            
        return history
