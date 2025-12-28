import torch
import os
from src.dataset import NMTDataset
from src.config import Config
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def save_vocabs():
    cfg = Config()
    print("Loading data to build vocab...")
    train_dataset = NMTDataset(cfg.TRAIN_FILE, build_vocab=True)
    
    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.makedirs(cfg.MODEL_SAVE_DIR)
        
    print(f"Saving vocabs to {cfg.MODEL_SAVE_DIR}")
    torch.save(train_dataset.src_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'src_vocab.pt'))
    torch.save(train_dataset.tgt_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'tgt_vocab.pt'))
    print("Done!")

if __name__ == '__main__':
    save_vocabs()
