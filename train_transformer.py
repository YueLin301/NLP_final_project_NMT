import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import NMTDataset, Collate
from src.models.transformer import Transformer
from src.train import Trainer
from src.utils import set_seed, count_parameters, init_weights, save_experiment_results
from src.evaluate import calculate_bleu
from src.config import Config
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Transformer NMT')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='Normalization type')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_name', type=str, default='transformer-model.pt', help='Model save name')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fast)')
    
    args = parser.parse_args()

    cfg = Config()
    cfg.N_EPOCHS = args.n_epochs
    if args.debug:
        cfg.DEBUG = True
        print("!!! RUNNING IN DEBUG MODE (limited batches) !!!")
    
    set_seed(1234)
    
    device = cfg.DEVICE
    print(f'Using device: {device}')
    
    # Load Data
    print("Loading data...")
    train_dataset = NMTDataset(cfg.TRAIN_FILE, build_vocab=True)
    valid_dataset = NMTDataset(cfg.VALID_FILE, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
    
    pad_idx = train_dataset.tgt_vocab.stoi["<pad>"]
    collate = Collate(pad_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate)
    
    input_dim = len(train_dataset.src_vocab)
    output_dim = len(train_dataset.tgt_vocab)
    
    print(f'Source Vocab Size: {input_dim}')
    print(f'Target Vocab Size: {output_dim}')
    
    # Initialize Model
    # Transformer params usually need adjustment: d_model=256, n_head=4 etc for small data
    print(f"Initializing Transformer with {args.norm_type}...")
    model = Transformer(
        src_vocab_size=input_dim,
        tgt_vocab_size=output_dim,
        d_model=cfg.EMBED_DIM, # Reusing Config param but Transformer usually likes 512
        n_head=4, # 256 / 4 = 64
        n_layers=3, # Smaller for faster training on small data
        d_ff=512,
        dropout=cfg.DROPOUT,
        norm_type=args.norm_type,
        device=device
    ).to(device)
    
    def init_transformer_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
            
    model.apply(init_transformer_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for Transformer often better
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.makedirs(cfg.MODEL_SAVE_DIR)
        
    save_path = os.path.join(cfg.MODEL_SAVE_DIR, args.save_name)
    
    trainer = Trainer(model, train_loader, valid_loader, optimizer, criterion, device, config=cfg)
    
    print(f"Starting training for {args.n_epochs} epochs...")
    history = trainer.train(cfg.N_EPOCHS, cfg.CLIP, save_path)
    
    # Save Results and Plot
    save_base_name = os.path.splitext(args.save_name)[0]
    save_experiment_results(history, 'results', save_base_name)
    
    # Evaluate BLEU
    model.load_state_dict(torch.load(save_path))
    bleu_score = calculate_bleu(valid_loader, model, device)
    print(f'BLEU score on validation set: {bleu_score:.2f}')
    
    # Save vocabs for inference
    if args.save_name == 'transformer-model.pt':
        torch.save(train_dataset.src_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'src_vocab.pt'))
        torch.save(train_dataset.tgt_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'tgt_vocab.pt'))

if __name__ == '__main__':
    main()


