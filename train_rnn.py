import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import NMTDataset, Collate
from src.models.rnn import Encoder, Decoder, Attention, Seq2Seq
from src.train import Trainer
from src.utils import set_seed, count_parameters, init_weights, save_experiment_results
from src.evaluate import calculate_bleu
from src.config import Config
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train RNN NMT')
    parser.add_argument('--attention', type=str, default='dot', choices=['dot', 'general', 'concat'], help='Attention method')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_name', type=str, default='rnn-model.pt', help='Model save name')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (fast)')
    
    args = parser.parse_args()

    cfg = Config()
    # Override config with args
    cfg.TEACHER_FORCING_RATIO = args.teacher_forcing
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
    print(f"Initializing RNN with {args.attention} attention...")
    attn = Attention(cfg.HIDDEN_DIM, method=args.attention)
    enc = Encoder(input_dim, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.N_LAYERS, cfg.DROPOUT)
    dec = Decoder(output_dim, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.N_LAYERS, cfg.DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
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
    # Load best model
    model.load_state_dict(torch.load(save_path))
    bleu_score = calculate_bleu(valid_loader, model, device)
    print(f'BLEU score on validation set: {bleu_score:.2f}')
    
    # Save vocabs for inference (only if default model name to avoid overwriting with partial experiments)
    if args.save_name == 'rnn-model.pt':
        torch.save(train_dataset.src_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'src_vocab.pt'))
        torch.save(train_dataset.tgt_vocab, os.path.join(cfg.MODEL_SAVE_DIR, 'tgt_vocab.pt'))

if __name__ == '__main__':
    main()



