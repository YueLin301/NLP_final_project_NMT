import torch
import argparse
import jieba
from src.dataset import tokenize_en, tokenize_zh, Vocabulary
from src.models.rnn import Encoder, Decoder, Attention, Seq2Seq
from src.models.transformer import Transformer
from src.config import Config
import os

def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_len=50, is_transformer=False):
    model.eval()
    
    tokens = tokenize_zh(sentence)
    src_indexes = [src_vocab.stoi["<sos>"]] + [src_vocab.stoi.get(token, src_vocab.stoi["<unk>"]) for token in tokens] + [src_vocab.stoi["<eos>"]]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    if is_transformer:
        src_mask = model.make_src_mask(src_tensor, src_vocab.stoi["<pad>"])
        
        tgt_indexes = [tgt_vocab.stoi["<sos>"]]
        
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor, tgt_vocab.stoi["<pad>"])
            
            with torch.no_grad():
                output = model(src_tensor, tgt_tensor) # Note: forward in training takes (src, tgt) but predicts next token
                # Actually my transformer forward takes full tgt and masks it.
                # So we pass current tgt_tensor
                # Output: [1, cur_len, vocab]
            
            pred_token = output.argmax(2)[:,-1].item()
            tgt_indexes.append(pred_token)
            
            if pred_token == tgt_vocab.stoi["<eos>"]:
                break
                
        trg_tokens = [tgt_vocab.itos[i] for i in tgt_indexes]
        # Remove sos and eos
        trg_tokens = trg_tokens[1:-1]
        
        # Join with space
        sentence = " ".join(trg_tokens)
        
        # Simple detokenization for punctuation
        for p in [".", ",", "?", "!", ":", ";"]:
             sentence = sentence.replace(f" {p}", p)
             
        return sentence
        
    else:
        # RNN
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)
        
        tgt_indexes = [tgt_vocab.stoi["<sos>"]]
        
        for i in range(max_len):
            tgt_tensor = torch.LongTensor([tgt_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden = model.decoder(tgt_tensor, hidden, encoder_outputs)
            
            pred_token = output.argmax(1).item()
            tgt_indexes.append(pred_token)
            
            if pred_token == tgt_vocab.stoi["<eos>"]:
                break
                
        trg_tokens = [tgt_vocab.itos[i] for i in tgt_indexes]
        trg_tokens = trg_tokens[1:-1]
        
        # Join with space
        sentence = " ".join(trg_tokens)
        
        # Simple detokenization for punctuation
        for p in [".", ",", "?", "!", ":", ";"]:
             sentence = sentence.replace(f" {p}", p)
             
        return sentence

def load_model(model_type, model_path, src_vocab, tgt_vocab, device, **kwargs):
    cfg = Config()
    input_dim = len(src_vocab)
    output_dim = len(tgt_vocab)
    
    if model_type == 'rnn':
        # Default to dot, but allow override or guess from filename
        attention = kwargs.get('attention', 'dot')
        if 'general' in model_path:
            attention = 'general'
        elif 'concat' in model_path:
            attention = 'concat'
            
        print(f"Loading RNN with attention: {attention}")
        attn = Attention(cfg.HIDDEN_DIM, method=attention)
        enc = Encoder(input_dim, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.N_LAYERS, cfg.DROPOUT)
        dec = Decoder(output_dim, cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.N_LAYERS, cfg.DROPOUT, attn)
        model = Seq2Seq(enc, dec, device).to(device)
    elif model_type == 'transformer':
        # Default to layernorm, but allow override or guess from filename
        norm_type = kwargs.get('norm_type', 'layernorm')
        if 'rmsnorm' in model_path:
            norm_type = 'rmsnorm'
            
        print(f"Loading Transformer with norm_type: {norm_type}")
        model = Transformer(
            src_vocab_size=input_dim,
            tgt_vocab_size=output_dim,
            d_model=cfg.EMBED_DIM,
            n_head=4,
            n_layers=3,
            d_ff=512,
            dropout=cfg.DROPOUT,
            norm_type=norm_type,
            device=device
        ).to(device)
    else:
        raise ValueError("Unknown model type")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model

def main():
    parser = argparse.ArgumentParser(description='NMT Inference')
    parser.add_argument('--text', type=str, help='Chinese sentence to translate')
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer'], help='Model type (rnn or transformer)')
    parser.add_argument('--model_path', type=str, default='checkpoints/rnn-model.pt', help='Path to model checkpoint')
    parser.add_argument('--src_vocab', type=str, default='checkpoints/src_vocab.pt', help='Path to source vocab')
    parser.add_argument('--tgt_vocab', type=str, default='checkpoints/tgt_vocab.pt', help='Path to target vocab')
    
    parser.add_argument('--attention', type=str, default=None, help='Attention method (dot, general, concat)')
    parser.add_argument('--norm_type', type=str, default=None, help='Normalization type (layernorm, rmsnorm)')
    
    args = parser.parse_args()
    
    cfg = Config()
    device = cfg.DEVICE
    
    # Check if vocab paths exist, if not try to use defaults if model_path is in the same dir
    if not os.path.exists(args.src_vocab) or not os.path.exists(args.tgt_vocab):
         # Try looking in the same directory as the model
         model_dir = os.path.dirname(args.model_path)
         possible_src = os.path.join(model_dir, 'src_vocab.pt')
         possible_tgt = os.path.join(model_dir, 'tgt_vocab.pt')
         
         if os.path.exists(possible_src) and os.path.exists(possible_tgt):
             args.src_vocab = possible_src
             args.tgt_vocab = possible_tgt
         else:
             print(f"Vocab files not found at {args.src_vocab} and {args.tgt_vocab}, nor in {model_dir}")
             return

    print(f"Loading vocabs from {args.src_vocab} and {args.tgt_vocab}")
    src_vocab = torch.load(args.src_vocab, weights_only=False)
    tgt_vocab = torch.load(args.tgt_vocab, weights_only=False)
    
    print(f"Loading model from {args.model_path}")
    if not os.path.exists(args.model_path):
        print("Model file not found. Please train the model first.")
        return

    # Pass optional args
    kwargs = {}
    if args.attention: kwargs['attention'] = args.attention
    if args.norm_type: kwargs['norm_type'] = args.norm_type
    
    model = load_model(args.model_type, args.model_path, src_vocab, tgt_vocab, device, **kwargs)
    
    if args.text:
        translation = translate_sentence(args.text, src_vocab, tgt_vocab, model, device, is_transformer=(args.model_type=='transformer'))
        print(f"Source: {args.text}")
        print(f"Translation: {translation}")
    else:
        # Run a batch of examples if no specific text provided
        examples = [
            "今天天气很好。",
            "我喜欢学习自然语言处理。",
            "这本书很有趣。",
            "由于经济危机，很多人失去了工作。",
            "我们必须采取行动保护环境。",
            "人工智能正在改变世界。",
            "你会说英语吗？",
            "这是一个非常复杂的问题。",
            "我们需要更多的时间来完成这个项目。",
            "历史总是惊人的相似。"
        ]
        
        print("\n" + "="*30)
        print(f"Running Inference Examples (Model: {args.model_type})")
        print("="*30 + "\n")
        
        for ex in examples:
            translation = translate_sentence(ex, src_vocab, tgt_vocab, model, device, is_transformer=(args.model_type=='transformer'))
            print(f"Source: {ex}")
            print(f"Translation: {translation}")
            print("-" * 30)

if __name__ == '__main__':
    main()

