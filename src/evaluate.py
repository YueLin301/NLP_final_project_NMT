import torch
import sacrebleu
from tqdm import tqdm

def calculate_bleu(data_loader, model, device, max_len=50):
    model.eval()
    
    trgs = []
    pred_trgs = []
    
    # We need the vocab to decode
    tgt_vocab = data_loader.dataset.tgt_vocab
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(tqdm(data_loader, desc="Calculating BLEU")):
            # Debug: stop after 2 batches if config says so, but we don't have config here directly.
            # Usually passed via model or we can check a global flag, or just rely on data_loader being short.
            # Let's check if the dataset has a debug flag or similar? 
            # Or pass a 'debug' arg. For now, let's keep it full unless manually set.
            # Actually, to make 'debug_run.py' effective, we should limit this too.
            # Let's use a quick hack: if len(trgs) > 100, stop.
            
            src = src.to(device)
            # tgt is used only for reference, model should generate
            
            # Simple greedy decoding for evaluation
            # For RNN:
            if not (isinstance(model, torch.nn.Transformer) or hasattr(model, 'encoder_layers')):
                # RNN greedy decode
                # This depends on how RNN is implemented. 
                # The current Seq2Seq forward assumes tgt is passed for teacher forcing or length.
                # For proper inference we need a separate generation loop.
                # But wait, the Seq2Seq forward with teacher_forcing=0 essentially does greedy decoding 
                # IF it feeds its own prediction back.
                
                # However, it needs 'tgt' to know the length. 
                # Ideally, we should have a 'generate' method.
                
                # Let's use the forward pass with teacher_forcing=0 and the ground truth target length
                # This is "constrained" decoding (limited to reference length), but okay for quick validation.
                # For real inference we stop at <eos>.
                
                # Make sure tgt is on device
                tgt = tgt.to(device)
                
                outputs = model(src, tgt, teacher_forcing_ratio=0)
                # outputs = [batch, tgt len, vocab]
                preds = outputs.argmax(2)
                
            else:
                # Transformer greedy decode
                # This is more complex because we need to loop.
                # Simple loop:
                batch_size = src.shape[0]
                
                # Start with <sos>
                sos_idx = tgt_vocab.stoi["<sos>"]
                eos_idx = tgt_vocab.stoi["<eos>"]
                
                ys = torch.ones(batch_size, 1).fill_(sos_idx).type(torch.long).to(device)
                
                # We need to run encoder once ideally, but my Transformer is monolithic.
                # So we run forward repeatedly.
                
                # Limit length
                for i in range(max_len):
                    output = model(src, ys)
                    # output = [batch, cur len, vocab]
                    prob = output[:, -1]
                    _, next_word = torch.max(prob, dim=1)
                    
                    ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                    
                    # If all batches have produced EOS, we could stop, but batching makes it tricky.
                    # Simplified: just run to max_len.
                
                preds = ys
            
            # Convert indices to words
            # preds: [batch, len]
            # tgt: [batch, len]
            
            for i in range(preds.shape[0]):
                pred_sent = []
                for token in preds[i]:
                    token_item = token.item()
                    if token_item == tgt_vocab.stoi["<eos>"]:
                        break
                    if token_item not in [tgt_vocab.stoi["<sos>"], tgt_vocab.stoi["<pad>"]]:
                        pred_sent.append(tgt_vocab.itos[token_item])
                
                trg_sent = []
                for token in tgt[i]:
                    token_item = token.item()
                    if token_item == tgt_vocab.stoi["<eos>"]:
                        break
                    if token_item not in [tgt_vocab.stoi["<sos>"], tgt_vocab.stoi["<pad>"]]:
                        trg_sent.append(tgt_vocab.itos[token_item])
                
                # Join tokens
                # For English (target), we usually want spaces between tokens.
                # For Chinese (source), it depends, but we typically just join.
                # Since target is English here:
                pred_str = " ".join(pred_sent)
                trg_str = " ".join(trg_sent)
                
                # Post-processing: remove space before punctuation (optional but looks better)
                # Simple regex or replace for common punctuations
                for p in [".", ",", "?", "!", ":", ";"]:
                     pred_str = pred_str.replace(f" {p}", p)
                     trg_str = trg_str.replace(f" {p}", p)

                pred_trgs.append(pred_str)
                trgs.append(trg_str)
                
    # Calculate BLEU
    # SacreBLEU expects list of strings for sys and list of list of strings for refs
    bleu = sacrebleu.corpus_bleu(pred_trgs, [trgs])
    return bleu.score

