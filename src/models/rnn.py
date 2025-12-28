import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src len, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim, method='dot'):
        super().__init__()
        self.method = method # 'dot', 'general', 'concat'
        self.hid_dim = hid_dim
        
        if method == 'general':
            self.W = nn.Linear(hid_dim, hid_dim)
        elif method == 'concat': # Additive
            self.W = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, hid dim] (last layer hidden state)
        # encoder_outputs = [batch size, src len, hid dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Squeeze the hidden state to [batch size, hid dim]
        hidden = hidden.squeeze(0)
        
        if self.method == 'dot':
            # score = hidden * encoder_output
            # [batch size, 1, hid dim] @ [batch size, hid dim, src len] -> [batch size, 1, src len]
            score = torch.bmm(hidden.unsqueeze(1), encoder_outputs.permute(0, 2, 1))
            
        elif self.method == 'general':
            # score = hidden * W * encoder_output
            x = self.W(encoder_outputs) # [batch size, src len, hid dim]
            score = torch.bmm(hidden.unsqueeze(1), x.permute(0, 2, 1))
            
        elif self.method == 'concat':
            # score = v * tanh(W * [hidden; encoder_output])
            # Repeat hidden src_len times: [batch size, src len, hid dim]
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(combined)) # [batch size, src len, hid dim]
            score = self.v(energy).permute(0, 2, 1) # [batch size, 1, src len]
            
        return F.softmax(score, dim=2)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size] (one token)
        # hidden = [n layers, batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]
        
        input = input.unsqueeze(1) # [batch size, 1]
        embedded = self.dropout(self.embedding(input)) # [batch size, 1, emb dim]
        
        # Calculate attention weights
        # Use the last layer's hidden state for attention
        attn_weights = self.attention(hidden[-1:], encoder_outputs) # [batch size, 1, src len]
        
        # Calculate context vector
        context = torch.bmm(attn_weights, encoder_outputs) # [batch size, 1, hid dim]
        
        # Concatenate embedding and context vector
        rnn_input = torch.cat((embedded, context), dim=2) # [batch size, 1, emb dim + hid dim]
        
        # Pass to RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # output = [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        
        # Predict next token
        prediction = torch.cat((output, context, embedded), dim=2).squeeze(1)
        prediction = self.fc_out(prediction)
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # tgt = [batch size, tgt len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        # Use the hidden state from encoder as initial hidden state for decoder
        # For multi-layer, we just pass the hidden state. 
        # Note: If Encoder is bidirectional, we'd need to handle that, but requirements say unidirectional.
        
        input = tgt[:, 0] # Start token
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = tgt[:, t] if teacher_force else top1
            # Ensure input is on correct device
            input = input.to(self.device)
            
        return outputs


