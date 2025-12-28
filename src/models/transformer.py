import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1)
        return x / (norm + self.eps) * self.scale

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_k = d_model // n_head
        self.n_head = n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        
        return self.fc(x), attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        _x, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(_x))
        
        _x = self.feed_forward(x)
        x = self.norm2(x + self.dropout(_x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        _x, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))
        
        _x, _ = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(_x))
        
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8, n_layers=6, d_ff=2048, max_len=100, dropout=0.1, norm_type='layernorm', device='cpu'):
        super().__init__()
        
        self.device = device
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, d_ff, dropout, norm_type) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_head, d_ff, dropout, norm_type) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        
    def make_src_mask(self, src, pad_idx):
        # src = [batch size, src len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def make_tgt_mask(self, tgt, pad_idx):
        # tgt = [batch size, tgt len]
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        
        tgt_len = tgt.shape[1]
        tril = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        
        tgt_sub_mask = tril.unsqueeze(0).unsqueeze(0)
        
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        # src = [batch size, src len]
        # tgt = [batch size, tgt len]
        
        src_mask = self.make_src_mask(src, 0) # Assuming 0 is pad
        tgt_mask = self.make_tgt_mask(tgt, 0)
        
        batch_size, src_len = src.shape
        batch_size, tgt_len = tgt.shape
        
        # Positional Encoding
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.src_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tgt = self.dropout((self.tgt_embedding(tgt) * self.scale) + self.pos_embedding(pos))
        
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
            
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
            
        return self.fc_out(tgt)


