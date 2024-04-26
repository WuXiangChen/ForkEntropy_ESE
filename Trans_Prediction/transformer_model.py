# transformer_model.py

import torch.nn as nn
from models.layers.multi_head_attention import MultiHeadAttention
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, output_dim, num_layers):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, src):
        src = src.permute(1, 0, 2)
        attn_output, _ = self.self_attn(q=src,k= src,v= src)
        output = self.feed_forward(attn_output)
        return output

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, nhead)
        self.multihead_attn = MultiHeadAttention(input_dim, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, tgt, memory):
        tgt = tgt.permute(1, 0, 2)  # Transformer expects seq_len, batch, output_dim
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + tgt2
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + tgt2
        output = self.feed_forward(tgt)
        output = output.permute(1, 0, 2)
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, target_dim, nhead, num_layers):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, nhead, output_dim, num_layers)
        self.decoder = Decoder(output_dim, hidden_dim,output_dim, nhead, num_layers)
        self.fc = nn.Linear(output_dim, target_dim)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        output = self.fc(decoder_output)
        return output