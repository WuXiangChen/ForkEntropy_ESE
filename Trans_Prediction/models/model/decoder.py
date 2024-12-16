"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,device=device,
                                        vocab_size=dec_voc_size)
        # decoder部分的embedding和encoder部分的没有区别
        # 都是token embedding和position embedding的计算之和的形式

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, device=device)
                                     for _ in range(n_layers)])

        # 层与层之间可以不存在resNet结构

        self.linear = nn.Linear(d_model, 1,device=device)



    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        # 这里面有个问题，那就是所有layer的输入层，其输入参数都是一致的？
        # 这个设计是作者的误解，还是原设计就是这个样子的？
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)

        return output
