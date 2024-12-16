"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self,vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        # 这里为什么存在两种embedding的内容啊？
        # self.tok_emb = TokenEmbedding(vocab_size, d_model)
        # token Embedding指的是什么？
        # 这个embedding的内容应该指的是原始信息的语义输出
        # 这里的token应该不是 ts任务中必要的过程，因为它已经被编码过了


        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        # PositionEncoding指的是什么
        # 这部分内容应该指的是位序输出

        self.drop_out = nn.Dropout(p=drop_prob)
        # 这里是随机失活

    def forward(self, x):
        # x1 = torch.tensor(x).to(torch.float32)
        # tok_emb = self.tok_emb(x)
        # 上面的token过程应该不用考虑了
        # 这里面的x是正常的编码后的输入

        # 不embedding了，直接维度拓展
        tok_emb = x.unsqueeze(dim=-1).expand((x.shape[0], x.shape[1], self.d_model))
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
