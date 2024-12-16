"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                             dec_voc_size,enc_voc_size,
                             d_model, n_head, max_len,
                             ffn_hidden, n_layers, drop_prob,
                             device):

        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        # 这里记录的是经过pad以后的有效数据输入

        self.device = device

        #  整个模型无疑是两部分组成，Encoder和Decoder，这里需要更加关注的是，Transformer在具体实现过程中的细节信息
        #
        self.encoder = Encoder(d_model=d_model,
                               enc_voc_size = enc_voc_size,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               dec_voc_size = dec_voc_size,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        # 根据代码构造逻辑，并不是layer的问题，而是这里的计算逻辑我还没搞清楚

        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        # 看到这里模型结构的设计基本上搭建完毕了

        return output

    def make_src_mask(self, src):
        # The method used for ignoring the padding procedure.
        # 为什么需要用不等号来进行padding的判别？
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # 这里的等号判别式用于判定，有效位置。并且，这里的sec_pad_idx是由build_vocab来进行定义的
        # 在本实验中，这里的padding才知道padding的确认是1
        # 这里的unsqueeze是增加维度，但是为什么需要增加维度呢？
        # 这里的mask过程是为了在进行计算的过程中不将padding信息带入到计算过程中
        return src_mask

   # 这里的trg_mask的作用与src_mask的作用还不一样
   # 一方面它确实也是用于进行padding的判别，另一方面，它还用于进行trg的mask
   # 对于第二个作用，这里trg mask应该可以看作是token级别的，在预测每一个token的时候，
   #       都需要将当前和之后的token进行mask
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # 为什么 这里它的维数是这样设计的？
        # 更加准确一点的问题是这里的broadcasting过程具体来说是什么样的？
        # broadcasting机制的不同，导致这里的维数拓展方式的差异

        trg_len = trg.shape[1]
        # trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.BoolTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        # 这段代码有问题，为什么这里的&可以使用在两个不同类型的tensor上？
        # 因为这里的&是按位与，而不是逻辑与，所以可以以广播的形式作用在两个维数不一致的tensor上

        return trg_mask
