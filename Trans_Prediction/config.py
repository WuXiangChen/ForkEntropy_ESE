# 基本数据参数
seq_length = 4
# input_dim = 30
# enc_voc_size = input_dim
# 这个参数实时计算出来

output_dim = 1
target_dim = 1

# 模型参数
d_model = 16 # 输入模型的维度
hidden_dim = 32
nhead = 2
num_layers = 6
# 增加超参
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epochs = 200
clip = 1.0
weight_decay = 5e-4
inf = float('inf')


# 输入数据模型构建而得的参数
# generate parameters for Transformer
src_pad_idx = 0
# 输入序列的填充索引，这里是-1
trg_pad_idx = 0
# 目标序列的填充索引，这里是-1
trg_sos_idx = 1
# 目标序列中的其实位置
# 这里目标序列的起始位置需要修正一下啊！

dec_voc_size =  4
# 这里应该是输入序列的词汇表大小，但是在时序预测过程中，这个信息好像并不是非常必要啊
# 这里唯一的作用是在进行encoder和decoder的embedding的时候，
# 需要一个长度信息以便能够进行positional encoding的计算过程
# 所以在这里我直接将其定位输入数据的input_dim大小
# 现在五个参数都已经定好了，接下来就是构建模型了