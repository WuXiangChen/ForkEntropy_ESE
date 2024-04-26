
seq_length = 4

output_dim = 1
target_dim = 1

d_model = 16
hidden_dim = 64
nhead = 2
num_layers = 6
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epochs = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')

src_pad_idx = 0
trg_pad_idx = 0
trg_sos_idx = 1
dec_voc_size =  1
