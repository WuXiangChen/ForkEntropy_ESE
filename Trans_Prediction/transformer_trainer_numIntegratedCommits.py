# train.py
from training_process import *

import warnings

# 禁止所有警告
warnings.simplefilter("ignore")


all_predictions =[]
for delay in range(3,21):
    enc_voc_size = input_dim = delay * 7
    batch_size = 4
    pro_DV = "numIntegratedCommits"
    datasets = generate_datasets_allPro(delay=delay, target=pro_DV)
    dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True) for ds in datasets]

    # max_len的长度应当与seq_length长度一致
    transformer_model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                                    d_model=d_model, n_head=nhead, max_len=input_dim, dec_voc_size=dec_voc_size,
                                    ffn_hidden=64, n_layers=8, drop_prob=0.1,
                                    device="cpu", enc_voc_size=enc_voc_size, finalOutputShape=delay)

    transformer_model.apply(initialize_weights)
    optimizer = Adam(params=transformer_model.parameters(),
                     lr=init_lr,
                     weight_decay=weight_decay,
                     eps=adam_eps)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)

    criterion = torch.nn.MSELoss()
    losses = train_model(transformer_model, dataloaders, criterion, optimizer, epochs, delay , pro_DV, input_dim)

    plot_losses(losses)
    save_model(transformer_model,delay,pro_DV)
    predictions = predict_and_evaluate(transformer_model, dataloaders, criterion,input_dim,delay)
    all_predictions.append(predictions)

print(all_predictions)

