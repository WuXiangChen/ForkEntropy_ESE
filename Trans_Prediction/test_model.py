# 这部分负责加载模型，在测试集合上完成测试任务
# 这边就是给定一个开源软件项目，对其未来目标进行预测

# train.py
import math
import os
import numpy as np
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dataset.generate_datasets import generate_datasets_allPro
from models.model.transformer import Transformer
from transformer_model import TransformerModel
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from  utils import save_pre_and_reals
import argparse
import torch

# Save the models
def save_model(model, pth_path='model_saved/transformer_model'):
    torch.save(model.state_dict(), pth_path)

def extract_inputs_targets(batch):
    encoder_inputs = batch[0][:, :input_dim].view(seq_length, input_dim)
    decoder_targets = batch[0][:, input_dim:input_dim+1].view(seq_length, output_dim)
    predict_targets = batch[0][:, -1].view(seq_length, target_dim)
    return encoder_inputs, decoder_targets, predict_targets

# Predict
def predict_and_evaluate(model, dataloaders, criterion,pro_DV):
    model.eval()
    predictions = []
    reals = []
    mse_loss = 0.0
    total_samples = 0

    dataloader = dataloaders[0]
    test_train_data = list(dataloader)
    test_len = int(0.8*len(test_train_data))
    test_data = test_train_data[test_len:]

    train_data = test_train_data[:test_len]
    for i, batch in enumerate(test_data):
        encoder_inputs, decoder_targets, predict_targets = extract_inputs_targets(batch)
        with torch.no_grad():
            output = model(encoder_inputs, decoder_targets)

            output_reshape = output.contiguous().view(-1,)
            predict_targets_ = predict_targets.contiguous().view(-1)

            mse_loss += criterion(output_reshape, predict_targets_).item()
            total_samples += predict_targets.numel()
            predictions.extend(output.view(-1).tolist())
            reals.extend(predict_targets_.view(-1).tolist())

    train_values = []
    for i, batch in enumerate(train_data):
        _, _ , train_targets = extract_inputs_targets(batch)
        train_targets = train_targets.contiguous().view(-1).detach().cpu().numpy()
        train_values.extend(train_targets)

    predictions = train_values + predictions
    reals = train_values + reals

    folder_path = "results/" + pro_DV + "/"
    output_csv_path = folder_path + f"delay_size_{str(delay_size)} + Ablation_{str(ablaNum)}" + '_predictions_and_reals.csv'
    if not os.path.exists(output_csv_path):
        os.mkdir(folder_path)
    save_pre_and_reals(predictions,reals,output_csv_path)
    average_mse = mse_loss / total_samples
    rmse = math.sqrt(average_mse)
    print(f'RMSE on Test Data: {rmse}')

# Extract inputs and targets from a batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Model Testing')
    parser.add_argument('--target', type=str, default='ratioMergedPrs', help='Target variable for prediction')
    parser.add_argument('--ablaNum', type=int, default=-1, help='the num choiced for disable the corresponding feature')
    args = parser.parse_args()

    pro_DV = args.target
    ablaNum = args.ablaNum
    if pro_DV == "ratioMergedPrs":
        delay_size = 7
    else:
        delay_size = 8

    model_saved_path = "model_saved/"
    # 创建一个列表，其中每个元素都是一个 TensorDataset
    for delay in range(1,10):
        enc_voc_size = input_dim = (delay * delay_size) - 2
        batch_size = 4
        datasets = generate_datasets_allPro(delay=delay, target=pro_DV, ablaNum=ablaNum)
        dataloaders = [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True) for ds in datasets]

        transformer_model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                                        d_model=d_model, n_head=nhead, max_len=input_dim, dec_voc_size=dec_voc_size,
                                        ffn_hidden=64, n_layers=8, drop_prob=0.1,
                                        device="cuda", enc_voc_size=enc_voc_size)

        pth_path = model_saved_path + "TM_" + pro_DV + "_" + str(delay) + ".pth"
        save_model(transformer_model,pth_path)

        # checkpoint = torch.load(pth_path)
        # transformer_model.load_state_dict(checkpoint)
        criterion = torch.nn.MSELoss()

        predictions = predict_and_evaluate(transformer_model, dataloaders, criterion, pro_DV)



