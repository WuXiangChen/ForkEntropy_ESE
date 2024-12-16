# train.py
import math
import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import torch
from dataset.generate_datasets import generate_datasets_allPro
from models.model.transformer import Transformer
import pandas as pd
import matplotlib.pyplot as plt
from config import *

seed = 42
torch.random.initial_seed()  
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
# torch.set_deterministic(True)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Data
# Model, Loss function, Optimizer
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

# Training
def train_model(model, dataloaders, criterion, optimizer, epochs ,delay, pro_DV, input_dim):
    losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        count = 0
        train_data = list(dataloaders)
        # 本意是用来划分测试与数据集的
        for i, batch in enumerate(train_data):
            encoder_inputs, decoder_targets, predict_targets = extract_inputs_targets(batch,input_dim)
            # 这里要对prdict_targets进行处理,将其纵向扩展为4*4维度

            optimizer.zero_grad()
            encoder_inputs = encoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            outputs = model(encoder_inputs, decoder_targets)

            output_reshape = outputs.contiguous().view(-1,).to(device)
            predict_targets_ = predict_targets.contiguous().view(-1).to(device)

            loss = criterion(output_reshape, predict_targets_)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        losses.append(epoch_loss / count)
        print(f'Epoch {epoch + 1} Loss {epoch_loss / count}')
        # Convert losses to a DataFrame
    df_losses = pd.DataFrame(losses, columns=['Loss'])

    # Save DataFrame to a file
    folder_path = f"results/TrainLoss/{pro_DV}/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    df_losses.to_csv(folder_path + f"delay_{str(delay)} + delay_size_{str(delay_size)} + Ablation_{str(ablaNum)}_trainLoss.csv", index=False)
    return losses

# Plotting
def plot_losses(losses):
    plt.plot(losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Save the models
def save_model(model, pth_path='model_saved/TM_'):
    torch.save(model.state_dict(), pth_path)

# Predict
def predict_and_evaluate(model, dataloaders, criterion, input_dim):
    model.eval()
    predictions = []
    mse_loss = 0.0
    total_samples = 0
    # 本意是用来划分测试与数据集的
    for i, batch in enumerate(dataloaders):
        encoder_inputs, decoder_targets, predict_targets = extract_inputs_targets(batch,input_dim)
        with torch.no_grad():
            encoder_inputs = encoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            output = model(encoder_inputs, decoder_targets)

            output_reshape = output.contiguous().view(-1,).to(device)
            predict_targets_ = predict_targets.contiguous().view(-1).to(device)

            mse_loss += criterion(output_reshape, predict_targets_).item()
            total_samples += predict_targets.numel()
            predictions.extend(output.view(-1).tolist())

    average_mse = mse_loss / total_samples
    rmse = math.sqrt(average_mse)
    print(f'RMSE on Test Data: {rmse}')

# Extract inputs and targets from a batch
def extract_inputs_targets(batch, input_dim):
    encoder_inputs = batch[:, :input_dim].reshape(seq_length, -1)
    decoder_targets = batch[:, input_dim:input_dim+1].reshape(seq_length, -1)
    predict_targets = batch[:, -1].reshape(seq_length, -1)
    return encoder_inputs, decoder_targets, predict_targets

def main(delay_start, delay_end, batch_size, pro_DV, delay_size, device):
    all_predictions = []
    model_saved_path = f"model_saved/{pro_DV}/"
    if not os.path.exists(model_saved_path):
        os.mkdir(model_saved_path)

    for delay in range(delay_start, delay_end, -1):
        enc_voc_size = input_dim = ((delay + 1) * delay_size) - 2
        train_dataset, test_datasets = generate_datasets_allPro(delay=delay, target=pro_DV, ablaNum=ablaNum)

        dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        # 初始化Transformer模型
        transformer_model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                                        d_model=d_model, n_head=nhead, max_len=input_dim*5, dec_voc_size=dec_voc_size,
                                        ffn_hidden=64, n_layers=8, drop_prob=0.1,
                                        device=device, enc_voc_size=enc_voc_size).to(device)
        
        transformer_model.apply(initialize_weights)
        optimizer = Adam(params=transformer_model.parameters(),
                         lr=init_lr,
                         weight_decay=weight_decay,
                         eps=adam_eps)
        criterion = torch.nn.MSELoss()
        # 训练模型
        losses = train_model(transformer_model, dataloaders, criterion, optimizer, epochs, delay, pro_DV, input_dim)
        # 绘制损失图
        #plot_losses(losses)
        pth_path = model_saved_path + "TM_" + pro_DV + "_delay_size_"+ str(delay_size) + "_delay_" + str(delay) + ".pth"
        # 保存模型
        save_model(transformer_model,pth_path)
        dataloaders = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, drop_last=True)
        # 预测和评估
        predictions = predict_and_evaluate(transformer_model, dataloaders, criterion, input_dim)
        all_predictions.append(predictions)

    print(all_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model on GitHub data.")
    parser.add_argument("--delay_start", type=int, default=2, help="Start delay value.")
    parser.add_argument("--delay_end", type=int, default=1, help="End delay value.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for DataLoader.")
    parser.add_argument("--pro_DV", type=str, default="ratioMergedPrs", help="Target variable.")
    parser.add_argument("--device", type=int, default=0, help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--ablaNum', type=int, default=0, help='the num choiced for disable the corresponding feature')
    args = parser.parse_args()
    ablaNum = args.ablaNum
    pro_DV = args.pro_DV
    if pro_DV == "ratioMergedPrs":
        delay_size = 8
    else:
        delay_size = 7

    if ablaNum!=-1:
        delay_size-=1

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    main(args.delay_start, args.delay_end, args.batch_size, pro_DV, delay_size, device)