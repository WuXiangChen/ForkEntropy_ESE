
import math

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

def initialize_weights_(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Training
def train_model(model, dataloaders, criterion, optimizer, epochs,delay,pro_DV,input_dim):
    losses = []
    for epoch in range(epochs):
        epoch_loss = []
        for j, dataloader in enumerate(dataloaders):
            train_data = list(dataloader)
            train_len = int(0.8*len(train_data))
            train_data = train_data[0:train_len]
            for i, batch in enumerate(train_data):
                encoder_inputs, decoder_targets, predict_targets = extract_inputs_targets(batch,input_dim,delay)

                outputs = model(encoder_inputs, decoder_targets)
                output_reshape = outputs.contiguous().view(-1,)
                predict_targets_ = predict_targets.contiguous().view(-1)

                optimizer.zero_grad()
                loss = criterion(output_reshape, predict_targets_)
                optimizer.step()
                epoch_loss.append(loss.item())

        each_epoch_loss = np.nanmean(epoch_loss)
        losses.append(each_epoch_loss)
        print(f'Epoch {epoch} Loss {each_epoch_loss}')
        # Convert losses to a DataFrame
    df_losses = pd.DataFrame(losses, columns=['Loss'])

    # Save DataFrame to a file
    df_losses.to_csv("results/loss_record_"+pro_DV+"_"+str(delay)+".csv", index=False)
    return losses

# Plotting
def plot_losses(losses):
    plt.plot(losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Save the models
def save_model(model,delay,pro_DV, filename='model_saved/TM_'):
    filename = filename+"_"+pro_DV+"_"+ str(delay)+".pth"
    torch.save(model.state_dict(), filename)

# Predict
def predict_and_evaluate(model, dataloaders, criterion,input_dim,delay):
    model.eval()
    predictions = []
    mse_loss = 0.0
    total_samples = 0
    for dataloader in dataloaders:
        test_data = list(dataloader)
        test_len = int(0.8*len(test_data))
        test_data = test_data[test_len:]
        for i, batch in enumerate(test_data):
            encoder_inputs, decoder_targets, predict_targets = extract_inputs_targets(batch,input_dim,delay)
            with torch.no_grad():
                output = model(encoder_inputs, decoder_targets)

                output_reshape = output.contiguous().view(-1,)
                predict_targets_ = predict_targets.contiguous().view(-1)

                mse_loss += criterion(output_reshape, predict_targets_).item()
                total_samples += predict_targets.numel()
                predictions.extend(output.view(-1).tolist())

    average_mse = mse_loss / total_samples
    rmse = math.sqrt(average_mse)
    print(f'RMSE on Test Data: {rmse}')
    return rmse

# Extract inputs and targets from a batch
def extract_inputs_targets_(batch,input_dim,delay):
    encoder_inputs = batch[0][:, :(input_dim-delay)].view(seq_length, -1)
    decoder_targets = batch[0][:, (input_dim-delay):input_dim].view(seq_length, -1)
    predict_targets = batch[0][:, (input_dim-delay+1):].view(seq_length, -1)
    return encoder_inputs, decoder_targets, predict_targets

def extract_inputs_targets(batch,input_dim,delay):
    encoder_inputs = batch[0][:, :(input_dim-delay)].view(seq_length, -1)
    decoder_targets = batch[0][:, (input_dim-delay):input_dim].view(seq_length, -1)
    predict_targets = batch[0][:, -1].view(seq_length, -1)
    return encoder_inputs, decoder_targets, predict_targets

