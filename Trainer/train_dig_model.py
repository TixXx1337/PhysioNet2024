import copy
import os
import sys
from tqdm import tqdm
import torch
from typing import NamedTuple, List
from torch.utils.data import DataLoader
from pathlib import Path
import helper_code
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from Datahandling.Dataloader_withYOLO import ECG_Turned
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from model.ECG_Dig import get_model
from scipy.stats import zscore
from scipy.signal import correlate
from scipy.ndimage import shift




class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    SNR : float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    SNR : List[float]

class FitResult(NamedTuple):
    loss_train: List[float]
    SNR_train : List[float]
    loss_val: List[float]
    SNR_val : List[float]


class Ecg12LeadImageNetTrainerDig():
    def __init__(self, model, optimizer, loss_fn, device="cpu"):
        self.model = model.to(device,dtype=torch.float)
        self.loss_fn = loss_fn.to(device,dtype=torch.float)
        self.optimizer = optimizer
        self.device = device

    def train(self, num_of_epochs, train_dataloader, val_datloader):
        SNR = []
        loss = []
        SNR_val = []
        loss_val = []
        for epoch in range(num_of_epochs):
            running_loss_train = 0
            running_SNR_train = 0
            progress_bar_train = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_train:.4f}')
            for idx, batch in enumerate(progress_bar_train):
                batch_result = self._train_batch(batch)
                running_loss_train += batch_result.loss
                running_SNR_train += batch_result.SNR
                avg_loss = running_loss_train / (idx + 1)
                progress_bar_train.set_description(f'Train Epoch {epoch + 1}/{num_of_epochs} - Loss: {avg_loss:.4f}')
            SNR.append(running_loss_train/len(train_dataloader))
            loss.append(running_loss_train/len(train_dataloader))
            print(f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {loss[-1]:.4f} - SNR: {SNR[-1]:.4f}')


            #Val loop
            running_loss_val = 0
            running_SNR_val = 0
            progress_bar_val = tqdm(val_datloader, desc=f'Val Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_val:.4f}')
            for idx, batch in enumerate(progress_bar_val):
                batch_result = self._val_batch(batch)
                running_loss_val += batch_result.loss
                running_SNR_val += batch_result.SNR
                avg_loss = running_loss_val / (idx + 1)
                avg_SNR = running_SNR_val / (idx + 1)
                progress_bar_val.set_description(f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {avg_loss:.4f} - SNR: {avg_SNR:.4f}')

            loss_val.append(running_loss_val/len(val_datloader))
            SNR_val.append(running_SNR_val/len(val_datloader))
            print(f"Epoch {epoch+1} Val: SNR: {SNR_val[-1]}, Loss: {loss_val[-1]}\n")
        return FitResult(loss, SNR,loss_val, SNR_val)


    def _train_batch(self, batch):
        x,_, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        snr = np.array([helper_code.compute_snr(y[i].detach().cpu(), out[i].detach().cpu()) for i in range(out.shape[0])])
        return BatchResult(loss, snr.mean())

    def _val_batch(self, batch):
        x,_, y = batch
        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out, y)
        snr = np.array([helper_code.compute_snr(y[i].detach().cpu(), out[i].detach().cpu()) for i in range(out.shape[0])])
        return BatchResult(loss, snr.mean())



def plot_results(f1_score:List,loss:List, plot_name:str, output_path:str=None)->None:
    plt.plot(range(len(f1_score)), f1_score,label='F1_score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.title('F1 Score')
    plt.savefig(f'{output_path}/F1_score_{plot_name}.png', transparent=True)
    plt.close()
    plt.plot(range(len(loss)), loss,label='F1_score')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.savefig(f'{output_path}/Loss_{plot_name}.png')
    plt.close()



def normalize_signal(signal, min_val=-3, max_val=3):
    # Normalize to range [0, 1]
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal

def inverse_normalize_signal(normalized_signal, min_val=-3, max_val=3):
    # Inverse normalize from range [-1, 1] back to original range
    signal = (normalized_signal * (max_val - min_val)) + min_val
    return signal


def del_wrong_data(signal, threshold:float=15):
    z_scores = np.abs(zscore(signal, axis=0))
    outlier_indices = np.where(z_scores > threshold)[0]
    unique_outlier_indices = np.unique(outlier_indices)
    return unique_outlier_indices


def align_series(data):
    # Calculate the mean series to use as the reference
    reference_series = np.mean(data, axis=0)

    # Initialize the aligned data array
    aligned_data = np.zeros_like(data)

    # Iterate over each series in the data
    for i in range(data.shape[0]):
        series = data[i]

        # Compute cross-correlation between the series and the reference
        correlation = correlate(series, reference_series, mode='full')

        # Find the shift that maximizes the correlation
        shift_index = np.argmax(correlation) - (len(series) - 1)

        # Shift the series to align the peaks
        aligned_data[i] = shift(series, shift_index, mode='nearest')

    return aligned_data


if __name__ == '__main__':
    device = 1
    model = get_model(model_name="ViT", decoder_name="ANN", dim=256)
    #model = get_model(model_name="ViT", decoder_name="LSTM")
    batch_size = 1
    path = os.getcwd() + "/data_preprocessed"
    #raw_signal = np.load(os.path.join(path, "signals.npy"))
    #path_to_dataset = ["/work/scratch/td38heni/all"]
    #path_to_dataset = ["/work/home/td38heni/CinC_cleaned/Datahandling/test_data"] #testing server
    #path_to_dataset = [r"C:\Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train\test_data"] #testing me
    #path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/fine_tune/train_set"]
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/20images"]
    yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    dataset = ECG_Turned(path_to_dataset, samples=0, YOLO_path=yolo_model)
    dataset.signal, dataset.dx, dataset.img = np.load(os.path.join(path, "signals.npy"))[:2], np.load(os.path.join(path, "dxs.npy"))[:2],  np.load(os.path.join(path, "imgs.npy"))[:2]
    raw_signal = copy.deepcopy(dataset.signal)
    normalized_signal = np.array(raw_signal)
    normalized_signal = np.nan_to_num(normalized_signal)
    #unique_outlier_indices = del_wrong_data(normalized_signal)
    #dataset.signal = np.delete(dataset.signal, unique_outlier_indices, axis=0)
    #dataset.dx = np.delete(dataset.dx, unique_outlier_indices, axis=0)
    #dataset.img = np.delete(dataset.img, unique_outlier_indices, axis=0)
    #for i in range(16):
    #    normalized_signal[:,i*250:250*(i+1)] = normalize_signal(normalized_signal[:,i*250:250*(i+1)],normalized_signal[:,i*250:250*(i+1)].min(),normalized_signal[:,i*250:250*(i+1)].max())
    for i in range(len(normalized_signal)):
        normalized_signal[i] = normalize_signal(normalized_signal[i],normalized_signal[i].min(),normalized_signal[i].max())
    #normalized_signal = align_series(normalized_signal)
    dataset.signal = normalized_signal
    ds_train, ds_val = torch.utils.data.random_split(dataset, [0.5,0.5])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    #os.makedirs(f"{path}/dir")
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss(reduction='mean')
    trainer = Ecg12LeadImageNetTrainerDig(model=model, optimizer=optimizer, loss_fn=loss_fn,device=device)
    train_result = trainer.train(num_of_epochs=25, train_dataloader=dl_train, val_datloader=dl_val)
    snrs = []
    normalized_signal =  np.load(os.path.join(path, "signals.npy"))#reload unnormalized data
    normalized_signal = np.nan_to_num(normalized_signal)
    for i in range(len(normalized_signal)):
        normalized_signal[i] = normalize_signal(normalized_signal[i],normalized_signal[i].min(),normalized_signal[i].max())
    #normalized_signal = align_series(normalized_signal)
    for i in range(16):
        normalized_signal[:,i*250:250*(i+1)] = normalize_signal(normalized_signal[:,i*250:250*(i+1)],normalized_signal[:,i*250:250*(i+1)].min(),normalized_signal[:,i*250:250*(i+1)].max())
    for i in range(len(normalized_signal)):
        normalized_signal[i] = normalize_signal(normalized_signal[i],normalized_signal[i].min(),normalized_signal[i].max())
    for batch in dl_train:
        x, dx, y = batch
        out = model(x.to(device, dtype=torch.float32)).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        #for i in range(16):
            #out[:, i*250:250 * (i + 1)] = inverse_normalize_signal(out[:, i*250:250 * (i + 1)],
            #                                                 normalized_signal[:,i*250:250*(i+1)].min(),
            #                                                 normalized_signal[:,i*250:250*(i+1)].max())
            #y[:, i*250:250 * (i + 1)] = inverse_normalize_signal(y[:, i*250:250 * (i + 1)],
            #                                                 normalized_signal[:,i*250:250*(i+1)].min(),
            #                                                 normalized_signal[:,i*250:250*(i+1)].max())
        plt.plot(y[0])
        plt.plot(out[0])
        plt.show()

        snr = np.array([helper_code.compute_snr(y[i], out[i]) for i in range(out.shape[0])]).mean()
        snrs.append(snr)


