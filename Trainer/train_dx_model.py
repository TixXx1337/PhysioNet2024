# noinspection PyPackageRequirements
import abc
import os
import sys
from typing import NamedTuple, List
from torch.utils.data import DataLoader, RandomSampler
from typing import Callable, Any
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
#from model.ECG_Dx import *
from tqdm import tqdm
from Datahandling.Dataloader_withYOLO import ECG_Turned,ECG_cropped
from ultralytics import YOLO
import torch
import torch.nn as nn


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int
    num_TP: int
    num_TN: int
    num_FP: int
    num_FN: int
    out : List[float]
    y : List[int]


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float
    num_TP: int
    num_TN: int
    num_FP: int
    num_FN: int
    out : float
    y : int

class FitResult(NamedTuple):
    f1_score: List[float]
    loss: List[float]
    f1_score_val: List[float]
    loss_val: List[float]



class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device=0, classification_threshold=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.classification_threshold = classification_threshold
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, save_every:int=None,early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []




        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            (loss, acc, TP, TN, FP, FN, out, y) = train_result
            train_loss += loss
            train_acc.append(acc)
            if dl_test is not None:
                test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
                (loss, acc, TP, TN, FP, FN, out, y) = test_result
                test_loss += loss
            #if len(train_loss) > 10:
            #    last_values = test_loss[-4:]
            #    if all(last_values[i] < last_values[i + 1] for i in range(3)):
            #        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)


            actual_num_epochs += 1

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        y = []
        out = []
        num_correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                y.append(batch_res.y)
                out.append(batch_res.out)
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                TP += batch_res.num_TP
                TN += batch_res.num_TN
                FP += batch_res.num_FP
                FN += batch_res.num_FN

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            f1 = 0
            if TP + TN + FP + FN > 0:
                f1 = 2 * TP / (2*TP + FP + FN)

            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f}), '
                                 f"F1-score {f1:.3f}"
                                 )

        return EpochResult(losses=losses, accuracy=accuracy, num_TP=TP, num_TN=TN, num_FP=FP, num_FN=FN, y=y, out=out)


class Ecg12LeadNetTrainerBinary(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x).flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        num_correct = torch.sum((out > 0) == (y == 1))
        TP = torch.sum((out > 0) * (y == 1))
        TN = torch.sum((out <= 0) * (y == 0))
        FP = torch.sum((out > 0) * (y == 0))
        FN = torch.sum((out <= 0) * (y == 1))

        return BatchResult(loss.item(), num_correct.item(), TP, TN, FP, FN, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x).flatten()
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1))
            out_norm = torch.sigmoid(out)
            if self.classification_threshold == None:
                TP = torch.sum((out > 0) * (y == 1))
                TN = torch.sum((out <= 0) * (y == 0))
                FP = torch.sum((out > 0) * (y == 0))
                FN = torch.sum((out <= 0) * (y == 1))
            else:
                TP = torch.sum((out_norm >= self.classification_threshold) * (y == 1))
                TN = torch.sum((out_norm < self.classification_threshold) * (y == 0))
                FP = torch.sum((out_norm >= self.classification_threshold) * (y == 0))
                FN = torch.sum((out_norm < self.classification_threshold) * (y == 1))
                num_correct = torch.sum((out_norm > self.classification_threshold) == (y == 1))

        return BatchResult(loss.item(), num_correct.item(), TP, TN, FP, FN, out, y)


class Ecg12LeadNetTrainerMulticlass(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)  # .flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        indices = out > 0  # torch.max(out, 1)  #_,
        indices1 = y > 0  # torch.max(y, 1)  #_,

        num_correct = torch.sum(indices == indices1)

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out.flatten(), y.flatten())
            indices = out > 0  # torch.max(out, 1) _,
            indices1 = y > 0  # torch.max(y, 1) _,

            num_correct = torch.sum(indices == indices1)
        return BatchResult(loss.item(), num_correct.item())


class Ecg12LeadImageNetTrainerBinary(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x).flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        num_correct = torch.sum((out > 0) == (y == 1))
        TP = torch.sum((out > 0) * (y == 1))
        TN = torch.sum((out <= 0) * (y == 0))
        FP = torch.sum((out > 0) * (y == 0))
        FN = torch.sum((out <= 0) * (y == 1))

        return BatchResult(loss.item(), num_correct.item(), TP, TN, FP, FN, out, y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x).flatten()
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1))
            out_norm = torch.softmax(out, dim=-1)
            if self.classification_threshold == None:
                TP = torch.sum((out > 0) * (y == 1))
                TN = torch.sum((out <= 0) * (y == 0))
                FP = torch.sum((out > 0) * (y == 0))
                FN = torch.sum((out <= 0) * (y == 1))
            else:
                TP = torch.sum((out_norm > self.classification_threshold) * (y == 1))
                TN = torch.sum((out_norm <= self.classification_threshold) * (y == 0))
                FP = torch.sum((out_norm > self.classification_threshold) * (y == 0))
                FN = torch.sum((out_norm <= self.classification_threshold) * (y == 1))
        return BatchResult(loss.item(), num_correct.item(), TP, TN, FP, FN, out, y)


class Ecg12LeadImageNetTrainerMulticlass():
    def __init__(self, model, optimizer, loss_fn, device="cpu"):
        self.model = model.to(device,dtype=torch.float)
        self.loss_fn = loss_fn.to(device,dtype=torch.float)
        self.optimizer = optimizer
        self.device = device
        self.m = nn.Softmax(dim=1)

    def train(self, num_of_epochs, train_dataloader, val_datloader):
        f1_score = []
        loss = []
        f1_score_val = []
        loss_val = []
        for epoch in range(num_of_epochs):
            running_loss_train = 0
            #TP, TN, FP, FN, num_correct = 0,0,0,0,0
            # Train loop
            y_pred_train = []
            y_true_train = []
            progress_bar_train = tqdm(train_dataloader)
            for idx, batch in enumerate(progress_bar_train):
                self.model = self.model.train()
                batch_result = self._train_batch(batch)
                running_loss_train += batch_result.loss
                avg_loss = running_loss_train / (idx + 1)
                progress_bar_train.set_description(f'Train Epoch {epoch + 1}/{num_of_epochs} - Loss: {avg_loss:.4f}')
                y_true_train.append(batch_result.y.detach().cpu().numpy())
                out = self.m(batch_result.out)
                max_values, max_indices = torch.max(out, dim=1)
                mask = out == max_values.unsqueeze(1)
                y_pred_train.append(mask.float().cpu().numpy())

            y_pred_train = np.concatenate(y_pred_train, axis=0).astype(np.float32)
            y_true_train = np.concatenate(y_true_train, axis=0).astype(np.float32)
            f1_score.append(metrics.f1_score(y_true_train, y_pred_train, average='macro'))
            loss.append(running_loss_train/len(train_dataloader))
            print(f"Epoch {epoch+1} Train: F1 Score: {f1_score[-1]}, Loss: {loss[-1]}\n")


            #Val loop

            self.model =  self.model.eval()
            running_loss_val = 0
            progress_bar_val = tqdm(val_datloader, desc=f'Val Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_val:.4f}')
            y_pred_val = []
            y_true_val = []
            for idx, batch in enumerate(progress_bar_val):
                batch_result = self._val_batch(batch)
                running_loss_val += batch_result.loss
                avg_loss = running_loss_val / (idx + 1)
                progress_bar_val.set_description(f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_val:.4f}')
                y_true_val.append(batch_result.y.detach().cpu().numpy())
                max_values, max_indices = torch.max(batch_result.out, dim=1)
                mask = batch_result.out == max_values.unsqueeze(1)
                y_pred_val.append(mask.float().cpu().numpy())
            y_pred_val = np.concatenate(y_pred_val, axis=0).astype(np.float32)
            y_true_val = np.concatenate(y_true_val, axis=0).astype(np.float32)
            f1_score_val.append(metrics.f1_score(y_true_val, y_pred_val, average='macro'))
            loss_val.append(avg_loss)
            print(f"Epoch {epoch+1} Val: F1 Score: {f1_score_val[-1]}, Loss: {loss_val[-1]}\n")
        return FitResult(f1_score, loss, f1_score_val, loss_val)


    def _train_batch(self, batch):
        x, y = batch
        #x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        num_correct = torch.sum((out > 0) == (y == 1)).item()
        TP = torch.sum((out > 0) * (y == 1)).item()
        TN = torch.sum((out <= 0) * (y == 0)).item()
        FP = torch.sum((out > 0) * (y == 0)).item()
        FN = torch.sum((out <= 0) * (y == 1)).item()
        return BatchResult(loss.item(), num_correct, TP, TN, FP, FN, out, y)

    def _val_batch(self, batch):
        x, y = batch
        #x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)
        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1)).item()
            TP = torch.sum((out > 0) * (y == 1)).item()
            TN = torch.sum((out <= 0) * (y == 0)).item()
            FP = torch.sum((out > 0) * (y == 0)).item()
            FN = torch.sum((out <= 0) * (y == 1)).item()
        return BatchResult(loss.item(), num_correct, TP, TN, FP, FN, out, y)


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



def balance_dataset(X,y):
    X,y = np.array(X), np.array(y)
    h,w = X.shape[-2], X.shape[-1]
    ros = RandomOverSampler(random_state=42)
    X_flat = X.reshape(X.shape[0], -1)
    X_resampled, y_resampled = ros.fit_resample(X_flat, y)
    X_resampled = X_resampled.reshape(-1,13,3,h,w)
    return X_resampled, y_resampled



def delete_unused_columns(dxs:np.array):
    columns_with_only_zeros = np.all(dxs == 0, axis=0)
    # Get the indices of these columns
    indices_of_columns_with_only_zeros = np.where(columns_with_only_zeros)[0]
    filtered_array = np.delete(dxs, indices_of_columns_with_only_zeros, axis=1)
    return filtered_array



if __name__ == '__main__':
    device = 0
    #path_to_dataset = ["/work/scratch/td38heni/all"]
    #path_to_dataset = ["/work/home/td38heni/CinC_cleaned/Datahandling/test_data"] #testing server
    path_to_dataset = [r"C:\Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train\test_data"] #testing me
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/fine_tune/train_set"]
    yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    yolo = YOLO(yolo_model)

    num_of_samples=303#add config file for runs
    num_of_epochs=40
    batch_size=64
    path = os.getcwd() + "/data_preprocessed"
    #path = "/work/scratch/td38heni/CinC_cleaned"
    ds = ECG_cropped(path_to_dataset, get_signal=False, samples=num_of_samples, YOLO_path=yolo_model)
    #ds.dx, ds.img = np.load(os.path.join(path, "dxs.npy")),  np.load(os.path.join(path, "imgs.npy"))
    ##ds.dx = delete_unused_columns(ds.dx)
    ##ds_train, ds_val = torch.utils.data.random_split(ds, [0.8,0.2])
    ##dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)
    ##ds.img, ds.dx = balance_dataset(ds[ds_train.indices][0], ds[ds_train.indices][1])
    ##dl_train = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    ###dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ##model = get_model("ViT", num_classes=ds.dx.shape[-1])
    ##optimizer = optim.Adam(model.parameters(), lr=0.001)
    ##loss_fn = nn.CrossEntropyLoss()
    ##trainer = Ecg12LeadImageNetTrainerMulticlass(model=model, optimizer=optimizer, loss_fn=loss_fn,device=device)
    ##train_result = trainer.train(num_of_epochs=num_of_epochs, train_dataloader=dl_train, val_datloader=dl_val)
    ##model = model.eval()
    ##results = []
    ###for x,y in dl_val:
    #    y_pred =  model(x.to(device, dtype=torch.float))
    #    max_values, max_indices = torch.max(y_pred, dim=1)
    #    mask = y_pred == max_values.unsqueeze(1)
    #    result = mask.float()
    #    results.append(result.detach().cpu().numpy())
    #model_dict = copy.deepcopy(model.state_dict())
    #torch.save(model_dict, f'{path}/model_dx.pt')
    #plot_results(train_result.f1_score, train_result.loss, "model_dx_val", output_path=f"{path}/{name}")
    #plot_results(train_result.f1_score_val, train_result.loss_val, "model_dx_val", output_path=f"{path}/{name}")
    #df = pd.DataFrame(train_result._asdict())
    #df.to_csv(f'{path}}/train_result.csv')



