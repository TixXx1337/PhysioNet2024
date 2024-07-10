import abc
import math
import os
import sys
from tqdm import tqdm
import torch
from typing import NamedTuple, List
from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
import torch.nn.functional as F
from model.model_zoo import Ecg12ImageNet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from Datahandling.Dataloader import ECGImage_Class_Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt



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
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, num_of_epochs, train_dataloader, val_datloader):
        f1_score = []
        loss = []
        f1_score_val = []
        loss_val = []
        for epoch in range(num_of_epochs):
            running_loss_train = 0
            #TP, TN, FP, FN, num_correct = 0,0,0,0,0
            # Train loop
            progress_bar_train = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_train:.4f}')
            for idx, batch in enumerate(progress_bar_train):
                batch_result = self._train_batch(batch)
                running_loss_train += batch_result.loss
                avg_loss = running_loss_train / (idx + 1)
                progress_bar_train.set_description(f'Train Epoch {epoch + 1}/{num_of_epochs} - Loss: {avg_loss:.4f}')
                if idx==0:
                    y_pred_train = batch_result.out.detach().cpu().numpy()
                    y_true_train = batch_result.y.detach().cpu().numpy()
                else:
                    y_pred_train = np.vstack([y_pred_train, batch_result.out.detach().cpu().numpy()])
                    y_true_train = np.vstack([y_true_train, batch_result.y.detach().cpu().numpy()])
            y_pred_train[y_pred_train <= 0] = 0
            y_pred_train[y_pred_train > 0] = 1
            y_true_train[y_true_train <= 0] = 0
            y_true_train[y_true_train > 0] = 1
            f1_score.append(metrics.f1_score(y_true_train, y_pred_train, average='weighted'))
            loss.append(running_loss_train/len(train_dataloader))
            print(f"Epoch {epoch+1} Train: F1 Score: {f1_score[-1]}, Loss: {loss[-1]}\n")


            #Val loop


            running_loss_val = 0
            progress_bar_val = tqdm(val_datloader, desc=f'Val Epoch {epoch + 1}/{num_of_epochs} - Loss: {running_loss_val:.4f}')
            for idx, batch in enumerate(progress_bar_val):
                batch_result = self._val_batch(batch)
                running_loss_val += batch_result.loss
                avg_loss = running_loss_val / (idx + 1)
                progress_bar_val.set_description(f'Epoch {epoch + 1}/{num_of_epochs} - Loss: {avg_loss:.4f}')
                if idx==0:
                    y_pred_val = batch_result.out.detach().cpu().numpy()
                    y_true_val = batch_result.y.detach().cpu().numpy()
                else:
                    y_pred_val = np.vstack([y_pred_val, batch_result.out.detach().cpu().numpy()])
                    y_true_val = np.vstack([y_true_val, batch_result.y.detach().cpu().numpy()])
            y_pred_val[y_pred_val <= 0] = 0
            y_pred_val[y_pred_val > 0] = 1
            y_true_val[y_true_val <= 0] = 0
            y_true_val[y_true_val > 0] = 1
            f1_score_val.append(metrics.f1_score(y_true_val, y_pred_val, average='weighted'))
            loss_val.append(avg_loss)
            print(f"Epoch {epoch+1} Val: F1 Score: {f1_score_val[-1]}, Loss: {loss_val[-1]}\n")
        return FitResult(f1_score, loss, f1_score_val, loss_val)


    def _train_batch(self, batch):
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
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
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
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


def plot_results(f1_score:List,loss:List, plot_name:str)->None:
    plt.plot(range(len(f1_score)), f1_score,label='F1_score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.title('F1 Score')
    plt.savefig(f'F1_score_{plot_name}.png', transparent=True)
    plt.close()
    plt.plot(range(len(loss)), loss,label='F1_score')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.savefig(f'Loss_{plot_name}.png', transparent=True)
    plt.close()



if __name__ == '__main__':
    device = 0
    path_to_dataset = ["/work/scratch/td38heni/all"]
    hidden_channels = [8, 16, 32]
    kernel_sizes = [3, 3, 5]
    model = Ecg12ImageNet(in_channels=1, hidden_channels=hidden_channels, kernel_sizes=kernel_sizes, in_h=512, in_w=512,
                 fc_hidden_dims=[128], dropout=0.2, stride=1, dilation=1, batch_norm=True, num_of_classes=11).to(device, dtype=torch.float)
    ds = ECGImage_Class_Dataset(path_to_dataset, get_image=True, get_dx=True, get_signal=False)
    ds_train, dl_val, _ = torch.utils.data.random_split(ds, [0.8,0.2])
    del ds
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
    dl_val = torch.utils.data.DataLoader(dl_val, batch_size=1, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss().to(device, dtype=torch.float)
    #x,y = next(iter(dl))
    #x = x.transpose(1, 2).transpose(1, 3).to(device, dtype=torch.float)
    #y = y.to(device, dtype=torch.float)
    trainer = Ecg12LeadImageNetTrainerMulticlass(model=model, optimizer=optimizer, loss_fn=loss_fn,device=device)
    f1_score, loss, f1_score_val, loss_val = trainer.train(num_of_epochs=15, train_dataloader=dl_train, val_datloader=dl_val)
    torch.save(model.state_dict(), "model_dx_all.pt")
    plot_results(f1_score, loss, "train")
    plot_results(f1_score_val, loss_val, "val")
    #fit_result = trainer.fit(dl, dl, num_epochs=10)


