import abc
import os
import sys
import tqdm
import torch

from typing import NamedTuple, List
from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path


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
    out : float
    y : int


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
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]




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


class EcgImageToDigitizedTrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = (y[0].to(self.device, dtype=torch.float), y[1].to(self.device, dtype=torch.float))
        batch_size = y[0].shape[0]
        dim_ratio = y[0].nelement() / y[1].nelement()

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = dim_ratio * self.loss_fn(out[0], y[0]) + self.loss_fn(out[1], y[1])
        loss.backward()
        self.optimizer.step()

        num_correct = batch_size * (
                    torch.sum(torch.abs(out[0] - y[0]) < 0.01) + torch.sum(torch.abs(out[1] - y[1]) < 0.01)) \
                      / (torch.numel(y[0]) + torch.numel(y[1]))

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = (y[0].to(self.device, dtype=torch.float), y[1].to(self.device, dtype=torch.float))
        batch_size = y[0].shape[0]

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out[0], y[0]) + self.loss_fn(out[1], y[1])
            num_correct = \
                batch_size * (
                        torch.sum(torch.abs(out[0] - y[0]) < 0.01) + torch.sum(torch.abs(out[0] - y[0]) < 0.01)) \
                / (torch.numel(y[0]) + torch.numel(y[1]))

        return BatchResult(loss.item(), num_correct.item())









