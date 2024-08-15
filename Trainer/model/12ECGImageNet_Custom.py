import torch
import torch.nn as nn
import math
import Dataloader
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import numpy

# A simple but versatile d1 convolutional neural net
class ConvNet1d(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# A simple but versatile d2 convolutional neural net
class ConvNet2d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_sizes: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_sizes[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12ImageNet_Custom(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        self.cnn = ConvNet2d(in_channels, hidden_channels, kernel_sizes, dropout, stride, dilation, batch_norm)

        out_channels = hidden_channels[-1]
        out_h = calc_out_length(in_h, kernel_sizes, stride, dilation)
        out_w = calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w
        in_dim = 10000
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape((x.shape[0], -1))
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def calc_out_length(l_in: int, kernel_lengths: list, stride: int, dilation: int, padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = math.floor((l_out + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    return l_out


class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.output_dim, 1)  # Repeat latent vector to match output length
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only take the last output
        return out



class ImageToSignalModel(nn.Module):
    def __init__(self, cnn, decoder):
        super(ImageToSignalModel, self).__init__()
        self.cnn = cnn
        self.decoder = decoder

    def forward(self, x):
        latent_vector = self.cnn(x)
        reconstructed_signal = self.decoder(latent_vector)
        return reconstructed_signal


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader) as tepoch:
            for images, signals in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                images = images.float().to(1)  # Ensure images are in float32
                images = images.transpose(1, 3)
                signals = signals.float().to(1).flatten()  # Ensure signals are in float32
                optimizer.zero_grad()
                outputs = model(images).flatten()
                loss = criterion(outputs, signals)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                tepoch.set_postfix(loss=loss.item())


    print('Training complete')


if __name__ == '__main__':
    hidden_channels = [8, 16]
    kernel_sizes = [3, 3]
    deconv_hidden_channels_short = [2]
    deconv_kernel_lengths_short = [3] * len(deconv_hidden_channels_short)
    deconv_hidden_channels_long = [2,]
    deconv_kernel_lengths_long = [5] * len(deconv_hidden_channels_long)
    fc_hidden_dims = [500, 128]
    #model_cnn = Ecg12ImageNet_Custom( in_channels=1, hidden_channels=hidden_channels, kernel_sizes=kernel_sizes, in_h=512, in_w=512,
    #             fc_hidden_dims =fc_hidden_dims, dropout=None, stride=1, dilation=1, batch_norm=False)
    model_cnn = torchvision.models.vgg16(weights=True)
    model_cnn.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    x = torch.rand(5,1,512,512)
    y = model_cnn(x)
    hidden_dim = 256
    num_layers = 2
    decoder = RNNDecoder(input_dim=1000, output_dim=4000, hidden_dim=hidden_dim, num_layers=num_layers)
    dataset = Dataloader.ECG_12Lead_Dataset("/home/tdege/DeTECRohr/ptb-xl/test", flatten=True)
    dl_train = torch.utils.data.DataLoader(dataset[0:100], batch_size=8, num_workers=1, shuffle=True)
    #dataset_size = len(dataset)
    #indices = list(range(dataset_size))
    #split = int(np.floor(0.2 * dataset_size))
    model = ImageToSignalModel(cnn=model_cnn, decoder=decoder).to(1)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dl_train, criterion, optimizer)