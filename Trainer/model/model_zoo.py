import torch
import torch.nn as nn
import math
import time
import torch
import torch.nn as nn
import math
import time


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


# A simple but versatile d1 "deconvolution" neural net
class DeConvNet1d(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, out_channels: int, out_kernel: int,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False, output_padding=0):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.ConvTranspose1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                             stride=stride, dilation=dilation, output_padding=output_padding))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2))

            layer_in_channels = layer_out_channels

        layers.append(nn.ConvTranspose1d(layer_in_channels, out_channels, out_kernel, stride, dilation))
        layers.append(torch.nn.Tanh()) #added to have negative outputs
        self.dcnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dcnn(x)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12LeadNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


class Ecg12LeadMultiClassNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2):
        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * calc_out_length(short_input_length, short_kernel_lengths,
                                                             short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * calc_out_length(long_input_length, long_kernel_lengths,
                                                           long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)


class Ecg12ImageNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        self.cnn = ConvNet2d(in_channels, hidden_channels, kernel_sizes, dropout, stride, dilation, batch_norm)
        self.n = torch.nn.Sigmoid()
        out_channels = hidden_channels[-1]
        out_h = calc_out_length(in_h, kernel_sizes, stride, dilation)
        out_w = calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w

        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape((x.shape[0], -1))
        out = self.fc(out)
        return self.n(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Ecg12ImageToSignalNet(nn.Module):
    def __init__(self, in_channels: int, deconv_in_channels: int, in_h: int, in_w: int,
                 conv_hidden_channels: list, conv_kernel_sizes: list,
                 deconv_hidden_channels_short: list, deconv_kernel_lengths_short: list,
                 deconv_hidden_channels_long: list, deconv_kernel_lengths_long: list,
                 conv_dropout=None, conv_stride=1, conv_dilation=1, conv_batch_norm=False,
                 deconv_dropout_short=None, deconv_stride_short=2, deconv_dilation_short=1,
                 deconv_batch_norm_short=False, deconv_out_kernel_short=5,
                 deconv_dropout_long=None, deconv_stride_long=3, deconv_dilation_long=1,
                 deconv_batch_norm_long=False, deconv_out_kernel_long=5,output_padding=0,
                 fc_hidden_dims=(), l_out_long=1000, l_out_short=250, short_leads=12, long_leads=1):
        super().__init__()

        self.deconv_in_channels = deconv_in_channels
        self.l_out_short = l_out_short
        self.l_out_long = l_out_long

        self.l_in_short = calc_out_length(l_out_short, deconv_kernel_lengths_short + [deconv_out_kernel_short],
                                          deconv_stride_short, deconv_dilation_short)
        self.l_in_long = calc_out_length(l_out_long, deconv_kernel_lengths_long + [deconv_out_kernel_long],
                                         deconv_stride_long, deconv_dilation_long)

        self.short_leads_latent_dim = self.l_in_short * deconv_in_channels
        self.long_leads_latent_dim = self.l_in_long * deconv_in_channels
        latent_dim = self.short_leads_latent_dim + self.long_leads_latent_dim

        self.cnn2d = Ecg12ImageNet(in_channels, hidden_channels=conv_hidden_channels, kernel_sizes=conv_kernel_sizes,
                                   in_h=in_h, in_w=in_w, fc_hidden_dims=fc_hidden_dims, dropout=conv_dropout,
                                   stride=conv_stride, dilation=conv_dilation, batch_norm=conv_batch_norm,
                                   num_of_classes=latent_dim)
        self.ReLu = nn.ReLU()
        self.dcnn1d_short = DeConvNet1d(deconv_in_channels, hidden_channels=deconv_hidden_channels_short,
                                        out_channels=short_leads, out_kernel=deconv_out_kernel_short,
                                        kernel_lengths=deconv_kernel_lengths_short,
                                        dropout=deconv_dropout_short, stride=deconv_stride_short,
                                        dilation=deconv_dilation_short, batch_norm=deconv_batch_norm_short, output_padding=output_padding)
        self.dcnn1d_long = DeConvNet1d(deconv_in_channels, hidden_channels=deconv_hidden_channels_long,
                                       out_channels=long_leads, out_kernel=deconv_out_kernel_long,
                                       kernel_lengths=deconv_kernel_lengths_long,
                                       dropout=deconv_dropout_long, stride=deconv_stride_long,
                                       dilation=deconv_dilation_long, batch_norm=deconv_batch_norm_long, output_padding=output_padding)

    def forward(self, x):
        out = self.cnn2d(x)
        out = self.ReLu(out)

        latent_long = out[:, 0:self.long_leads_latent_dim]
        latent_long = latent_long.reshape((x.shape[0], self.deconv_in_channels, self.l_in_long))
        latent_short = out[:, self.long_leads_latent_dim:self.long_leads_latent_dim + self.short_leads_latent_dim]
        latent_short = latent_short.reshape((x.shape[0], self.deconv_in_channels, self.l_in_short))

        out_short = self.dcnn1d_short(latent_short)
        out_long = self.dcnn1d_long(latent_long)
        return out_short, out_long

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NewNet(nn.Module):

    def __init__(self, in_channels: int, in_h: int, in_w: int,
                 conv_hidden_channels: list, conv_kernel_sizes: list,
                 conv_dropout=None, conv_stride=1, conv_dilation=1, conv_batch_norm=False,
                 fc_hidden_dims=(), l_out_long=1000, l_out_short=250, short_leads=12, long_leads=1):
        super().__init__()
        self.short_leads = short_leads
        self.long_leads = long_leads
        self.l_out_short = l_out_short
        self.l_out_long = l_out_long
        self.out_dim_short = l_out_short * short_leads
        self.out_dim_long = l_out_long * long_leads
        self.out_dim = self.out_dim_long + self.out_dim_short

        self.cnn2d = Ecg12ImageNet(in_channels, hidden_channels=conv_hidden_channels, kernel_sizes=conv_kernel_sizes,
                                   in_h=in_h, in_w=in_w, fc_hidden_dims=fc_hidden_dims, dropout=conv_dropout,
                                   stride=conv_stride, dilation=conv_dilation, batch_norm=conv_batch_norm,
                                   num_of_classes=self.out_dim)

        self.cnn1d_short = nn.Conv1d(short_leads, short_leads, 5, 1, 2, 1, 1, True)
        self.cnn1d_long = nn.Conv1d(long_leads, long_leads, 5, 1, 2, 1, 1, True)

        self.s = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.cnn2d(x)
        out_long = out[:, 0:self.out_dim_long].reshape(batch_size, self.long_leads, self.l_out_long)
        out_short = out[:, self.out_dim_long:].reshape(batch_size, self.short_leads, self.l_out_short)
        out_long = self.cnn1d_long(out_long)
        out_short = self.cnn1d_short(out_short)
        return self.s(out_short), self.s(out_long)

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


def calc_out_length_deconv(l_in: int, kernel_lengths: list, out_kernel: int, stride: int, dilation: int,
                           padding=1, output_padding=0):
    l_out = l_in
    for kernel in kernel_lengths:
        l_out = (l_out - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    l_out = (l_out - 1) * stride - 2 * padding + dilation * (out_kernel - 1) + 1
    return l_out



if __name__ == '__main__':
    hidden_channels = [8, 16, 32, 64]
    kernel_sizes = [3, 3, 5, 5]
    deconv_hidden_channels_short = [2,4,8]
    deconv_kernel_lengths_short = [3] * len(deconv_hidden_channels_short)
    deconv_hidden_channels_long = [2,4,8,16]
    deconv_kernel_lengths_long = [5] * len(deconv_hidden_channels_long)
    ds = Dataloader.ECG_12Lead_Dataset("/work/home/td38heni/CinC_cleaned/Trainer/20images")
    device = 0
    model = Ecg12ImageToSignalNet(in_channels=1, deconv_in_channels=len(hidden_channels), in_h=512, in_w=512,
                 conv_hidden_channels=hidden_channels, conv_kernel_sizes= kernel_sizes,
                 deconv_hidden_channels_short= deconv_hidden_channels_short, deconv_kernel_lengths_short= deconv_kernel_lengths_short,
                 deconv_hidden_channels_long= deconv_hidden_channels_long, deconv_kernel_lengths_long= deconv_kernel_lengths_long,
                 fc_hidden_dims=[128], l_out_long=1220, l_out_short=280, short_leads=12, long_leads=1).to(device) #TODO change 1Ddeconv for better output!

    ds = Dataloader.ECG_12Lead_Dataset("C:\\Users\\Tizian Dege\\PycharmProjects\\DeTECRohr\\output")
    images = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    x,y = next(iter(images))
    image = x.to(device, dtype=torch.float)
    image = image.transpose(1,2).transpose(1,3)
    model.eval()
    out = model(image)
    short_out = out[0]
    y_out = y[0]
