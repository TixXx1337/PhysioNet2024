import copy
import os
import sys
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from cnn import Ecg12ImageNet

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ANNDecoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ANNDecoder, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))
        layers.append(nn.Sigmoid())  # Ensures output is between 0 and 1

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Ecg12Dxnet(nn.Module):
    def __init__(self, encoder, num_classes:int=11, parallel:int=13,use_whole_image:bool=False):
        super(Ecg12Dxnet, self).__init__()
        self.encoders = []
        self.parallel = parallel
        self.layers =[]
        self.whole_image = use_whole_image
        #for i in range(parallel): #last for long lead
        self.encoders = nn.ModuleList([copy.deepcopy(encoder) for i in range(parallel)])
        self.layers.append(nn.Linear(num_classes*parallel, num_classes))
        self.layers.append(nn.Softmax(dim=1))
        self.out = nn.Sequential(*self.layers)


    def forward(self, x):
        assert x.shape[1] == self.parallel, "Input needs to have the same number of Images as you have models"
        x = torch.cat([encoder(x[:, idx, :, :, :]) for idx, encoder in enumerate(self.encoders)], dim=1)
        if self.training:
            return self.out[:-1](x)
        return self.out(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes:int):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*53*68, 512)          # Adjusted  to input size (425, 550)
        self.fc2 = nn.Linear(512, self.num_classes)  # 11 output classes for multilabel classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, x.shape[1] * x.shape[2]* x.shape[3])
        x = torch.relu(self.fc1(x))
        if self.training:
            return self.fc2(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for multilabel classification
        return x


def get_simple_model(model_name:str,num_classes:int=11, image_size:int=(640,1024), patch_size:int=32, dim:int=256,droput:float=0.1):
    if model_name == 'ViT':
        model = ViT(image_size = image_size[1],
                    patch_size = patch_size,
                    dim = dim,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = droput,
                    emb_dropout = 0.1,
                    num_classes=num_classes
                    )

    elif model_name == 'CNN':
        #hidden_channels = [64, 32, 16, 8, 4]
        #kernel_sizes = [9, 7, 5, 3, 3]
        hidden_channels = [3,16,32,64]
        kernel_sizes = [3, 3, 3, 3]
        model = Ecg12ImageNet(in_channels=3, hidden_channels=hidden_channels, kernel_sizes=kernel_sizes, in_h= image_size[0], in_w=image_size[1],
                 fc_hidden_dims=[512],  stride=2, dilation=1, batch_norm=True, num_of_classes=num_classes)

    elif model_name == 'CNN_small':
        model = SimpleCNN(num_classes=num_classes)
    return model


def get_model(model_name:str,num_classes:int=11, image_size:int=128, patch_size:int=32, dim:int=256,droput:float=0.1):
    if model_name == 'ViT':
        encoder = ViT(image_size = image_size,
                    patch_size = patch_size,
                    dim = dim,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = droput,
                    emb_dropout = 0.1,
                    num_classes=num_classes
                    )
    #elif model_name == 'CNN':
    #    encoder = CNN(num_classes=num_classes)
    return Ecg12Dxnet(encoder=encoder, num_classes=num_classes)



if __name__ == '__main__':
    #path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/20images"]
    ##path_to_dataset = [r"C:\\Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train\\20images"]
    #yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    ##yolo_model = r"C:\\Users\Tizian Dege\PycharmProjects\CinC_cleaned\YOLO\LEAD_detector.pt"
    #dataset_image = ECG_Turned(path_to_dataset, samples=None, YOLO_path=yolo_model)
    #model = get_model(model_name="CNN")
    model = get_simple_model(model_name="VIT",image_size=(512,512), num_classes=11)
    x = torch.randn(5, 3, 512, 512)
    y = model(x)
    #dl_train = torch.utils.data.DataLoader(dataset_image, batch_size=4, shuffle=False)
    #imgs, dx, signal = next(iter(dl_train))
    #imgs = imgs.to(dtype=torch.float)
    #model = model.to(dtype=torch.float)
    #y = model(imgs)
