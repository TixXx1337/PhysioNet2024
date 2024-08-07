import copy
from Datahandling.Dataloader_withYOLO import ECG_Turned
from tqdm import tqdm
import numpy
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import TransformerDecoder, TransformerDecoderLayer

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
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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

        #self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.to_latent(x)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x has shape (batch_size, latent_dim)
        batch_size = x.shape[0]
        # Transform latent vector to match LSTM input
        hidden_init = self.latent_to_hidden(x).view(batch_size, self.seq_len, -1)
        lstm_out, _ = self.lstm(hidden_init)
        # Reshape to (batch_size * seq_len, hidden_dim) for the linear layer
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[2])
        output = self.fc(lstm_out)
        # Reshape back to (batch_size, seq_len, output_dim)
        output = output.view(batch_size, self.seq_len, -1)
        return output


class TransformerDecoderModel(nn.Module):
    def __init__(self, latent_dim, nhead, num_decoder_layers, dim_feedforward, output_dim, seq_len):
        super(TransformerDecoderModel, self).__init__()
        self.seq_len = seq_len
        self.latent_to_seq = nn.Linear(latent_dim, latent_dim * seq_len)
        decoder_layer = TransformerDecoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(latent_dim, output_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, latent_dim))

    def forward(self, x):
        # x has shape (batch_size, latent_dim)
        batch_size = x.shape[0]
        # Transform latent vector to match Transformer input
        seq_input = self.latent_to_seq(x).view(self.seq_len, batch_size, -1)
        seq_input = seq_input + self.positional_encoding.unsqueeze(1)  # Add positional encoding
        transformer_out = self.transformer_decoder(seq_input, seq_input)
        # Reshape to (seq_len * batch_size, latent_dim) for the linear layer
        transformer_out = transformer_out.view(-1, transformer_out.shape[2])
        output = self.fc(transformer_out)
        # Reshape back to (seq_len, batch_size, output_dim)
        output = output.view(self.seq_len, batch_size, -1)
        # Permute to get shape (batch_size, seq_len, output_dim)
        output = output.permute(1, 0, 2)
        return output

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


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class Ecg12Dignet(nn.Module):
    def __init__(self, encoder, decoder_short,decoder_long, parallel_short:int=12):
        super(Ecg12Dignet, self).__init__()
        #self.encoder = encoder
        #self.decoder = decoder
        #self.decoder_long = decoder_long
        self.encoders = []
        self.decoders = []
        self.parallel_short = parallel_short
        #for i in range(parallel_short+1): #last for long lead
        self.encoders = nn.ModuleList([copy.deepcopy(encoder) for i in range(parallel_short+1)])
        #for i in range(parallel_short):
        self.decoders = nn.ModuleList([copy.deepcopy(decoder_short) for i in range(parallel_short)])
        self.decoders.append(decoder_long)



    def forward(self, x):
        assert x.shape[1] == self.parallel_short+1, "Input needs to have the same number of Images as you have models"
        x = torch.cat([encoder(x[:, idx, :, :, :]) for idx, encoder in enumerate(self.encoders)], dim=1)
        x = torch.cat([decoder(x).flatten(start_dim=1) for idx, decoder in enumerate(self.decoders)], dim=1)
        return x




def get_model(model_name:str,decoder_name:str="LSTM", image_size:int=128, patch_size:int=32, dim:int=256,hidden_dim:int=32,droput:float=0.1):
    if model_name == 'ViT':
        encoder = ViT(image_size = image_size,
                    patch_size = patch_size,
                    dim = dim,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = droput,
                    emb_dropout = 0.1
                    )
    if decoder_name=="LSTM":
        short_decoder = LSTMDecoder(13*dim, hidden_dim, 1, 250, num_layers=3)
        long_decoder = LSTMDecoder(13*dim, hidden_dim, 1, 1000,num_layers=3)
    if decoder_name=="Transformer":
        short_decoder = TransformerDecoderModel(13 * dim,nhead=4,num_decoder_layers=3,dim_feedforward=hidden_dim, output_dim=1, seq_len=250)
        long_decoder = TransformerDecoderModel(13 * dim,nhead=4,num_decoder_layers=3,dim_feedforward=hidden_dim*4, output_dim=1, seq_len=1000)
    if decoder_name == "ANN":
        short_decoder = ANNDecoder(dim*13, [dim*12,dim*12//2,dim*12], 250)
        long_decoder = ANNDecoder(dim*13, [dim*12,dim*12//2,dim*12], 1000)
    model = Ecg12Dignet(encoder, short_decoder, long_decoder)
    return model



if __name__ == '__main__':
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/20images"]
    ##path_to_dataset = [r"C:\Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train\20images"]
    yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    ##yolo_model = r"C:\Users\Tizian Dege\PycharmProjects\CinC_cleaned\YOLO\LEAD_detector.pt"
    #dataset_image = ECG_Turned(path_to_dataset, samples=None, YOLO_path=yolo_model)
    model = get_model(model_name="ViT", decoder_name="Transformer", dim=2)
    x = torch.randn(5, 13, 3, 128, 128)
    y = model(x)
    #dl_train = torch.utils.data.DataLoader(dataset_image, batch_size=4, shuffle=False)
    #imgs, dx, signal = next(iter(dl_train))
    #imgs = imgs.to(dtype=torch.float)
    #model = model.to(dtype=torch.float)
    #y = model(imgs)