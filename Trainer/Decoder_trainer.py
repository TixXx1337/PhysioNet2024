import torch
from torch.utils.data import Dataset
import pandas as pd
import helper_code
import glob
import os
from pathlib import Path
import copy
import numpy as np
from sequitur.models import LSTM_AE
from sequitur.models import CONV_LSTM_AE
from sequitur.models import LINEAR_AE
from sequitur import quick_train
from sklearn import preprocessing







def get_class(row) -> str:
    path = Path(row["header"])
    #row["image"] = os.path.join(path.parent, f'{row["basename"]}-0.png')
    with open(row["header"], "r") as f:
        for line in f.readlines():
            if "#" in line:
                line = line.replace("#", "")
                line = line.strip()
                label, definition = line.split(":")
                row[label.strip()] = definition.strip()
    row["image"] = os.path.join(path.parent, row["Image"])
    return row


class ECG_12Lead_Dig_Dataset(Dataset):
    def __init__(self, path_to_dataset=None, transform=None, verbose:bool=False):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        if type(path_to_dataset) is list:
            df_dataset_all = []
            for path in path_to_dataset:
                header = glob.glob(f"{path}/*.hea")
                df_dataset = pd.DataFrame({"header": header})
                df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
                df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
                df_dataset = df_dataset.apply(func=get_class, axis=1)
                df_dataset_all.append(df_dataset)
            df_dataset_all = pd.concat(df_dataset_all, ignore_index=True)
        else:
            header = glob.glob(f"{path_to_dataset}/*.hea")
            df_dataset = pd.DataFrame({"header": header})
            df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
            df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
            df_dataset = df_dataset.apply(func=get_class, axis=1)
            df_dataset_all = df_dataset
        self.data = df_dataset_all
        self.classes = {"Normal":0, "Abnormal":1}
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "Dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)
        if transform == "normalize":
            self.transform = transform
            self.scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        self.signals = []
        for index,data in self.data.iterrows():
            path = Path(data["header"]).parent
            signal = helper_code.load_signal(os.path.join(path, data["basename"]))
            signal = pd.DataFrame(signal[0])
            long_lead = copy.deepcopy(signal[1])
            short_leads = []
            for i in range(4):
                short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
            short_leads = pd.concat(short_leads, axis=1)
            short_leads, long_lead = np.array(short_leads).T, np.array(long_lead).reshape(-1, 1).T
            ret = np.vstack((short_leads.reshape(3, 1000), long_lead)).reshape(-1, 1)
            if self.transform is not None:
                ret = self.scaler.fit_transform(ret)
            self.signals.append(ret)

        self.signals = np.array(self.signals)





    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #path = Path(self.data.iloc[idx]["header"]).parent
        #signal = helper_code.load_signal(os.path.join(path, self.data.iloc[idx]["basename"]))
        #signal = pd.DataFrame(signal[0])
        #long_lead = copy.deepcopy(signal[1])
        #short_leads = []
        #for i in range(4):
        #    short_leads.append(signal[np.arange(i*3, (i+1)*3)].dropna().reset_index(drop=True))
        #short_leads = pd.concat(short_leads, axis=1)
        #short_leads, long_lead = np.array(short_leads).T, np.array(long_lead).reshape(-1,1).T
        #ret = np.vstack((short_leads.reshape(3, 1000), long_lead)).reshape(-1, 1)
        #if self.transform is not None:
        #    ret = self.scaler.fit_transform(ret)
        return self.signals[idx]

    def load_image(self, image):
        """
        Simple Functions that
        :param image:
        :return:
        """
        image = image.resize((512, 512))
        image = np.array(image)
        image = image.reshape(512, 512, 1)
        image=image/255
        return image


def plot_ecg(ecg_signal:np.array):
    return 0


if __name__ == '__main__':
    model = LSTM_AE(
        input_dim=1000,
        encoding_dim=250,
        h_dims=[64],
        h_activ=None,
        out_activ=None
    )
    dataset_path = "ptb-xl/delete/"
    dataset = ECG_12Lead_Dig_Dataset(path_to_dataset=dataset_path)
    trains_set = []
    for data in dataset:
        trains_set.append(torch.tensor(data.flatten(), dtype=torch.float).to("cuda"))

    encoder, decoder, whats, that = quick_train(LINEAR_AE, trains_set,verbose=True, encoding_dim=250)


