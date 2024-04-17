import copy
import sys
sys.path.append("Digitized-vs-Image-ECG-classification")
from torch.utils.data import Dataset
import glob
import pandas as pd
import torch
import numpy as np
import os
from PIL import Image
from pathlib import Path
import helper_code


class ECG_12Lead_Dataset(Dataset):
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




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]["image"]).convert('L')
        image = self.load_image(image)
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signal(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i*3, (i+1)*3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        return image ,np.array(long_lead), np.array(short_leads).T

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



class ECG_Class_Dataset(Dataset):
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




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]["image"]).convert('L')
        image = self.load_image(image)
        dx = self.data.iloc[idx]["Dx"]
        dx = self.classes[dx.strip()]
        return image, dx

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



def prepare_dataloader(ds_train, ds_val, batch_size:int=16, num_workers:int=4):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val




#if __name__ == '__main__':
#    path_to_dataset = "/home/tdege/DeTECRohr/output"
#    ds_train = ECG_12Lead_Dataset(path_to_dataset)
#    y = ds_train[0]
#

