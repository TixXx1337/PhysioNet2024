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
    def __init__(self, path_to_dataset=None, transform=None,flatten:bool=False, verbose:bool=False):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        self.flatten = flatten
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
        dx = self.data.iloc[idx]["Labels"]
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i*3, (i+1)*3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        if self.flatten:
            short_leads, long_lead = np.array(short_leads).T, np.array(long_lead).reshape(-1, 1).T
            return torch.tensor(image, dtype=torch.float), np.vstack((short_leads.reshape(3, 1000), long_lead)).reshape(-1, 1)
        return image,(np.array(short_leads).T, np.array(long_lead)), dx

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



class ECGImage_Class_Dataset(Dataset):
    def __init__(self, path_to_dataset:list=None, transform=None, verbose:bool=False, get_image:bool=True):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        self.get_image = get_image
        df_dataset_all = []
        for path in path_to_dataset:
            header = glob.glob(f"{path}/*.hea")
            df_dataset = pd.DataFrame({"header": header})
            df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
            df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
            df_dataset = df_dataset.apply(func=get_class, axis=1)
            df_dataset_all.append(df_dataset)
        df_dataset_all = pd.concat(df_dataset_all, ignore_index=True)
        self.data = df_dataset_all
        self.classes = {}
        all_labels = []
        for entry in self.data["Labels"].drop_duplicates():
            all_labels += entry.split(",")
        all_labels = [s.replace(" ", "") for s in all_labels if s.strip() != ""]
        all_labels = list(set(all_labels))
        for class_idx, class_name in enumerate(all_labels):
            self.classes.update({class_name:class_idx})
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "Labels"]:
                data = self.data[category].value_counts().to_dict()
                print(data)




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        dx = self.data.iloc[idx]["Labels"]
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i*3, (i+1)*3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        if self.get_image:
            image = Image.open(self.data.iloc[idx]["image"]).convert('L')
            image = self.load_image(image)
            return np.array(short_leads).T, np.array(long_lead), dx, image
        return np.array(short_leads).T, np.array(long_lead), dx

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


class ECG_Lead(Dataset):
    def __init__(self, path_to_dataset:list=None, transform=None, verbose:bool=False):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        df_dataset_all = []
        for path in path_to_dataset:
            header = glob.glob(f"{path}/*.hea")
            df_dataset = pd.DataFrame({"header": header})
            df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
            df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
            df_dataset = df_dataset.apply(func=get_class, axis=1)
            df_dataset_all.append(df_dataset)
        df_dataset_all = pd.concat(df_dataset_all, ignore_index=True)
        self.data = df_dataset_all
        self.classes = {}
        #self.data["Labels"] = self.data["Labels"].apply(func=str.split, args=",")
        all_labels = []
        for entry in self.data["Labels"].drop_duplicates():
            all_labels += entry.split(",")
        all_labels = [s.replace(" ", "") for s in all_labels if s.strip() != ""]
        all_labels = list(set(all_labels))
        for class_idx, class_name in enumerate(all_labels):
            self.classes.update({class_name:class_idx})
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "Dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i*3, (i+1)*3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        return np.array(short_leads).T, np.array(long_lead)



def get_class(row) -> str:
    path = Path(row["header"])
    row["Image_name"] = os.path.join(path.parent, f'{row["basename"]}-0.png')
    with open(row["header"], "r") as f:
        for line in f.readlines():
            if "#" in line:
                line = line.replace("#", "")
                line = line.strip()
                label, definition = line.split(":")
                row[label.strip()] = definition.strip()
    row["image"] = os.path.join(path.parent, row["Image_name"])
    return row



def prepare_dataloader(ds_train, ds_val, batch_size:int=16, num_workers:int=4):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,num_workers=num_workers, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val




if __name__ == '__main__':
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/Train/test_data"]
    #dataset = ECG_Lead(path_to_dataset)
    #signal = dataset[0]
    dataset_image = ECGImage_Class_Dataset(path_to_dataset, get_image=True)
    dataset_image[0]




