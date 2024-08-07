import copy
import sys
from tqdm import tqdm

sys.path.append("Digitized-vs-Image-ECG-classification")
from torch.utils.data import Dataset
import glob
import pandas as pd
import torch
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from PIL import Image
from pathlib import Path
import helper_code
import cv2
from imblearn.over_sampling import RandomOverSampler
import h5py
import math


class ECG_12Lead_Dataset(Dataset):
    def __init__(self, path_to_dataset=None, transform=None, flatten: bool = False, verbose: bool = False):
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
        self.classes = {"Normal": 0, "Abnormal": 1}
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
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        if self.flatten:
            short_leads, long_lead = np.array(short_leads).T, np.array(long_lead).reshape(-1, 1).T
            return torch.tensor(image, dtype=torch.float), np.vstack((short_leads.reshape(3, 1000), long_lead)).reshape(
                -1, 1)
        return image, (np.array(short_leads).T, np.array(long_lead)), dx

    def load_image(self, image):
        """
        Simple Functions that
        :param image:
        :return:
        """
        image = image.resize((512, 512))
        image = np.array(image)
        image = image.reshape(512, 512, 1)
        image = image / 255
        return image


class ECGImage_Class_Dataset(Dataset):
    def __init__(self, path_to_dataset: list = None, transform=None, verbose: bool = False,
                 get_image: bool = True, get_dx: bool = True, use_single_class: bool = True, get_signal: bool = False,
                 samples: int = None, split_data: int = None, output: str = ""):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        self.get_image = get_image
        self.get_dx = get_dx
        self.get_signal = get_signal
        self.split_data = split_data
        self.path_to_dataset = path_to_dataset
        self.output = output
        self.use_single_class = use_single_class
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
        self.classes = {'NORM': 0, 'STTC': 1, 'PAC': 2, 'Old MI': 3, 'HYP': 4, 'TACHY': 5, 'CD': 6, 'BRADY': 7,
                        'AFIB/AFL': 8, 'PVC': 9, 'Acute MI': 10}
        self.reversed_classed = {v: k for k, v in self.classes.items()}
        #all_labels = []
        #for entry in self.data["Labels"].drop_duplicates():
        #    all_labels += entry.split(",")
        #all_labels = [s.replace(" ", "") for s in all_labels if s.strip() != ""]
        #all_labels = list(set(all_labels))
        #for class_idx, class_name in enumerate(all_labels):
        #    self.classes.update({class_name:class_idx})
        self.img, self.dx, self.signal = [], [], []
        if self.split_data is not None:  #created preprocessed data
            h5py_output = f"{self.output}/split_data"
            if not os.path.isdir(h5py_output):
                os.makedirs(h5py_output)
            split_df = np.array_split(self.data, math.ceil(len(self.data) / self.split_data))
            for df_idx, data in enumerate(split_df):
                with h5py.File(f'{h5py_output}/ecg_dataset_split{df_idx}.h5', 'w') as h5f:
                    num_images = len(data)
                    # Create datasets for images and labels
                    h5f.create_dataset('images', shape=(num_images, 512, 512, 1), dtype=np.float32)
                    h5f.create_dataset('labels', shape=(num_images, 11), dtype=np.int32)
                    h5f.create_dataset("long_lead", shape=(num_images, 1000), dtype=np.float32)
                    h5f.create_dataset("short_lead", shape=(num_images, 12, 250), dtype=np.float32)
                    for i, (_, row) in enumerate(data.iterrows()):
                        image, labels, (short_lead, long_lead) = self.load_data_split(row)
                        h5f['images'][i] = image
                        h5f['labels'][i] = labels
                        h5f['long_lead'][i] = long_lead
                        h5f['short_lead'][i] = short_lead
                    print(f'Created {h5py_output}/ecg_dataset_split{df_idx}.h5 successfully')

        if samples is not None:
            self.data = self.data.sample(samples)
        progress_bar = tqdm(self.data.iterrows())
        for idx, entry in enumerate(progress_bar):
            img, dx, signal = self.load_data(idx)
            self.img.append(img)
            self.dx.append(dx)
            self.signal.append(signal)
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, dx, signal = self.img[idx], self.dx[idx], self.signal[idx]
        return [value for value in [image, dx, signal] if value is not None]

    def load_data_split(self, row):
        path = Path(row["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, row["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = (np.array(short_leads).T, np.array(long_lead))
        #load image
        image = self.load_image(row["image"])
        #dx
        dxs = [0] * len(self.classes)
        for dx in row["dx"]:
            dxs[self.classes[dx]] = 1
            break
        dxs = np.array(dxs)
        return image, dxs, signal

    def load_data(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal, dx, image = None, None, None
        if self.get_signal:
            signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
            signal = pd.DataFrame(signal[0])
            long_lead = copy.deepcopy(signal[1])
            short_leads = []
            for i in range(4):
                short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
            short_leads = pd.concat(short_leads, axis=1)
            signal = (np.array(short_leads).T, np.array(long_lead))
        if self.get_image:
            #image = Image.open(self.data.iloc[idx]["image"]).convert('L')
            image = self.load_image(self.data.iloc[idx]["image"])
        if self.get_dx:
            dxs = [0] * len(self.classes)
            for dx in self.data.iloc[idx]["dx"]:
                dxs[self.classes[dx]] = 1
                break
            dxs = np.array(dxs)
        return image, dxs, signal

    def load_image(self, image_path):
        """
        Simple Functions that
        :param image:
        :return:
        """
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 1: Noise Reduction
        # Apply Gaussian Blur to reduce noise
        denoised_image = cv2.GaussianBlur(image, (3, 3), 100)

        # Resize the image to 512x512
        resized_image = cv2.resize(denoised_image, (512, 512))
        resized_image = resized_image / 255
        resized_image.resize(512, 512, 1)
        return resized_image


class ECG_Lead(Dataset):
    def __init__(self, path_to_dataset: list = None, transform=None, verbose: bool = False):
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
            self.classes.update({class_name: class_idx})
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
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        return np.array(short_leads).T, np.array(long_lead)


class ECG_Turned(Dataset):
    def __init__(self, path_to_dataset: list = None, transform=None, verbose: bool = False, samples: int = None):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.transform = transform
        self.path_to_dataset = path_to_dataset
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
        self.classes = {'NORM': 0, 'STTC': 1, 'PAC': 2, 'Old MI': 3, 'HYP': 4, 'TACHY': 5, 'CD': 6, 'BRADY': 7,
                        'AFIB/AFL': 8, 'PVC': 9, 'Acute MI': 10}
        self.reversed_classed = {v: k for k, v in self.classes.items()}
        self.img, self.dx, self.signal = [], [], []

        if samples is not None:
            self.data = self.data.sample(samples)
        progress_bar = tqdm(self.data.iterrows())
        for idx, entry in enumerate(progress_bar):
            img, dx, signal = self.load_data(idx)
            self.img.append(img)
            self.dx.append(dx)
            self.signal.append(signal)
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, dx, signal = self.img[idx], self.dx[idx], self.signal[idx]
        return [value for value in [image, dx, signal] if value is not None]



    def load_data(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal, dx, image = None, None, None
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = (np.array(short_leads).T, np.array(long_lead))
        #image = Image.open(self.data.iloc[idx]["image"]).convert('L')
        image = self.load_image(self.data.iloc[idx]["image"])
        dxs = [0] * len(self.classes)
        for dx in self.data.iloc[idx]["dx"]:
            dxs[self.classes[dx]] = 1
            break
        dxs = np.array(dxs)
        return image, dxs, signal

    def load_image(self, image_path):
        """
        Simple Functions that
        :param image:
        :return:
        """
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 1: Noise Reduction
        # Apply Gaussian Blur to reduce noise
        denoised_image = cv2.GaussianBlur(image, (3, 3), 100)

        # Resize the image to 512x512
        resized_image = cv2.resize(denoised_image, (512, 512))
        resized_image = resized_image / 255
        resized_image.resize(512, 512, 1)
        return resized_image


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
    row["dx"] = [dx.strip() for dx in row["Labels"].split(",") if dx != ""]
    return row




if __name__ == '__main__':
    #äpath_to_dataset = [r"C:\Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train\20images"]
    path_to_dataset = ["/work/scratch/td38heni/all"]
    path_to_dataset = ["/work/scratch/td38heni/train_set"]
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/20images"]
    #dataset = ECG_Lead(path_to_dataset)
    #signal = dataset[0]
    dataset_image = ECG_Turned(path_to_dataset, samples=None)
    #hi = dataset_image[0]
    #X,y = dataset_image.img, dataset_image.dx
    #ros = RandomOverSampler(random_state=42)
    #X,y = np.array(X), np.array(y)
    #images_flat = X.reshape(X.shape[0], -1)
