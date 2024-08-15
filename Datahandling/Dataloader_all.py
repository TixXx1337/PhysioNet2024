import copy
import sys

import numpy
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
from ultralytics import YOLO
import matplotlib.pyplot as plt


class ECG_multi(Dataset):
    def __init__(self, path_to_dataset: list = None,get_signal:bool=False, transform=None, verbose: bool = False,
                 samples: int = None, YOLO_path: str = None):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.get_signal = get_signal
        self.transform = transform
        self.YOLO_path = YOLO_path
        self.path_to_dataset = path_to_dataset
        self.lead_name = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
        self.reversed_lead_names = {v: k for k, v in self.lead_name.items()}
        #self.classes = {'NORM': 0, 'STTC': 1, 'PAC': 2, 'Old MI': 3, 'HYP': 4, 'TACHY': 5, 'CD': 6, 'BRADY': 7,
        #                'AFIB/AFL': 8, 'PVC': 9, 'Acute MI': 10}
        self.classes = {'NORM': 0, 'Old MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4, 'AFIB/AFL': 5, 'PVC': 6, 'TACHY': 7, 'BRADY': 8, 'PAC': 9, 'Acute MI': 10}
        self.reversed_classed = {v: k for k, v in self.classes.items()}
        df_dataset_all = []
        for path in path_to_dataset:
            tqdm.pandas()
            header = glob.glob(f"{path}/*.hea")
            df_dataset = pd.DataFrame({"header": header})
            df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
            df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
            df_dataset = df_dataset.progress_apply(func=get_class, axis=1)
            df_dataset_all.append(df_dataset)
        df_dataset_all = pd.concat(df_dataset_all, ignore_index=True)
        self.data = df_dataset_all
        if samples is not None:
            self.data = self.data.sample(samples)
        self.data["Labels"] = self.data["dx"]
        self.data = self.data.explode("dx")
        self.data.dropna(inplace=True)
        self.data['labels'] = self.data['dx'].apply(normalize_string)
        #gets all class combinations
        #l = list(self.data['labels'].value_counts().index)
        #self.classes = {l[i]: i for i in range(len(l))}
        #self.reversed_classed = {v: k for k, v in self.classes.items()}
        self.data["dx"] = self.data["dx"].apply(lambda x: [x])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_to_img = Path(self.data.iloc[idx]["image"])
        dx = self.get_dx(idx)
        image = cv2.imread(path_to_img)
        # gaussian blur for denoising
        image = cv2.GaussianBlur(image, (3, 3), 100)
        image = self.get_whole_image(image)
        return image, dx


    def get_dx(self, idx):
        dxs = [0] * len(self.classes)
        #for dx in self.data.iloc[idx]["dx"]:
        for dx in self.data.iloc[idx]["Labels"]:
            dxs[self.classes[dx]] = 1
            #break
        return np.array(dxs)

    def get_whole_image(self, image):
        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1) / 255
        return image


class ECG_all(Dataset):
    def __init__(self, path_to_dataset: list = None,get_signal:bool=False, transform=None, verbose: bool = False,
                 samples: int = None, YOLO_path: str = None):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.get_signal = get_signal
        self.transform = transform
        self.YOLO_path = YOLO_path
        self.path_to_dataset = path_to_dataset
        self.lead_name = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
        self.reversed_lead_names = {v: k for k, v in self.lead_name.items()}
        df_dataset_all = []
        for path in path_to_dataset:
            tqdm.pandas()
            header = glob.glob(f"{path}/*.hea")
            df_dataset = pd.DataFrame({"header": header})
            df_dataset["basename"] = df_dataset["header"].apply(lambda x: os.path.basename(x))
            df_dataset["basename"] = df_dataset["basename"].apply(lambda x: x[:-4])
            df_dataset = df_dataset.progress_apply(func=get_class, axis=1)
            df_dataset_all.append(df_dataset)
        df_dataset_all = pd.concat(df_dataset_all, ignore_index=True)
        self.data = df_dataset_all
        if samples is not None:
            self.data = self.data.sample(samples)
        self.data = self.data.explode("dx")
        self.data.dropna(inplace=True)
        self.data['labels'] = self.data['Labels'].apply(normalize_string)
        #gets all class combinations
        l = list(self.data['labels'].value_counts().index)
        self.classes = {l[i]: i for i in range(len(l))}
        self.reversed_classed = {v: k for k, v in self.classes.items()}
        self.data["dx"] = self.data["dx"].apply(lambda x: [x])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_to_img = Path(self.data.iloc[idx]["image"])
        dx = self.get_dx(idx)
        image = cv2.imread(path_to_img)
        # gaussian blur for denoising
        image = cv2.GaussianBlur(image, (3, 3), 100)
        image = self.get_whole_image(image)
        return image, dx


    def get_dx(self, idx):
        #Use one class per Image
        size =  len(self.classes)
        dx = np.zeros(size, dtype=int)
        index = self.classes[self.classes[self.data.iloc[idx]["labels"]]]
        dx[index] = 1
        return dx

    def get_whole_image(self, image):
        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1) / 255
        return image


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

def normalize_string(s):
    # Split the string by commas, strip whitespace, sort the elements, and then join them back
    return ', '.join(sorted([item.strip() for item in s.split(',')]))



"""
if __name__ == '__main__':
    #äpath_to_dataset = [r"C:/Users\Tizian Dege\PycharmProjects\CinC_cleaned\Datahandling\Train/20images"]
    path_to_dataset = ["/work/scratch/td38heni/all"]
    path_to_dataset = ["/work/scratch/td38heni/train_set"]
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/20images"]
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/Train/test_data"]
    yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    dataset_image = ECG_Turned(path_to_dataset, samples=1, YOLO_path=yolo_model)

"""

