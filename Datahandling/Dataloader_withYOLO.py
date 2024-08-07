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



class ECG_cropped(Dataset):
    def __init__(self, path_to_dataset: list = None,get_signal:bool=False, transform=None, verbose: bool = False,
                 samples: int = None, YOLO_path: str = None):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.get_signal = get_signal
        self.transform = transform
        self.path_to_dataset = path_to_dataset
        self.yolo = YOLO(YOLO_path)
        self.lead_name = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
        self.reversed_lead_names = {v: k for k, v in self.lead_name.items()}
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
        if samples is not None:
            self.data = self.data.sample(samples)
        images = list(self.data["image"])
        max_images = 50
        self.results = []
        for i in range(0,len(images), max_images):
            results = self.yolo(images[i:i+max_images])
            results = [result.cpu() for result in results]
            self.results.extend(results)
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_to_img = Path(self.data.iloc[idx]["image"])
        dx = self.get_dx(idx)
        if self.get_signal:
            signal = self.load_signal(idx)
        result = self.results[idx]
        short_leads_sorted, long_lead = self.sort_leads(result)
        image = cv2.imread(path_to_img)
        # gaussian blur for denoising
        image = cv2.GaussianBlur(image, (3, 3), 100)
        images = self.get_cropped_images(image, short_leads_sorted, long_lead)
        if self.get_signal:
            return images, dx, signal
        else:
            return images, dx


    def get_dx(self, idx):
        dxs = [0] * len(self.classes)
        for dx in self.data.iloc[idx]["dx"]:
            #Use one class per Image
            dxs[self.classes[dx]] = 1
            break
        return np.array(dxs)

    def get_cropped_images(self, image, short_leads, long_lead, target_size=(128,128)):
        imgs = []
        for short_lead_sorted in short_leads:
            _,x, y, w, h = short_lead_sorted
            x, y = int(x - w / 2), int(y - h / 2)
            image_height, image_width = image.shape[:2]
            x = max(x, 0)
            y = max(y, 0)
            x_end = int(min(x + w, image_width))
            y_end = int(min(y + h, image_height))
            cropped_image = image[y:y_end, x:x_end]
            cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
            imgs.append(cropped_image.transpose(2,0,1)/255)
        _, x, y, w, h = long_lead.T
        x, y = int(x - w / 2), int(y - h / 2)
        image_height, image_width = image.shape[:2]
        x = max(x, 0)
        y = max(y, 0)
        x_end = int(min(x + w, image_width))
        y_end = int(min(y + h, image_height))
        cropped_image = image[y:y_end, x:x_end]
        cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
        imgs.append(cropped_image.transpose(2,0,1)/255)
        return np.array(imgs)



    def load_signal(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = np.concatenate(np.array(short_leads).T)
        signal = np.concatenate((signal, long_lead))
        return signal


    def sort_leads(self,result: torch.tensor):
        bboxes = np.concatenate((result.boxes.cls.reshape(result.boxes.cls.shape[0], 1), result.boxes.xywh), axis=1)
        short_leads = bboxes[bboxes[:, 0] == 0]
        if len(list(short_leads)) == 0:
            print("No short lead Detected use whole imeage instead")
            short_leads = 0, result.orig_shape[0] / 2, result.orig_shape[1] / 2, result.orig_shape[0], result.orig_shape[1] / 2  # use whole image
            short_leads = np.array([short_leads])
        if len(short_leads) != 12:
            short_leads = self.augment_data(short_leads) #fixes wrong predictions
        sorted_by_x = np.array(sorted(short_leads, key=lambda lead: lead[1]))
        chunks = np.array_split(sorted_by_x, len(sorted_by_x) // 3)
        sorted_chunks = [chunk[chunk[:, 2].argsort()] for chunk in chunks]
        sorted_by_x = np.vstack(sorted_chunks)
        try:
            long_leads = bboxes[bboxes[:, 0] == 1][0]
        except:
            print("No long lead Detected use whole imeage instead")
            long_leads = 1,result.orig_shape[0]/2, result.orig_shape[1]/2, result.orig_shape[0], result.orig_shape[1]/2 #use whole image
            long_leads = np.array(long_leads)
        return sorted_by_x, long_leads

    def augment_data(self, short_leads):
        if len(short_leads) > 12:
            return short_leads[:12]
        additional_rows_needed = 12 - len(short_leads)
        random_indices = np.random.choice(len(short_leads), additional_rows_needed, replace=True)
        additional_rows = short_leads[random_indices]
        return np.vstack((short_leads, additional_rows))




    def load_data(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = (np.array(short_leads).T, np.array(long_lead))
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

        # Apply Gaussian Blur to reduce noise
        denoised_image = cv2.GaussianBlur(image, (3, 3), 100)

        # Resize the image to 512x512
        resized_image = cv2.resize(denoised_image, (512, 512))
        resized_image = resized_image / 255
        resized_image.resize(512, 512, 1)
        return resized_image


class ECG_Turned(Dataset):
    def __init__(self, path_to_dataset: list = None,get_signal:bool=False, transform=None, verbose: bool = False,
                 samples: int = None, YOLO_path: str = None):
        super().__init__()
        self.data = None
        self.data_debug = []
        self.data_info = []
        self.get_signal = get_signal
        self.transform = transform
        self.path_to_dataset = path_to_dataset
        self.yolo = YOLO(YOLO_path)
        self.lead_name = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
        self.reversed_lead_names = {v: k for k, v in self.lead_name.items()}
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
            dx, signal, images = self.create_dataset(idx)
            self.img.append(images)
            self.dx.append(dx)
            self.signal.append(signal)
            progress_bar.update()
        if verbose:
            print(f"The Average Age is {self.data['Age'].astype(int).mean()}")
            print("We have")
            for category in ["Sex", "dx"]:
                data = self.data[category].value_counts().to_dict()
                print(data)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if self.get_signal:
            return self.img[idx], self.dx[idx], self.signal[idx]
        else:
            return self.img[idx], self.dx[idx]

    def create_dataset(self, idx):
        path_to_img = Path(self.data.iloc[idx]["image"])
        dx = self.get_dx(idx)
        if self.get_signal:
            signal = self.load_signal(idx)
        result = self.yolo(path_to_img)
        short_leads_sorted, long_lead = self.sort_leads(result[0].cpu())
        image = cv2.imread(path_to_img)
        # gaussian blur for denoising
        image = cv2.GaussianBlur(image, (3, 3), 100)
        images = self.get_cropped_images(image, short_leads_sorted, long_lead)
        return dx, signal, images


    def get_dx(self, idx):
        dxs = [0] * len(self.classes)
        for dx in self.data.iloc[idx]["dx"]:
            #Use one class per Image
            dxs[self.classes[dx]] = 1
            break
        return np.array(dxs)

    def get_cropped_images(self, image, short_leads, long_lead, target_size=(128,128)):
        imgs = []
        for short_lead_sorted in short_leads:
            _,x, y, w, h = short_lead_sorted
            x, y = int(x - w / 2), int(y - h / 2)
            image_height, image_width = image.shape[:2]
            x = max(x, 0)
            y = max(y, 0)
            x_end = int(min(x + w, image_width))
            y_end = int(min(y + h, image_height))
            cropped_image = image[y:y_end, x:x_end]
            cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
            imgs.append(cropped_image.transpose(2,0,1)/255)
        _, x, y, w, h = long_lead.T
        x, y = int(x - w / 2), int(y - h / 2)
        image_height, image_width = image.shape[:2]
        x = max(x, 0)
        y = max(y, 0)
        x_end = int(min(x + w, image_width))
        y_end = int(min(y + h, image_height))
        cropped_image = image[y:y_end, x:x_end]
        cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
        imgs.append(cropped_image.transpose(2,0,1)/255)
        return np.array(imgs)



    def load_signal(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = np.concatenate(np.array(short_leads).T)
        signal = np.concatenate((signal, long_lead))
        return signal


    def sort_leads(self,result: torch.tensor):
        bboxes = np.concatenate((result.boxes.cls.reshape(result.boxes.cls.shape[0], 1), result.boxes.xywh), axis=1)
        try:
            short_leads = bboxes[bboxes[:, 0] == 0]
        except:
            print("No short lead Detected use whole imeage instead")
            short_leads = 0, result.orig_shape[0] / 2, result.orig_shape[1] / 2, result.orig_shape[0], result.orig_shape[1] / 2  # use whole image
            short_leads = np.array([short_leads])
        if len(short_leads) != 12:
            short_leads = self.augment_data(short_leads) #fixes wrong predictions
        sorted_by_x = np.array(sorted(short_leads, key=lambda lead: lead[1]))
        chunks = np.array_split(sorted_by_x, len(sorted_by_x) // 3)
        sorted_chunks = [chunk[chunk[:, 2].argsort()] for chunk in chunks]
        sorted_by_x = np.vstack(sorted_chunks)
        try:
            long_leads = bboxes[bboxes[:, 0] == 1][0]
        except:
            print("No long lead Detected use whole imeage instead")
            long_leads = 1,result.orig_shape[0]/2, result.orig_shape[1]/2, result.orig_shape[0], result.orig_shape[1]/2 #use whole image
            long_leads = np.array(long_leads)
        return sorted_by_x, long_leads

    def augment_data(self, short_leads):
        if len(short_leads) > 12:
            return short_leads[:12]
        additional_rows_needed = 12 - len(short_leads)
        random_indices = np.random.choice(len(short_leads), additional_rows_needed, replace=True)
        additional_rows = short_leads[random_indices]
        return np.vstack((short_leads, additional_rows))




    def load_data(self, idx):
        path = Path(self.data.iloc[idx]["header"]).parent
        signal = helper_code.load_signals(os.path.join(path, self.data.iloc[idx]["basename"]))
        signal = pd.DataFrame(signal[0])
        long_lead = copy.deepcopy(signal[1])
        short_leads = []
        for i in range(4):
            short_leads.append(signal[np.arange(i * 3, (i + 1) * 3)].dropna().reset_index(drop=True))
        short_leads = pd.concat(short_leads, axis=1)
        signal = (np.array(short_leads).T, np.array(long_lead))
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
    path_to_dataset = ["/home/tdege/CinC_cleaned/Datahandling/Train/test_data"]
    yolo_model = "/home/tdege/CinC_cleaned/YOLO/LEAD_detector.pt"
    dataset_image = ECG_Turned(path_to_dataset, samples=1, YOLO_path=yolo_model)
    images, dx, signal = dataset_image[0]
    for image in images:
        plt.imshow(image.transpose(1,2,0))
        plt.show()


