#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import sys

import numpy as np

sys.path.append("Digitized-vs-Image-ECG-classification")
import joblib
import pandas as pd
import model_zoo
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import torch.nn as nn
import torch.optim as optim
from training import Ecg12LeadImageNetTrainerBinary

from helper_code import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ECG_Multilead_Dataset(Dataset):
    def __init__(self, path_to_dataset=None, transform=None):
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
        Simple Functions that loads the image and reshapes them
        :param image: PIL opbject
        :return: image resized
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
    row["Image"] = row["Image"].split(",")[0].strip()  #Only use one image for training TODO: change for later implememtation
    row["image"] = os.path.join(path.parent, row["Image"])
    return row


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()



# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7
    dropout = 0.2
    stride = 2
    dilation = 1
    batch_norm = True
    fc_hidden_dims = [128]
    num_of_classes = 2

    model = model_zoo.Ecg12ImageNet(1, hidden_channels, kernel_sizes, 512, 512,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=num_of_classes).to(device)



    dataset = ECG_Multilead_Dataset(path_to_dataset=data_folder)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=16)

    lr = 0.01
    epochs = 1
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadImageNetTrainerBinary(model, loss_fn, optimizer, device)
    fitResult2 = trainer.fit(train_dl,None, num_epochs=epochs, early_stopping=100, print_every=1)
    torch.save(model.state_dict(), os.path.join(model_folder,"dx_model.pt"))


# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    #device = 0
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7
    dropout = 0.2
    stride = 2
    dilation = 1
    batch_norm = True
    fc_hidden_dims = [128]
    num_of_classes = 2

    model = model_zoo.Ecg12ImageNet(1, hidden_channels, kernel_sizes, 512, 512,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=num_of_classes).to(device)

    path = Path().resolve()

    model_state_dict = torch.load(os.path.join(path, model_folder, 'dx_model.pt'))

    model.load_state_dict(model_state_dict) #not required can be changed in FUTURE!!!

    model.eval()

    return model


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model
    #if type(record) is list:
    #    path = os.path.split(record[0])[0]

    path = os.path.split(record)[0]


    image_files =get_image_files(record)

    images = []

    for image_path in image_files:
        image = Image.open(os.path.join(path,image_path)).convert('L')
        image = image.resize((512, 512))
        image = np.array(image)
        #plt.imshow(image)
        #plt.show()
        image = image.reshape(512, 512, 1)
        image = image / 255
        images.append(image)

    images = torch.utils.data.DataLoader(images, batch_size=32, shuffle=False) #TODO:Fix current validataion implementation (works until 32 images)


    for image in iter(images):
        image = image.transpose(1, 2).transpose(1, 3)
        image = image.to(device, dtype=torch.float)
        label = model(image)
        labels =label.flatten().detach().to("cpu")

    #labels = int(torch.sum(labels)/len(labels))
    labels = torch.where(label < 0, 0, 1).to("cpu").flatten()
    labels = list(np.array(labels))


    classes = {0: "Normal", 1: "Abnormal"}

    labels = [classes[key] for key in labels if key in classes]

    return labels



################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    joblib.dump(d, filename, protocol=0)


"""
if __name__ == '__main__':

    input = "ptb-xl/records100/00000"
    train_dx_model(input, model_folder="model", verbose=False)
    classes = {0:"Normal", 1:"Abnormal"}
    records = find_records(input)
    model = load_dx_model("model", verbose=True)
    model.eval()
    results_df = {"label":[], "record":[]}
    for record in tqdm(records, "Classifying Images"):
        label = run_dx_model(dx_model=model, record=os.path.join(input,record), signal="ptb-xl\\testset\\00001_lr.dat",verbose=True)
        label = int(torch.where(label<0, 0, 1))
        output = os.path.join("output", record)
        with open(output+".hea", "w") as f:
            f.write(f"#Image: {classes[label]}")
        results_df["label"].append(label)
        results_df["record"].append(record)
    results_df = pd.DataFrame(results_df)
    count = results_df["label"].value_counts().to_dict()
    print(f"There were {count[0]} Normal and {count[1]} Abnormal Signals detected!")
    #labels = run_dx_model(dx_model=model, record="ptb-xl\\testset\\00001_lr", signal="ptb-xl\t\estset\\00001_lr.dat", verbose=True)
"""