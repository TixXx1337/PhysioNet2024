#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################
import joblib
import numpy as np
import os
import sys
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from helper_code import *
from Trainer.model import ECG_Dx
from Trainer.train_dx_model import *
from Datahandling.Dataloader_withYOLO import *
from ultralytics import YOLO
import torch
#classes = {'NORM': 0, 'STTC': 1, 'PAC': 2, 'Old MI': 3, 'HYP': 4, 'TACHY': 5, 'CD': 6, 'BRADY': 7, 'AFIB/AFL': 8, 'PVC': 9, 'Acute MI': 10}
classes = {'NORM': 0, 'STTC': 1, 'Old MI': 2, 'HYP': 3, 'TACHY': 4, 'CD': 5, 'AFIB/AFL': 6, 'PVC': 7, 'Acute MI': 8}
class_dict = {v: k for k, v in classes.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    # Parameters
    batch_size = 32
    path = os.getcwd()
    num_of_epochs = 10
    yolo_model = os.path.join(path, model_folder, "YOLO", 'LEAD_detector.pt')


    #get model
    classification_model = ECG_Dx.get_model("ViT", 11)
    classification_model = classification_model.to(device)


    #dataset for training
    ds = ECG_cropped([data_folder], get_signal=False, YOLO_path=yolo_model)
    ds_train, ds_val = torch.utils.data.random_split(ds, [0.8, 0.2])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    #trainer

    optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Ecg12LeadImageNetTrainerMulticlass(model=classification_model, optimizer=optimizer, loss_fn=loss_fn,device=device)
    train_result = trainer.train(num_of_epochs=num_of_epochs, train_dataloader=dl_train, val_datloader=dl_val)

    #save model
    classification_model = classification_model.to("cpu")
    model_dict = copy.deepcopy(classification_model.state_dict())
    torch.save(model_dict, f'{model_folder}/model_dx.pt')




# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_model = None


    classification_model = ECG_Dx.get_model("ViT", 9)
    digitization_filename = os.path.join(model_folder, 'model_dx.pt')
    classification_model.load_state_dict(torch.load(digitization_filename))
    return digitization_model, classification_model


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.
    path_to_yolo = os.path.join(os.getcwd(),"model","YOLO", 'LEAD_detector.pt')
    digitization_model = digitization_model.to("cpu")
    yolo = YOLO(path_to_yolo)
    signal = None
    path = os.path.split(record)[0]
    image_files = get_image_files(record)
    path_to_img = os.path.join(path, image_files[0])
    result = yolo(path_to_img)
    short_leads_sorted, long_lead = sort_leads(result[0].cpu())
    image = cv2.imread(path_to_img)

    # gaussian blur for denoising
    image = cv2.GaussianBlur(image, (3, 3), 100)
    images = get_cropped_images(image, short_leads_sorted, long_lead)
    images = torch.tensor(images)
    images = images.reshape(1, *images.shape).to("cpu" ,dtype=torch.float32)
    out = classification_model(images)
    max_index = torch.argmax(out).item()
    labels = class_dict[max_index]
    return signal, [labels]


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def get_cropped_images(image, short_leads, long_lead, target_size=(128,128)):
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



def sort_leads(result: torch.tensor):
    bboxes = np.concatenate((result.boxes.cls.reshape(result.boxes.cls.shape[0], 1), result.boxes.xywh), axis=1)
    try:
        short_leads = bboxes[bboxes[:, 0] == 0]
    except:
        print("No short lead Detected use whole imeage instead")
        short_leads = 0, result.orig_shape[0] / 2, result.orig_shape[1] / 2, result.orig_shape[0], result.orig_shape[1] / 2  # use whole image
        short_leads = np.array([short_leads])
    if len(short_leads) != 12:
        short_leads = augment_data(short_leads) #fixes wrong predictions
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

def augment_data(short_leads):
    if len(short_leads) > 12:
        return short_leads[:12]
    additional_rows_needed = 12 - len(short_leads)
    random_indices = np.random.choice(len(short_leads), additional_rows_needed, replace=True)
    additional_rows = short_leads[random_indices]
    return np.vstack((short_leads, additional_rows))




def tensor_to_labels(tensor):
    tensor = tensor.squeeze().numpy()  # Remove batch dimension and convert to numpy array
    labels = []
    for i, value in enumerate(tensor):
        if value == 1:
            labels.append(class_dict[i])
    return labels


# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])


# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)


if __name__ == '__main__':
    model_folder = "/home/tdege/CinC_cleaned/model"
    data_folder = "/home/tdege/CinC_cleaned/Datahandling/20images/"
    record = '/home/tdege/CinC_cleaned/Datahandling/20images/00001_lr'
    digitization_model, classification_model = load_models(model_folder, True)
    classification_model.eval()
    records = find_records(data_folder)
    train_models(data_folder, "model", False)
    #signal, labels = run_models(record, digitization_model, classification_model, True)
