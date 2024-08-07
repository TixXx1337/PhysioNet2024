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
import torch


from helper_code import *

from Trainer.model import ECG_Dx

from ultralytics import YOLO

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
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    digitization_features = list()
    classification_features = list()
    classification_labels = list()

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i + 1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record)

        digitization_features.append(features)

        # Some images may not be labeled...
        labels = load_labels(record)
        if any(label for label in labels):
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.

    # This very simple model trains a random forest model with these very simple features.
    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    classification_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(
        classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()


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
    path_to_yolo =  os.path.join(os.getcwd(),"YOLO", 'LEAD_detector.pt')
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
    images = images.reshape(1, *images.shape).to(device ,dtype=torch.float32)
    out = classification_model(images)
    max_index = torch.argmax(out).item()
    labels = class_dict[max_index]
    return signal, labels


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
    #signal, labels = run_models(record, digitization_model, classification_model, True)