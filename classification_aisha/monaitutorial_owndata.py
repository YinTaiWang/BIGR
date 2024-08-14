# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:09:24 2023

@author: aisha
"""

import logging
import os
import sys
import shutil
import tempfile


import os
import numpy as np
import time
import glob
import pandas as pd
import sklearn.model_selection
import torchvision
from torchsummary import summary

from monai.networks.nets import resnet50
from monai.transforms import AddChanneld, Compose, LoadImaged, NormalizeIntensityd, ToTensord, ConcatItemsd, Spacingd, Orientationd
from monai.transforms import EnsureChannelFirstd
#from SoftTissueCAD.processing import MatchSpacingD
from monai.data import DataLoader, CacheDataset, Dataset

from torchmetrics.classification import BinaryAccuracy, AUROC

import pytorch_lightning as pl
#from pytorch_lightning.metrics import Accuracy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

# Set data directory
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
#label_path = r"C:\Users\aisha\Documents\Thesis\Scans\labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)
labels = df_label.Malignant
labels = np.array(labels, dtype=np.int64)
#labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float() # Only for BCEWithLogitsLoss

# load data
data_dir = "/trinity/home/agoedhart/Data/Scans/"
#data_dir = r"C:\Users\aisha\Documents\Thesis\Scans\Compleet_4_fasen\Bigger_dataset"
img_name = 'ph2_bbox.nii.gz'
seg_name = 'ph2_bbox_seg.nii.gz'
images = glob.glob(os.path.join(data_dir,'*',img_name))
segmentations = glob.glob(os.path.join(data_dir,'*',seg_name))

# data dictionary
data = []
for i in range(0,len(images)):
    #data.append({"img": images[i], "seg": segmentations[i], "label": labels[i]})
    data.append({"img": images[i], "label": labels[i]})

# Define transforms
#train_transforms = Compose(
#    [
#        LoadImaged(keys=["img", "seg"]),
#        EnsureChannelFirstd(keys=["img", "seg"]),
#        Orientationd(keys=["img", "seg"], axcodes="RAS"),
#        NormalizeIntensityd(keys=["img"]),
#        ConcatItemsd(keys=["img", "seg"], name="img"),
#        ToTensord(keys=["img", "seg"]),
#    ]
#)



# Set transforms
#val_transforms = Compose(
#    [
#        LoadImaged(keys=["img", "seg"]),
#        EnsureChannelFirstd(keys=["img", "seg"]),
#        Orientationd(keys=["img", "seg"], axcodes="RAS"),
#        NormalizeIntensityd(keys=["img"]),
#        ConcatItemsd(keys=["img", "seg"], name="img"),
#        ToTensord(keys=["img", "seg"]),
#    ]
#)


train_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        Orientationd(keys=["img"], axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        ConcatItemsd(keys=["img"], name="img"),
        ToTensord(keys=["img"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        Orientationd(keys=["img"], axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        ConcatItemsd(keys=["img"], name="img"),
        ToTensord(keys=["img"]),
    ]
)


# split train and val
train, val = sklearn.model_selection.train_test_split(data, test_size = 0.2, random_state=1, stratify=labels)

# create a training data loader
train_ds = Dataset(data=train, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = Dataset(data=val, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

loss_function = torch.nn.CrossEntropyLoss()
#loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

sigmoid = nn.Sigmoid()
auroc = AUROC(task="binary")

# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
val_loss_values = []
metric_values = []
metric_values_train = []
auc_values = []
writer = SummaryWriter()
max_epochs = 50

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    val_loss = 0
    step = 0
    metric_count_train = 0
    num_correct_train = 0.0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['img'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        # Accuracy
        value_train = torch.eq(outputs.argmax(dim=1), labels) # See if output is equal to label
        metric_count_train += len(value_train) # Number of predictions
        num_correct_train += value_train.sum().item() # Total number of correct predictions
    
    # Loss
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Accuracy
    metric_train = num_correct_train / metric_count_train # Accuracy = n correct / n total
    metric_values_train.append(metric_train) # Fill accuracy list every epoch


    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        prob = []
        target = []
        for val_data in val_loader:
            val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                #val_sigm = (sigmoid(val_outputs))[0][1].item()
                #val_target = val_labels[0][1].item()
                #prob.append(val_sigm)
                #target.append(val_target)
                #print('Sigmoid:', val_sigm)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
                loss2 = loss_function(val_outputs, val_labels) # Calculate loss
                val_loss += loss2.item() # Set loss value
	
        metric = num_correct / metric_count
        metric_values.append(metric)
        #prob1 = torch.FloatTensor(prob)
        #target1 = torch.FloatTensor(target)
        #print(prob1)
        #print(target1)
        #print('AUROC:', auroc(prob1, target1))
        #auc_values.append(auroc(prob1, target1).item())

        # Append loss    
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


plt.figure("train2", (10, 10))
plt.subplot(2, 2, 1)
plt.title("Train & Val Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
z = val_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="r", label = 'train')
plt.plot(x, z, color="b", label = 'val')
plt.ylim([0, 2])
plt.xlim([0, max_epochs])
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Train Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="r", label = 'train')
plt.ylim([0, 1])
plt.xlim([0, max_epochs])
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Train & Val accuracy")
x = [val_interval * (i + 1) for i in range(len(metric_values_train))]
y = metric_values_train
z = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="r", label = 'train')
plt.plot(x, z, color="b", label = 'val')
plt.ylim([0, 1.1])
plt.xlim([0, max_epochs])
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Validation accuracy")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
z = metric_values
plt.xlabel("epoch")
plt.plot(x, z, color="b", label = 'val')
plt.ylim([0, 1.1])
plt.xlim([0, max_epochs])
plt.legend()
#plt.show()

plt.savefig('/trinity/home/agoedhart/Data/Plots/plot_monai_CEL_valtrain.png')



