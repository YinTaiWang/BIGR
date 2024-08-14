# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:05:37 2023

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
import random
import nibabel as nib

from monai.networks.nets import resnet50, resnet101, resnet10, resnet18
from monai.transforms import AddChanneld, Compose, LoadImaged, NormalizeIntensityd, ToTensord, ConcatItemsd, Spacingd, Orientationd, RandRotated, RandGaussianNoised
from monai.transforms import EnsureChannelFirstd
#from SoftTissueCAD.processing import MatchSpacingD
from monai.data import DataLoader, CacheDataset, Dataset

from torchmetrics.classification import BinaryAccuracy, AUROC
from monai.metrics import ROCAUCMetric

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
    RandRotate,
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
#print(root_dir)


label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
#label_path = r"C:\Users\aisha\Documents\Thesis\Scans\labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)
labels = df_label.Malignant
labels = np.array(labels, dtype=np.int64)
labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float() #for BCE loss

# load data
data_dir = "/trinity/home/agoedhart/Data/Scans/"
#data_dir = r"C:\Users\aisha\Documents\Thesis\Scans\Compleet_4_fasen\Bigger_dataset"
img_name = 'ph2_bbox.nii.gz'
seg_name = 'ph2_bbox_seg.nii.gz'
img_name_ph1 = 'ph1_bbox.nii.gz'
seg_name_ph1 = 'ph1_bbox_seg.nii.gz'
images = glob.glob(os.path.join(data_dir,'*',img_name))
images_ph1 = glob.glob(os.path.join(data_dir,'*',img_name_ph1))
segmentations = glob.glob(os.path.join(data_dir,'*',seg_name))

# data dictionary
data = []
for i in range(0,len(images)):
    data.append({"img": images[i], "img_ph1": images_ph1[i], "seg": segmentations[i], "label": labels[i]})
    #data.append({"img": images[i], "label": labels[i]})


# Use segmentations and data augmentation  as input
use_seg = True
data_aug = False

# Define transforms
if use_seg == False: 
    print('Segmentations NOT used as input')
    in_channels = 2
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ph1"]),
            EnsureChannelFirstd(keys=["img", "img_ph1"]),
            Orientationd(keys=["img", "img_ph1"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img", "img_ph1",]),
            ConcatItemsd(keys=["img", "img_ph1"], name="img"),
            ToTensord(keys=["img", "img_ph1"]),
        ]
    )
    
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ph1"]),
            EnsureChannelFirstd(keys=["img", "img_ph1"]),
            Orientationd(keys=["img", "img_ph1"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img", "img_ph1",]),
            ConcatItemsd(keys=["img", "img_ph1"], name="img"),
            ToTensord(keys=["img", "img_ph1"]),
        ]
    )


if use_seg == True: 
    in_channels = 3
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ph1", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ph1", "seg"]),
            Orientationd(keys=["img", "img_ph1", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img", "img_ph1",]),
            ConcatItemsd(keys=["img", "img_ph1", "seg"], name="img"),
            ToTensord(keys=["img", "img_ph1", "seg"]),
        ]
    )
    
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ph1", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ph1", "seg"]),
            Orientationd(keys=["img", "img_ph1", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img", "img_ph1",]),
            ConcatItemsd(keys=["img", "img_ph1", "seg"], name="img"),
            ToTensord(keys=["img", "img_ph1", "seg"]),
        ]
    )


    
    
# split train and val
train, val = sklearn.model_selection.train_test_split(data, test_size = 0.2, random_state=42, stratify=labels)

# create a training data loader
train_ds = Dataset(data=train, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = Dataset(data=val, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)


# Check train_loader

train_img = []
subject = []
for i in range(0,8):
    img_path = random.sample(train_ds.data,8)[i]['img']
    subject.append((img_path[:-16])[-3:])
    nib_img = nib.load(img_path)
    nib_img = (nib_img.get_fdata())[:,:,48]
    train_img.append(nib_img)

for num, x in enumerate(train_img):
    img = (x.T)
    plt.subplot(2,4,num+1)
    plt.axis('off')
    plt.suptitle("Training data sample")
    plt.title(subject[num])
    plt.imshow(img, cmap="gray", origin="lower")
     
plt.savefig('/trinity/home/agoedhart/Data/Plots/training_data.png')






# Create a model, CrossEntropyLoss and Adam optimizer
#dropout = 1
#print('Dropout:', dropout)
architecture = ['DenseNet', 'ResNet'][1]

if architecture == 'DenseNet': 
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels = in_channels, out_channels=2).to(device)
if architecture == 'ResNet': 
    model = resnet10(n_input_channels=2, num_classes=2)
    
print('Architecture:', architecture) 
  
#print(model.summary())
if torch.cuda.is_available():
    model.cuda()

#loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

lr = 1e-4
print('learning rate:', lr)
optimizer = torch.optim.Adam(model.parameters(), lr)

# AUC
auc_metric = ROCAUCMetric()


# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
val_loss_values = []
metric_values = []
metric_values_train = []
writer = SummaryWriter()
max_epochs = 30

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
        inputs, labels = batch_data["img"].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
        value_train = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
        metric_count_train += len(value_train)
        num_correct_train += value_train.sum().item()

    # Accuracy
    metric_train = num_correct_train / metric_count_train
    metric_values_train.append(metric_train)

    #epoch_loss /= step
    epoch_loss /= len(train_loader)
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        
        num_correct = 0.0
        metric_count = 0
        
        for val_data in val_loader:
            val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                #value = torch.eq(val_outputs.argmax(dim=1), val_labels) #EDITED for CrossEntropyLoss
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                loss2 = loss_function(val_outputs, val_labels)
                val_loss += loss2.item()
                print('val_labels:', val_labels.argmax(dim=1), 'val_outputs:', val_outputs.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()
              
          
        
        # AUC        
        #auc_metric(y_pred_act, y_onehot)
        #auc_result = auc_metric.aggregate()
        #auc_metric.reset()
        #print(val_labels)
        #print()
        #auroc = AUROC(task="binary")
        #print('AUC:', auroc(val_outputs, val_labels))
        
        # Accuracy
        metric = num_correct / metric_count
        metric_values.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)
        
    val_loss /= len(val_loader)   
    val_loss_values.append(val_loss) 
        
print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()

#print('val_labels:', val_labels, 'val_outputs:', val_outputs.argmax(dim=1))
#print('val_outputs:', val_outputs.argmax(dim=1))


plt.figure("train", (12, 6))
plt.subplot(2, 2, 1)
plt.title("Train Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
z = val_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="r", label = 'train')
plt.legend()
plt.subplot(2, 2, 2)
plt.title("Validation Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
z = val_loss_values
plt.xlabel("epoch")
plt.plot(x, z, color="b", label = 'val')
plt.legend()
plt.subplot(2, 2, 3)
plt.title("Train accuracy")
x = [val_interval * (i + 1) for i in range(len(metric_values_train))]
y = metric_values_train
plt.xlabel("epoch")
plt.plot(x, y, color="r")
plt.show()
plt.subplot(2, 2, 4)
plt.title("Validation accuracy")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="b")
plt.show()

plt.savefig('/trinity/home/agoedhart/Data/Plots/loss_plot.png')

print('plot loss')


# create a validation data loader
#test_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
test_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
itera = iter(test_loader)



def get_next_im():
    test_data = next(itera)
    return test_data['img'].to(device), test_data['label'].unsqueeze(0).to(device)


def plot_occlusion_heatmap(im, heatmap):
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(im.cpu()))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()
    
    
# Get a random image and its corresponding label
img, label = get_next_im()

# Get the occlusion sensitivity map
occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=12, n_batch=10, stride=12)
# Only get a single slice to save time.
# For the other dimensions (channel, width, height), use
# -1 to use 0 and img.shape[x]-1 for min and max, respectively
depth_slice = img.shape[2] // 2
occ_sens_b_box = [depth_slice-1, depth_slice, -1, -1, -1, -1]

occ_result, _ = occ_sens(x=img, b_box=occ_sens_b_box)
occ_result = occ_result[0, label.argmax().item()][None]

fig, axes = plt.subplots(1, 2, figsize=(25, 15), facecolor="white")

for i, im in enumerate([img[:, :, depth_slice, ...], occ_result]):
    cmap = "gray" if i == 0 else "jet"
    ax = axes[i]
    im_show = ax.imshow(np.squeeze(im[0][0].detach().cpu()), cmap=cmap)
    ax.axis("off")
    fig.colorbar(im_show, ax=ax)
    
fig.savefig('/trinity/home/agoedhart/Data/Plots/occlusion.png')
print('End script')   