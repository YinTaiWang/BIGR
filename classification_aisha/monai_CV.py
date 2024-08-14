# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:52:56 2023

@author: aisha
"""


import logging
import os
import sys
import tempfile
import numpy as np
import glob
import pandas as pd
import sklearn.model_selection
import torchvision
import monai
import random
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
#import scikitplot as skplt

from monai.networks.nets import resnet50, resnet101, resnet10, resnet18, resnet34
from monai.transforms import  Compose, LoadImaged, NormalizeIntensityd, ToTensord, ConcatItemsd, Spacingd, Orientationd
from monai.transforms import RandRotated, RandGaussianNoised, RandFlipd, RandRotate90d
from monai.transforms import EnsureChannelFirstd
from monai.data import DataLoader, CacheDataset, Dataset
from monai.metrics import ROCAUCMetric
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

from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy, AUROC
from torchsummary import summary
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
#####################################################################################


pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#print_config()

##################### DATA ######################################
# labels
label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)
labels = df_label.Malignant
labels = np.array(labels, dtype=np.int64)
# images and segmentations
data_dir = "/trinity/home/agoedhart/Data/ScansMasked"
print('MASKED')
img_name = 'ph2_masked.nii.gz'
images = glob.glob(os.path.join(data_dir,'*',img_name))

# data dictionary
data = []
for i in range(0,len(images)):
    data.append({"img": images[i], "label": labels[i]})

#################### Method choices ################################
# Use segmentations as input
use_seg = False
# Use data augmentation
data_aug = True
# Choose architecture and pre-training
architecture = ['DenseNet', 'ResNet'][0]
pre = False
# Choose loss functions and optimizer
lossfunc = ['BCEWithLogitsLoss', 'BCELoss', 'CrossEntropyLoss'][2]
if lossfunc == 'BCEWithLogitsLoss':
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float() #for BCEWithLogitsLoss only
# Choose random seed
random_seed = 1
# Choose learning rate and weight decay
lr = 5e-4
wd = 0 #1e-5
# Choose max. epochs
max_epochs = 50
# Name of plot figure
fig_name = '/trinity/home/agoedhart/Data/Plots/plots_test_CV2.png'
###################### Data Transforms ###############################
# Define transforms based on segmentation and data augmentation
if use_seg == False and data_aug == False: 
    print('Segmentations not used as input')
    print('Data augmentation was not used')
    in_channels = 1
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

if use_seg == True  and data_aug == False: 
    print('Segmentations used as input')
    print('Data augmentation was not used')
    in_channels = 2
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img"]),
            ConcatItemsd(keys=["img", "seg"], name="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img"]),
            ConcatItemsd(keys=["img", "seg"], name="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )

if use_seg == True and data_aug == True: 
    print('Segmentations used as input')
    print('Data augmentation was used')
    in_channels = 2
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            # Preprocessing
            NormalizeIntensityd(keys=["img"]),
            # Data augmentation
            RandRotated(keys=["img", "seg"], range_z = 0.35, mode=("bilinear", "nearest"), prob = 0.3),
            RandFlipd(keys=["img", "seg"], prob = 0.5),
            RandGaussianNoised(keys=["img"], prob = 1.0, std = 0.05),
            # 
            ConcatItemsd(keys=["img", "seg"], name="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys=["img"]),
            ConcatItemsd(keys=["img", "seg"], name="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )
if use_seg == False and data_aug == True: 
    print('Segmentations not used as input')
    print('Data augmentation was used')
    in_channels = 1
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Orientationd(keys=["img"], axcodes="RAS"),
            # Preprocessing
            NormalizeIntensityd(keys=["img"]),
            # Data augmentation
            RandRotated(keys=["img"], range_z = 0.35, mode=("bilinear"), prob = 0.3),
            RandFlipd(keys=["img"], prob = 0.5),
            #RandGaussianNoised(keys=["img"], prob = 1.0, std = 0.05),
            
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



################################# Train and Val loader ###############################   
# split train, val and test set
crossval, test = sklearn.model_selection.train_test_split(data, test_size = 0.13, random_state=random_seed, stratify=labels)

crossval_img = [crossval[i]['img'] for i in range(len(crossval))]
crossval_label = [crossval[i]['label'] for i in range(len(crossval))]

# K-fold cross validation (stratified)
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(crossval_img, crossval_label)

train_cv = []
val_cv = []

for i, (train_index, val_index) in enumerate(skf.split(crossval_img, crossval_label)):
    # Split in folds
    train = [crossval[j] for j in train_index]
    val = [crossval[m] for m in val_index]
    # create a training data loader
    train_ds = Dataset(data=train, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory)
    # create a validation data loader
    val_ds = Dataset(data=val, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)
    # append loaders
    train_cv.append(train_loader)
    val_cv.append(val_loader)

print('# train:', len(train))
print('# val:', len(val))
print('# test:', len(test))

test_ds = Dataset(data=test, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

# Check train_loader and create image
examples = iter(train_loader)
example1 = next(examples)
example2 = next(examples)
example3 = next(examples)
example4 = next(examples)
example5 = next(examples)
example6 = next(examples)
example7 = next(examples)
example8 = next(examples)
samples = [example1, example2, example3, example4, example5, example6, example7, example8]
for i in range(0,8):
    a = samples[i]['img']
    b = a.numpy()
    c = b[0,0,:,:,48]
    plt.subplot(2,4,i+1)
    plt.axis('off')
    plt.suptitle("Training data sample")
    plt.imshow(c, cmap="gray", origin="lower") 
plt.savefig('/trinity/home/agoedhart/Data/Plots/training_data.png')

##################################### Classifier  ############################################
# Loss function
if lossfunc == 'BCEWithLogitsLoss': 
    loss_function = torch.nn.BCEWithLogitsLoss()  # add pos_weigth maybe
    n_out = 1
if lossfunc == 'BCELoss':
    loss_function = torch.nn.BCELoss()
    n_out =1
if lossfunc == 'CrossEntropyLoss':
    loss_function = torch.nn.CrossEntropyLoss()
    n_out = 2

# Print choices
print('Architecture:', architecture)
print('# input channels:', in_channels)
print('# output channels:', n_out)
print('Loss function:', lossfunc) 
print('Pre-trained:', pre)
print('learning rate:', lr)
print('weight decay:', wd)
print(fig_name)

# Create a model
if architecture == 'DenseNet': 
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels = in_channels, out_channels = n_out).to(device)
if architecture == 'ResNet': 
    model = resnet10(pretrained = pre, n_input_channels= in_channels, num_classes=2)
  
# Send to GPU
if torch.cuda.is_available():
    model.cuda()



# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

# Softmax and Sigmoid functions
softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()

# AUC
auc_metric = ROCAUCMetric()
auroc = AUROC(task="binary")

# Loss and accuracy for all folds
train_loss_cv = []
val_loss_cv = []
val_acc_cv = []
train_acc_cv = []
best_metrics = []
best_models = []

train_cv = train_cv[4]
val_cv = val_cv[4]

for train_loader, val_loader in zip(train_cv, val_cv):
    ##

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
            
            # Training loss
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            
            # Training accuracy
            value_train = torch.eq(outputs.argmax(dim=1), labels) # See if output is equal to label
            metric_count_train += len(value_train) # Number of predictions
            num_correct_train += value_train.sum().item() # Total number of correct predictions
        
        # Training loss
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        #Training accuracy
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
    
                    # Validation accuracy
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
    
                    # Validation loss
                    loss2 = loss_function(val_outputs, val_labels) # Calculate loss
                    val_loss += loss2.item() # Set loss value
    	    
            # Validation accuracy
            metric = num_correct / metric_count
            metric_values.append(metric)
    
            
            #prob1 = torch.FloatTensor(prob)
            #target1 = torch.FloatTensor(target)
            #print(prob1)
            #print(target1)
            #print('AUROC:', auroc(prob1, target1))
            #auc_values.append(auroc(prob1, target1).item())
    
            # Validation loss
            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)
    
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_model = model
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")
    
            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)
    
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    
    train_loss_cv.append(epoch_loss_values)
    val_loss_cv.append(val_loss_values)
    val_acc_cv.append(metric_values)
    train_acc_cv.append(metric_values_train)
    best_metrics.append(best_metric)
    best_models.append(best_model)

########################## Test set ####################################

# Choose best model
i_best = np.argmax(best_metrics)
print('best metric for fold', i_best+1)

for m in range(len(best_models)):
    file_name = 'best_metric_model_fold_' + str(m) + '.pth'
    torch.save(best_models[m].state_dict(), file_name)

best_model_file = 'best_metric_model_fold_' + str(i_best) + '.pth'
best_model_path = os.path.join("/trinity/home/agoedhart", best_model_file)
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_loss_values = []
test_metric_values = []
test_num_correct = 0.0
test_metric_count = 0
test_loss = 0

for test_data in test_loader:
    test_images, test_labels = test_data['img'].to(device), test_data['label'].to(device)
    with torch.no_grad():
        test_outputs = model(test_images)

        # test accuracy
        test_value = torch.eq(test_outputs.argmax(dim=1), test_labels)
        test_metric_count += len(test_value)
        test_num_correct += test_value.sum().item()

        # test loss
        loss3 = loss_function(test_outputs, test_labels) # Calculate loss
        test_loss += loss3.item() # Set loss value
	    
# test accuracy
test_metric = test_num_correct / test_metric_count
test_metric_values.append(test_metric)
print('Test accuracy:', test_metric)

# Validation loss
test_loss /= len(test_loader)
test_loss_values.append(test_loss)
print('Test loss:', test_loss)


#################### Check #############################################
if lossfunc == 'CrossEntropyLoss':
    print('#################################')
    print('CrossEntropyLoss function check')
    print('Output:', outputs)
    print('Label:', labels)
    print('Softmax:', softmax(outputs))
    print('-log(softmax):', -torch.log(softmax(outputs))) # Cross Entropy Loss formula
    print('CE loss:', loss_function(outputs,labels)) # Cross Entropy Loss Pytorch

######################### Performance plot #############################
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

plt.savefig(fig_name)
