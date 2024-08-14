# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:05:43 2023

@author: aisha
"""

#import ipdb
from genericpath import samefile
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

from monai.networks.nets import resnet50, resnet101, resnet10, resnet18, resnet34
from monai.networks.nets import Classifier
from monai.transforms import  Compose, LoadImaged, NormalizeIntensityd, ToTensord, ConcatItemsd, Spacingd, Orientationd
from monai.transforms import RandRotated, RandGaussianNoised, RandFlipd, RandZoomd
from monai.transforms import EnsureChannelFirstd, MapTransform, ScaleIntensityd, RandGaussianSmoothd
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

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report, f1_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torchmetrics.classification import BinaryAccuracy, AUROC
from torchsummary import summary
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
#####################################################################################


pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################### Method choices ################################
# Binary classification
n_classes = 1
# Use segmentations as input
use_seg = False
# Use data augmentation
data_aug = True
# Use T2 too
use_T2 = False
# Choose architecture and pre-training
architecture = ['DenseNet', 'ResNet', 'Classifier'][1]
# Choose loss functions and optimizer
lossfunc = ['BCEWithLogitsLoss', 'BCELoss', 'CrossEntropyLoss'][0]
# Choose random seed
random_seed = 42
# Hyperparameters
optimizer = 'Adam'
lr = 1e-5
wd = 0 #1e-5
wf = 1
drop = 0.25
bs = 1 # batch size
# Pretrained weights ResNet10
pre = True
frozen = False

# Fold 
n=3
print('Fold:', n+1)
# Image
m = 2  #main image
img_name = ['ph0', 'ph1','ph2', 'ph3', 'T2'][m]
print('Image:', img_name)

# Choose max. epochs
max_epochs = 500
# Name of plot figure
exp_name = '_RN10_unfrozen_noseg_lowlr'
fig_name = '/trinity/home/agoedhart/Data/Plots/Plot_fold' + str(n+1) + '_' + img_name + exp_name + '.png'
csv_path = '/trinity/home/agoedhart/Data/Results/Resnet10/NoMasking/Data_fold_' + str(n+1) + '_' + img_name + exp_name + '.csv'
print('Experiment:', img_name + exp_name)


##################### DATA ######################################
# labels
label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)
print(df_label.Patient[0], df_label.Patient[101])

# images and segmentations
data_dir = "/trinity/home/agoedhart/Data/DL_Data"

images = sorted(glob.glob(os.path.join(data_dir,'*', img_name + '_bbox.nii.gz')))
#segmentations = sorted(glob.glob(os.path.join(data_dir,'*', img_name + '_bbox_seg.nii.gz')))
#images1 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[1] + '_bbox.nii.gz')))
#segmentations1 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[1] + '_bbox_seg.nii.gz')))
#images2 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[2] + '_bbox.nii.gz')))
#segmentations2 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[2] + '_bbox_seg.nii.gz')))
#images3 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[3] + '_bbox.nii.gz')))
#segmentations3 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[3] + '_bbox_seg.nii.gz')))
imagesT2 = sorted(glob.glob(os.path.join(data_dir,'*', 'T2_bbox.nii.gz')))
#segmentationsT2 = sorted(glob.glob(os.path.join(data_dir,'*', 'T2_bbox_seg.nii.gz')))

# data dictionary
data = []
labels = []
for i in range(0,len(images)):
    name = images[i].split(os.sep)[-2]
    labels.append(int(df_label[df_label["Patient"] == name]["Malignant"][i]))

#labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()  #for BCEWithLogitsLoss only

#for i in range(0,len(images)):   
#    data.append({"img":  images[i],  "seg": segmentations[i], 
#                 "img1": images1[i], "seg1": segmentations1[i], 
#                 "img2": images2[i], "seg2": segmentations2[i], 
#                 "img3": images3[i], "seg3": segmentations3[i], 
#                 "img4": images4[i], "seg4": segmentations4[i], 
#                 "label": label[i]})

for i in range(0,len(images)):   
    data.append({"img": images[i], "imgT2": imagesT2[i], "label": labels[i]})
  
print(images[0], images[101])
if use_T2 == True:
  print(imagesT2[0], imagesT2[101])
print('Labels:', labels)

# Pretraining weights
pre_path = "/trinity/home/agoedhart/Data/Pretrained/resnet_10_23dataset.pth"
###################### Data Transforms ###############################

    
# Transform for masking
class Maskingd(MapTransform):
    def __init__(self, keys, gt_keys):
        super().__init__(keys)
        self.gt_keys = gt_keys
    def __call__(self, data):
        for key, gt_key in zip(self.keys, self.gt_keys):
            data[key] = data[key] * (data[gt_key] > 0)
        return data

# Transform for the 3 coronal T2 images        
class Coronald(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        
    def __call__(self, data):
        for key in self.keys:
            if (data[key].shape)[-1] == 160:
                data[key] = torch.transpose(data[key],2,3)
        return data  
    
if __name__ == '__main__':


    if use_T2 == False  and data_aug == True: 
        in_channels = 1
        train_transforms = Compose(
                [
                    LoadImaged(keys=["img"]),
                    EnsureChannelFirstd(keys=["img"]),
                    Orientationd(keys=["img"], axcodes="RAS"),
                    Coronald(keys=["img"]),
                    # Preprocessing
                    NormalizeIntensityd(keys=["img"]),
                    # Data augmentation
                    RandZoomd(keys=["img"], prob = 0.3, min_zoom=1.0, max_zoom=2.0),
                    RandRotated(keys=["img"], range_z = 0.35, mode=("bilinear"), prob = 0.3),
                    RandFlipd(keys=["img"], prob = 0.5),
                    RandGaussianNoised(keys=["img"], prob = 0.5, std = 0.05),
                    #Maskingd(["img"],["seg"]),                
                    #ConcatItemsd(keys=["img", "seg"], name="img"),
                    ToTensord(keys=["img"])
                ]
            )
        val_transforms = Compose(
                [
                    LoadImaged(keys=["img"]),
                    EnsureChannelFirstd(keys=["img"]),
                    Orientationd(keys=["img"], axcodes="RAS"),
                    Coronald(keys=["img"]),
                    NormalizeIntensityd(keys=["img"]),                
                    #Maskingd(["img"],["seg"]),                
                    #ConcatItemsd(keys=["img", "seg"], name="img"),
                    ToTensord(keys=["img"])
                ]
            )

    if use_T2 == True and data_aug == True: 
        print('T2 was also used as input')
        in_channels = 2
        train_transforms = Compose(
                [
                    LoadImaged(keys=["img", 'imgT2']),
                    EnsureChannelFirstd(keys=["img", 'imgT2']),
                    Orientationd(keys=["img", 'imgT2'], axcodes="RAS"),
                    Coronald(keys=['imgT2']),
                    # Preprocessing
                    NormalizeIntensityd(keys=["img", 'imgT2']),
                    # Data augmentation
                    RandZoomd(keys=["img", 'imgT2'], prob = 0.3, min_zoom=1.0, max_zoom=2.0),
                    RandRotated(keys=["img", 'imgT2'], range_z = 0.35, mode=("bilinear"), prob = 0.3),
                    RandFlipd(keys=["img", 'imgT2'], prob = 0.5),
                    RandGaussianNoised(keys=["img", 'imgT2'], prob = 0.5, std = 0.05),
                    #Maskingd(["img"],["seg"]),                
                    ConcatItemsd(keys=["img", 'imgT2'], name="img"),
                    ToTensord(keys=["img", 'imgT2'])
                ]
            )
        val_transforms = Compose(
                [
                    LoadImaged(keys=["img", 'imgT2']),
                    EnsureChannelFirstd(keys=["img", 'imgT2']),
                    Orientationd(keys=["img", 'imgT2'], axcodes="RAS"),
                    Coronald(keys=['imgT2']),
                    NormalizeIntensityd(keys=["img", 'imgT2']),                
                    #Maskingd(["img"],["seg"]),                
                    ConcatItemsd(keys=["img", 'imgT2'], name="img"),
                    ToTensord(keys=["img", 'imgT2'])
                ]
            )

    ############################## K-fold Cross Validation ###############################

    # Shuffle data before cross-validation splits
    random.Random(random_seed).shuffle(data)
    cv_data = data
    
    # Images and labels for stratified k-fold splits
    crossval_img = [cv_data[i]['img'] for i in range(len(cv_data))]
    crossval_label = [cv_data[i]['label'] for i in range(len(cv_data))]

    
    # K-fold cross validation (stratified)
    k = 5
    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(crossval_img, crossval_label)
    
    # Weighted sampling
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, bs)

    # List of train and validation loaders
    train_cv = []
    val_cv = []
    train_ds_cv = []
    val_ds_cv = []

    for i, (train_index, val_index) in enumerate(skf.split(crossval_img, crossval_label)):
        # Split in folds
        train = [cv_data[j] for j in train_index]
        val = [cv_data[m] for m in val_index]
        # create a training data loader
        train_ds = Dataset(data=train, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=pin_memory)
        # create a validation data loader
        val_ds = Dataset(data=val, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=bs, num_workers=2, pin_memory=pin_memory)
        # append loaders
        train_cv.append(train_loader)
        val_cv.append(val_loader)
        train_ds_cv.append(train_ds)
        val_ds_cv.append(val_ds)
        
    # Data for fold n 
    train_loader = train_cv[n]
    val_loader = val_cv[n]
    train_ds = train_ds_cv[n]
    val_ds = val_ds_cv[n]
    


    # Loss function
    if lossfunc == 'BCEWithLogitsLoss': 
        loss_function = torch.nn.BCEWithLogitsLoss()  # add pos_weigth maybe
        n_out = 1
    if lossfunc == 'BCELoss':
        loss_function = torch.nn.BCELoss()
        n_out =1
    if lossfunc == 'CrossEntropyLoss':
        loss_function = torch.nn.CrossEntropyLoss() #weight=weight
        n_out = 2

    # Print model choices
    print('Architecture:', architecture)
    print('# input channels:', in_channels)
    print('# output channels:', n_out)
    print('Loss function:', lossfunc) 
    print('Pre-trained:', pre)
    print('Optimizer:', optimizer)
    print('learning rate:', lr)
    print('weight decay:', wd)
    print('batch size:', bs)
   
    
    ##################################### Classifier model ####################################
    # DenseNet
    if architecture == 'DenseNet': 
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels = in_channels, out_channels = n_out).to(device)
    # ResNet
    if architecture == 'ResNet': 
        model = resnet10(n_input_channels= in_channels, num_classes=1, widen_factor = wf)
        print('Widen factor:', wf)
        # Pretraining ResNet
        if pre == True:
            weights = torch.load(pre_path, map_location=torch.device('cpu'))['state_dict']
            if in_channels > 1:
              weights['module.conv1.weight'] = weights['module.conv1.weight'].repeat(1,in_channels,1,1,1)
            value_list = list(weights.values())
            key_list = list(weights.keys())
            key_list = [i[7:] for i in key_list]
            pre_weights = {}
            for key in key_list:
                for value in value_list:
                    pre_weights[key] = value
                    value_list.remove(value)
                    break    
            model_dict = model.state_dict()
            model_dict.update(pre_weights)
            model.load_state_dict(model_dict)
            
            # Frozen layers
            if frozen == True:
                for name, para in model.named_parameters():
                    para.requires_grad = False
                    model.layer4[0].conv1.weight.requires_grad = True
                    model.layer4[0].bn1.weight.requires_grad = True
                    model.layer4[0].conv2.weight.requires_grad = True
                    model.layer4[0].bn2.weight.requires_grad = True
                    model.layer4[0].downsample[0].weight.requires_grad = True
                    model.layer4[0].downsample[1].weight.requires_grad = True
                    model.fc.weight.requires_grad = True
    # MONAI Classifier
    if architecture == 'Classifier':
        model = Classifier(in_shape=[in_channels, 192, 160, 96], classes=1, channels=(2, 4, 8, 16), 
        strides=(2, 2, 2, 2), kernel_size=3, num_res_units=1, dropout = drop)
        print('Dropout:', drop)
  
    # Send model to GPU
    if torch.cuda.is_available():
        model.cuda()

    # Optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Softmax and Sigmoid functions
    softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    # AUC
    auc_metric = ROCAUCMetric()
    auroc = AUROC(task="binary")
    
    # Check train and test data 
    train_subjects = []
    test_subjects = []
    for i in range(len(val_ds)):
        f = val_ds.data[i]['img'][-19:-16]
        if img_name == 'T2':
           f = val_ds.data[i]['img'][-18:-15]
        test_subjects.append(f)
    for i in range(len(train_ds)):
        f = train_ds.data[i]['img'][-19:-16]
        if img_name == 'T2':
           f = val_ds.data[i]['img'][-18:-15]
        train_subjects.append(f)
        
    print('train samples:')
    print(train_subjects)
    print('test samples:')
    print(test_subjects)
    subjects = sorted(test_subjects + train_subjects) 
    print('all samples:')
    print(subjects)

    # Number of samples
    print('# training samples:', len(train_subjects))
    print('# test samples:', len(test_subjects))
    print('# total samples:', len(subjects))


    # Check val_loader and create image
    examples = iter(val_loader)         
    samples = []
    for i in range(20):
        samples.append(next(examples))    
       
    for i in range(20):
        a = samples[i]['img']
        l = samples[i]['label'].item()
        f = samples[i]['img_meta_dict']['filename_or_obj'][0][-19:-16]
        if img_name == 'T2':
            f = samples[i]['img_meta_dict']['filename_or_obj'][0][-18:-15]
        b = a.numpy()
        c = b[0,0,:,:,48]
        plt.figure("train_data", (12, 12))
        plt.subplot(4,5,i+1)
        plt.title(str(f) + ',' + str(l))
        plt.axis('off')
        plt.suptitle("Test data sample")
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.imshow(c, cmap="gray", origin="lower")
    plt.savefig('/trinity/home/agoedhart/Data/Plots/test_data3.png')

# Print model summary    
print(summary(model, (in_channels,192,160,96)))
    
    
