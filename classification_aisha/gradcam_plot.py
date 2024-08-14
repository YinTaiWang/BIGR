# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:05:06 2023

@author: aisha
"""


#import ipdb
import logging
import os
import sys
import numpy as np
import glob
import pandas as pd
import monai
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from monai.networks.nets import resnet10
from monai.transforms import  Compose, LoadImaged, NormalizeIntensityd, ToTensord, ConcatItemsd, Orientationd
from monai.transforms import RandRotated, RandGaussianNoised, RandFlipd, RandZoomd
from monai.transforms import EnsureChannelFirstd, MapTransform
from monai.data import DataLoader, Dataset
from monai.visualize import GradCAM, blend_images

from ema_pytorch import EMA #EMA
import matplotlib.colors as clr

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
#####################################################################################

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#################### Method choices ################################
print('### Experiment ###')
# Binary classification
n_classes = 1
# Choose architecture and pre-training
architecture = 'ResNet'
# Choose loss functions and optimizer
lossfunc = 'BCEWithLogitsLoss'
# Choose random seed
random_seed = 42
# Hyperparameters
optimizer = 'Adam'
lr = 1e-4
wd = 0 #1e-5
wf = 1
bs = 2 # batch size
frozen = False
gradcam = True
# EMA
use_ema = True
# Use masks as input
use_seg = False
# Pretrained weights ResNet10
pre = False




# Choose max. epochs
max_epochs = 2
# Fold 
n=0
print('Fold:', n+1)
# Image
m = 3  #main image
img_names = ['ph0', 'ph1','ph2', 'ph3', 'T2']
# Subject
s = 4

# Weights
if pre == False and use_seg == True:
  pre_path = "/trinity/home/agoedhart/fold" + str(n+1) + "_all_phases_T2_RN10_gradcam_model.pth"
  exp_name = '_seg_nopre'
if pre == True and use_seg == True:
  pre_path = "/trinity/home/agoedhart/fold" + str(n+1) + "_all_phases_T2_RN10_gradcam_pretrained_model.pth"
  exp_name = '_seg_pre'
if pre == True and use_seg == False:
  pre_path = "/trinity/home/agoedhart/fold" + str(n+1) + "_all_phases_T2_RN10_gradcam_noseg_pretrained_model.pth"
  exp_name = '_noseg_pre'
if pre == False and use_seg == False:
  pre_path = "/trinity/home/agoedhart/fold" + str(n+1) + "_all_phases_T2_RN10_gradcam_noseg_model.pth"
  exp_name = '_noseg_nopre'

print('seg:', use_seg)
print('pre:', pre)  
print('CHECK:', pre_path, exp_name)
  
##### Use all phases, use T2 too, no contrast-enhancement combo
    # Individual images:
#all_phases, use_T2,  ph0_T2 = False, False, False
    # All phases + T2:
all_phases, use_T2,  ph0_T2 = True, True, False    
    # All phases:
#all_phases, use_T2,  ph0_T2 = True, False, False
    # Precontrast + T2:
#all_phases, use_T2,  ph0_T2 = False, True, True

# Name of plot figure



# Individual images
if all_phases == False and use_T2 == False and ph0_T2 == False:
    img_name = img_names[m]
    csv_name = 'fold' + str(n+1) + '_' + img_name + exp_name + '.csv'
    fig_name = 'fold' + str(n+1) + '_' + img_name + exp_name + '.png'
# All phases + T2
if all_phases == True and use_T2 == True and ph0_T2 == False:
    img_name = img_names[0]
    csv_name = 'fold' + str(n+1) + '_' + 'all_phases_T2' + exp_name + '.csv'
    fig_name = 'fold' + str(n+1) +  '_' + 'all_phases_T2' + exp_name + '.png'
# All phases
if all_phases == True and use_T2 == False and ph0_T2 == False:
    img_name = img_names[0]
    csv_name = 'fold' + str(n+1) + '_' + 'all_phases' + exp_name + '.csv'
    fig_name = 'fold' + str(n+1) + '_' + 'all_phases' + exp_name + '.png'
# Precontrast + T2
if all_phases == False and use_T2 == True and ph0_T2 == True:
    img_name = img_names[0]
    csv_name = 'fold' + str(n+1) + '_' + 'ph0_T2' + exp_name + '.csv'
    fig_name = 'fold' + str(n+1) + '_' + 'ph0_T2' + exp_name + '.png'

fig_path = '/trinity/home/agoedhart/Data/Plots/Plot_' + fig_name
csv_path = '/trinity/home/agoedhart/Data/Results/Resnet10/WithSegmentation/No_pretraining/Results_' + csv_name
print('Experiment:', csv_name[:-4])
print(fig_name)


####################### DATA ####################################
print('### Data ###')

# Labels
label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)

# Images and masks
data_dir = "/trinity/home/agoedhart/Data/DL_Data"

# Get all image and mask paths    
images =   sorted(glob.glob(os.path.join(data_dir,'*', img_name + '_bbox.nii.gz')))
masks =    sorted(glob.glob(os.path.join(data_dir,'*', img_name + '_bbox_seg.nii.gz')))
images1 =  sorted(glob.glob(os.path.join(data_dir,'*', img_names[1] + '_bbox.nii.gz')))
masks1 =   sorted(glob.glob(os.path.join(data_dir,'*', img_names[1] + '_bbox_seg.nii.gz')))
images2 =  sorted(glob.glob(os.path.join(data_dir,'*', img_names[2] + '_bbox.nii.gz')))
masks2 =   sorted(glob.glob(os.path.join(data_dir,'*', img_names[2] + '_bbox_seg.nii.gz')))
images3 =  sorted(glob.glob(os.path.join(data_dir,'*', img_names[3] + '_bbox.nii.gz')))
masks3 =   sorted(glob.glob(os.path.join(data_dir,'*', img_names[3] + '_bbox_seg.nii.gz')))
imagesT2 = sorted(glob.glob(os.path.join(data_dir,'*', 'T2_bbox.nii.gz')))
masksT2 =  sorted(glob.glob(os.path.join(data_dir,'*', 'T2_bbox_seg.nii.gz')))

# data dictionary
data = []
labels = []

#Get labels in right order of subjects
for i in range(0,len(images)):
    name = images[i].split(os.sep)[-2]
    labels.append(int(df_label[df_label["Patient"] == name]["Malignant"][i]))

# Individual images
if all_phases == False and use_T2 == False and ph0_T2 == False:
    one_phase = True
    for i in range(0,len(images)):   
        data.append({"img": images[i], "seg": masks[i], 
                     "label": labels[i]})
    # Dataloader 
    bs = 2   
    in_channels = 2
    keys=["img", "seg"]
    img_keys = ["img"]
    mode=("bilinear", "nearest")

# All phases + T2
if all_phases == True and use_T2 == True and ph0_T2 == False: 
    for i in range(0,len(images)):   
        data.append({"img":   images[i],   "seg":  masks[i], 
                     "img1":  images1[i],  "seg1": masks1[i], 
                     "img2":  images2[i],  "seg2": masks2[i], 
                     "img3":  images3[i],  "seg3": masks3[i], 
                     "img4":  imagesT2[i], "seg4": masksT2[i], 
                     "label": labels[i]})
    # Dataloader
    in_channels = 10
    bs = 2
    keys=["img", "img1", "img2", "img3", "img4",
          "seg", "seg1", "seg2", "seg3", "seg4"]
    img_keys = ["img", "img1", "img2", "img3", "img4"]
    mode=("bilinear","bilinear","bilinear","bilinear","bilinear",
          "nearest", "nearest", "nearest", "nearest", "nearest")
          
    if use_seg == False:
      print('NO segmentation')
      in_channels = 5
      bs = 2
      keys=["img", "img1", "img2", "img3", "img4"]
      img_keys = ["img", "img1", "img2", "img3", "img4"]
      mode=("bilinear","bilinear","bilinear","bilinear","bilinear")
    
# All phases
if all_phases == True and use_T2 == False and ph0_T2 == False:
    for i in range(0,len(images)):   
        data.append({"img":   images[i],   "seg":  masks[i], 
                     "img1":  images1[i],  "seg1": masks1[i], 
                     "img2":  images2[i],  "seg2": masks2[i], 
                     "img3":  images3[i],  "seg3": masks3[i], 
                     "label": labels[i]})
    # Dataloader
    in_channels = 8
    bs = 2
    keys=["img", "img1", "img2", "img3",
          "seg", "seg1", "seg2", "seg3"]
    img_keys = ["img", "img1", "img2", "img3"]
    mode=("bilinear","bilinear","bilinear","bilinear",
          "nearest", "nearest", "nearest", "nearest")
    
# Precontrast + T2    
if all_phases == False and use_T2 == True and ph0_T2 == True:
    for i in range(0,len(images)):   
        data.append({"img":   images[i],   "seg":  masks[i], 
                     "img4":  imagesT2[i], "seg4": masksT2[i], 
                     "label": labels[i]})
    # Dataloader
    bs = 2
    in_channels = 4
    keys=["img", "img4", "seg", "seg4"]
    img_keys = ["img", "img4"]
    mode=("bilinear","bilinear", "nearest", "nearest")
 
# Print first and last subject as check
print(images[0][-32:],   images[101][-32:])
print(images1[0][-32:],  images1[101][-32:])
print(images2[0][-32:],  images2[101][-32:])
print(images3[0][-32:],  images3[101][-32:])
print(imagesT2[0][-31:], imagesT2[101][-31:])
  
# Print labels as check
print('Labels:', labels)

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
    
    # Data transforms
    train_transforms = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                # Preprocessing
                Coronald(keys=keys),
                NormalizeIntensityd(keys=img_keys),
                # Data augmentation
                RandZoomd(keys=keys, prob = 0.3, min_zoom=1.0, max_zoom=2.0),
                RandRotated(keys=keys, range_z = 0.35, mode=mode, prob = 0.3),
                RandFlipd(keys=keys, prob = 0.5),
                RandGaussianNoised(keys=img_keys, prob = 0.5, std = 0.05),
                # To dataloader
                ConcatItemsd(keys=keys, name="img"),
                ToTensord(keys=keys)
            ]
        )
    val_transforms = Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                # Preprocessing
                Coronald(keys=keys),
                NormalizeIntensityd(keys=img_keys),    
                # To dataloader
                ConcatItemsd(keys=keys, name="img"),
                ToTensord(keys=keys)
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
           f = train_ds.data[i]['img'][-18:-15]
        train_subjects.append(f)
        
    print('Train samples:', len(train_subjects))
    print(train_subjects)
    print('Test samples:', len(test_subjects))
    print(test_subjects)
    subjects = sorted(test_subjects + train_subjects) 
    print('All samples:', len(subjects))
    print(subjects)


    # Print model choices
    print('### Hyperparameters ###')
    print('Architecture:', architecture)
    print('Input channels:', in_channels)
    print('Output channels:', n_classes)
    print('Loss function:', lossfunc) 
    print('Pre-trained:', pre)
    print('Optimizer:', optimizer)
    print('Learning rate:', lr)
    print('Weight decay:', wd)
    print('Batch size:', bs)
    print('Widen factor:', wf)
    print('EMA:', use_ema)
    
    ##################################### Classifier model ####################################
    
    # ResNet
    
    print(in_channels)
    print(pre_path)
    
    if architecture == 'ResNet': 
        model = resnet10(n_input_channels= in_channels, num_classes=n_classes, widen_factor = wf)
        weights = torch.load(pre_path)#['state_dict']
        model_dict = model.state_dict()
        model_dict.update(weights)
        model.load_state_dict(model_dict)
  
    
    # Send model to GPU
    if torch.cuda.is_available():
        model.cuda()
        
    if gradcam == True:
       
      #data_tensor = next(iter(val_loader))
      examples = iter(val_loader)         
      samples = []
      for i in range(s+1):
          samples.append(next(examples))  
      print(len(samples))  
      print('k is ', k)
      
      data_tensor = samples[s]
      
      img_tensor = data_tensor['img'] 
      img_tensor = img_tensor.cuda()
      
      if use_seg == True:
        seg_tensor = data_tensor['seg'] 
        seg_tensor = seg_tensor.cuda()
      
      #data_tensor = data_tensor.cuda()
      normalize = clr.Normalize(vmin=0, vmax=1)
      
      cam = GradCAM(nn_module=model, target_layers="layer4.0.conv2") #layer4
      result = cam(img_tensor) #val_images
      result = result[0,0,:,:,48]
      result = result.squeeze(0).cpu()
      result = (result.numpy()).T
      result = np.rot90(result,2)
      #print('numpy result:', np.shape(result))
      plt.figure("gradcam", (5, 5))
      plt.imshow(result, cmap = 'rainbow')
      plt.colorbar()
      plt.axis('off')
      plt.savefig('/trinity/home/agoedhart/Data/Plots/gradcam2')
      
      img2d = img_tensor[0,0,:,:,48]
      img2d = (img2d.cpu()).T
      img2d = np.rot90(img2d,2)
      plt.figure("image2d", (5, 5))
      plt.imshow(img2d, cmap = 'gray')
      plt.axis('off')
      plt.savefig('/trinity/home/agoedhart/Data/Plots/img2D2')
      
      if use_seg == True:
        seg2d = seg_tensor[0,0,:,:,48]
        seg2d = (seg2d.cpu()).T
        seg2d = np.rot90(seg2d,2)
        plt.figure("seg2d", (5, 5))
        plt.imshow(seg2d, cmap = 'gray')
        plt.axis('off')
        plt.savefig('/trinity/home/agoedhart/Data/Plots/seg2D2')
      
      #blend_img = blend_images(image=img2d, label=result, alpha=0.5, rescale_arrays=False)
      #plt.imshow(blend)
      fig=plt.figure()
      ax = plt.axes()
      plt.figure("image_gradcam")
      plt.imshow(img2d, cmap = 'gray')
      im = plt.imshow(result, cmap = 'rainbow', alpha = 0.5, norm = normalize)
      #plt.colorbar()
      plt.colorbar(im,fraction=0.038, pad=0.04)
      #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
      #plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
      plt.axis('off')
      plt.savefig('/trinity/home/agoedhart/Data/Plots/GradCAM/grad_img_' + csv_name[:-4] + str(s))
      
      if use_seg == True:
        plt.figure("image_seg")
        plt.imshow(img2d, cmap = 'gray')
        plt.imshow(seg2d, cmap = clr.ListedColormap(['black', 'red']), alpha = 0.3*(seg2d>0))
        
        #blend_img = blend_images(image=img2d, label=seg2d, alpha=0.5, rescale_arrays=False)
        #plt.imshow(blend)
        plt.axis('off')
        plt.savefig('/trinity/home/agoedhart/Data/Plots/GradCAM/seg_img_' + csv_name[:-4] + str(s))
  
        plt.figure("gradcam_seg", (5, 5))
        plt.imshow(seg2d, cmap = 'gray')
        plt.imshow(result, cmap = 'rainbow', alpha = 0.5, norm = normalize)
        plt.colorbar()
        plt.axis('off')
        
        #plt.savefig('/trinity/home/agoedhart/Data/Plots/GradCAM/seg_grad_' + csv_name[:-4] + str(s))
    