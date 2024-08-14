# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:53:21 2023

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
from monai.visualize import GradCAM

from ema_pytorch import EMA #EMA

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
wf = 0.5
bs = 2 # batch size
# EMA
use_ema = True
# Pretraining weights
pre_path = "/trinity/home/agoedhart/Data/Pretrained/resnet_10_23dataset.pth"
# Use masks as input
use_seg = True
# Pretrained weights ResNet10
pre = False
frozen = False
gradcam = False


# Choose max. epochs
max_epochs = 200
# Fold 
n=4
print('Fold:', n+1)
# Image
m = 2  #main image
img_names = ['ph0', 'ph1','ph2', 'ph3', 'T2']


##### Use all phases, use T2 too, no contrast-enhancement combo
    # Individual images:
all_phases, use_T2,  ph0_T2 = False, False, False
    # All phases + T2:
#all_phases, use_T2,  ph0_T2 = True, True, False    
    # All phases:
#all_phases, use_T2,  ph0_T2 = True, False, False
    # Precontrast + T2:
#all_phases, use_T2,  ph0_T2 = False, True, True

# Name of plot figure
exp_name = '_RN10_halfwf'
#exp_name = '_EMA_test_0_999'


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
csv_path = '/trinity/home/agoedhart/Data/Results/Resnet10/WithSegmentation/Pretraining/Results_' + csv_name
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
    if architecture == 'ResNet': 
        model = resnet10(n_input_channels= in_channels, num_classes=n_classes, widen_factor = wf)
        # Pretraining ResNet
        if pre == True:
            weights = torch.load(pre_path)['state_dict']
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
  
    # Send model to GPU
    if torch.cuda.is_available():
        model.cuda()
        
    # Loss function 
    if lossfunc == 'BCEWithLogitsLoss': 
        loss_function = torch.nn.BCEWithLogitsLoss()  # add pos_weigth maybe
    
    # Optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)

    # Sigmoid functions
    sigmoid = nn.Sigmoid()
    
    #EMA
    ema = EMA(
    model,
    beta = 0.9999, #0.9999,   # exponential moving average factor
    update_after_step = 5,    # only after this number of .update() calls will it start updating
    update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
    )
    ema.update() #EMA


    # Loss and accuracy for all folds
    train_loss_cv = []
    val_loss_cv = []
    val_acc_cv = []
    train_acc_cv = []
    best_metrics = []
    best_models = []

    # Start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    
    # Loss and standard metric
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    metric_values_train = []
    
    # Test AUC, accuracy, F1-score, tn, tp, fn, fp, true, probability
    auc_values = []
    acc_values = []
    f1_values = []
    tn_values = []
    tp_values = []
    fn_values = []
    fp_values = []
    test_true_values = []
    test_prob_values = []
    
    # Training AUC, accuracy, F1-score, tn, tp, fn, fp, true, probability
    auc_train_values = []
    acc_train_values = []
    f1_train_values = []
    tn_train_values = []
    tp_train_values = []
    fn_train_values = []
    fp_train_values = []
    train_true_values = []
    train_prob_values = []
    
    # Writer
    writer = SummaryWriter()
    
    # Training/Test loop
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        val_loss = 0
        step = 0
        metric_count_train = 0
        num_correct_train = 0.0
        train_pred = []
        train_true = []

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['label'].to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Training loss
            loss = loss_function(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            ema.update() #EMA
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
            # Training accuracy
            value_train = torch.eq(outputs.argmax(dim=1), labels) # See if output is equal to label
            metric_count_train += len(value_train) # Number of predictions
            num_correct_train += value_train.sum().item() # Total number of correct predictions

            # Probability and true labels
            train_sigmoid = sigmoid(outputs).squeeze()
            train_pred.append(train_sigmoid.cpu().detach().numpy())
            train_true.append(labels.cpu().numpy())
        
        # True labels and prediction
        train_true = np.concatenate(train_true)
        train_true = [x for x in train_true]
        #print(train_pred)
        if len(train_true) == 81 and bs == 2:
          train_pred[-1] = [train_pred[-1]]
        #print(train_pred)
        if bs == 2:
            train_pred = np.concatenate(train_pred)
        train_pred = [x for x in train_pred]
    
        # Training loss
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #Training accuracy
        metric_train = accuracy_score(train_true, [np.round(x) for x in train_pred])
        metric_values_train.append(metric_train) # Fill accuracy list every epoch

        # Training metrics: consufion matrix, AUC, accuracy, f1_score
        print('Train:')
        print(confusion_matrix(train_true, [np.round(x) for x in train_pred], labels = [0,1])) 
        train_auc_mean =  roc_auc_score(train_true, train_pred) 
        auc_train_values.append(train_auc_mean)    
        acc_train_values.append(accuracy_score(train_true, [np.round(x) for x in train_pred]))
        f1_train_values.append(f1_score(train_true, [np.round(x) for x in train_pred] , average = 'binary'))
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_true, [np.round(x) for x in train_pred]).ravel()
        tn_train_values.append(tn_train)
        fp_train_values.append(fp_train)
        fn_train_values.append(fn_train)
        tp_train_values.append(tp_train)
        train_true_values.append(train_true)
        train_prob_values.append(train_pred)


        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            prob = []
            target = []
            test_pred = []
            test_true = []
            print('Test:')
            
            for val_data in val_loader:
                val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    # Exponential Moving Average
                    if use_ema == True: 
                      val_outputs = ema(val_images) #EMA
                    val_labels = val_labels.float()

                    # Test accuracy
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()

                    # For AUC
                    val_sigmoid = sigmoid(val_outputs).squeeze()
                    test_pred.append(val_sigmoid.cpu().detach().numpy())
                    test_true.append(val_labels.cpu().numpy())
                   
                    # Test loss
                    loss2 = loss_function(val_outputs, val_labels.unsqueeze(1)) # Calculate loss
                    val_loss += loss2.item() # Set loss value
	    
            # Test labels and prediction
            test_true = np.concatenate(test_true)
            test_true = [x for x in test_true]
            if len(test_true) == 21 and bs > 1: #For batchsize 2,4,5,10
              test_pred[-1] = [test_pred[-1]]
            if bs == 2:
                test_pred = np.concatenate(test_pred)
            test_pred = [x for x in test_pred]
                    
            # Test accuracy
            metric = accuracy_score(test_true, [np.round(x) for x in test_pred])
            metric_values.append(metric)

            # Test metrics
            val_auc_mean =  roc_auc_score(test_true, test_pred) 
            print('test prob:', [np.round(x,2) for x in test_pred])
            print('Labels:', [int(x) for x in test_true])
            print('Predic:', [int(np.round(x)) for x in test_pred])
            print(confusion_matrix(test_true, [np.round(x) for x in test_pred], labels = [0,1])) 
            print('test acc:', metric) 
            print('test auc:', val_auc_mean)
            auc_values.append(val_auc_mean)
            acc_values.append(accuracy_score(test_true, [np.round(x) for x in test_pred]))
            f1_values.append(f1_score(test_true, [np.round(x) for x in test_pred], average = 'binary'))
            tn, fp, fn, tp = confusion_matrix(test_true, [np.round(x) for x in test_pred]).ravel()
            tn_values.append(tn)
            fp_values.append(fp)
            fn_values.append(fn)
            tp_values.append(tp)
            test_true_values.append(test_true)
            test_prob_values.append(test_pred)

            # Test loss
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

            if (epoch+1) % 10 == 0:
                ######################### Performance plot #############################
                plt.figure("train2", (10, 10))

                plt.subplot(2, 2, 1)
                plt.title("Train & Val Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                z = val_loss_values
                plt.plot(x, y, color="r", label = 'train')
                plt.plot(x, z, color="b", label = 'val')
                plt.xlabel("epoch")
                plt.ylim([0, 2])
                plt.xlim([0, len(epoch_loss_values)])
                #plt.legend()

                plt.subplot(2, 2, 2)
                plt.title("Train Loss")
                x = [i + 1 for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("epoch")
                plt.plot(x, y, color="r", label = 'train')
                plt.ylim([0, 1])
                plt.xlim([0, len(epoch_loss_values)])
                #plt.legend()

                plt.subplot(2, 2, 3)
                plt.title("Train & Val accuracy")
                x = [val_interval * (i + 1) for i in range(len(metric_values_train))]
                y = acc_train_values
                z = acc_values
                plt.xlabel("epoch")
                plt.plot(x, y, color="r", label = 'train')
                plt.plot(x, z, color="b", label = 'val')
                plt.ylim([0, 1.1])
                plt.xlim([0, len(epoch_loss_values)])
                #plt.legend()

                plt.subplot(2, 2, 4)
                plt.title("AUC values")
                x = [i + 1 for i in range(len(auc_values))]
                y = auc_train_values
                z = auc_values
                plt.plot(x, y, color="r", label = 'train')
                plt.plot(x, z, color="b", label = 'test')
                plt.axhline(y = 0.5, color = 'k', linestyle = '-')
                plt.ylim([0, 1.1])
                plt.xlim([0, len(epoch_loss_values)])
                #plt.legend()

                plt.savefig(fig_path)
                #################################################################################
                
                # Dataframe with values
                data_metrics = list(zip(epoch_loss_values, val_loss_values, acc_train_values, acc_values, auc_values, auc_train_values, f1_train_values, tn_train_values, tp_train_values, fn_train_values, fp_train_values, train_true_values, train_prob_values, f1_values, tn_values, tp_values, fn_values, fp_values, test_true_values, test_prob_values))
                df = pd.DataFrame(data_metrics, columns =['train loss', 'test loss', 'train acc', 'test acc', 'test auc', 'train auc', 'train f1', 'train tn', 'train tp','train fn','train fp','train true', 'train prob', 'test f1', 'test tn', 'test tp','test fn','test fp','test true', 'test prob'])
                df.to_csv(csv_path)  
                
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    
    
    if gradcam == True:
      cam = GradCAM(nn_module=model, target_layers="layer4") #layer4[0].relu
      result = cam(val_images)
    # if one_phase == True and use_seg == False: 
    #     in_channels = 1
    #     keys = ["img"]
    #     train_transforms = Compose(
    #             [
    #                 LoadImaged(keys=keys),
    #                 EnsureChannelFirstd(keys=keys),
    #                 Orientationd(keys=keys, axcodes="RAS"),
    #                 Coronald(keys=keys),
    #                 # Preprocessing
    #                 NormalizeIntensityd(keys=keys),
    #                 # Data augmentation
    #                 RandZoomd(keys=keys, prob = 0.3, min_zoom=1.0, max_zoom=2.0),
    #                 RandRotated(keys=keys, range_z = 0.35, mode=("bilinear"), prob = 0.3),
    #                 RandFlipd(keys=keys, prob = 0.5),
    #                 RandGaussianNoised(keys=keys, prob = 0.5, std = 0.05),
    #                 ToTensord(keys=keys)
    #             ]
    #         )
    #     val_transforms = Compose(
    #             [
    #                 LoadImaged(keys=keys),
    #                 EnsureChannelFirstd(keys=keys),
    #                 Orientationd(keys=keys, axcodes="RAS"),
    #                 Coronald(keys=keys),
    #                 NormalizeIntensityd(keys=keys),                
    #                 ToTensord(keys=keys)
    #             ]
    #         )