# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:57:53 2023

@author: aisha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:05:37 2023

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
#import scikitplot as skplt

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
#print_config()

##################### DATA ######################################
# labels
label_path = "/trinity/home/agoedhart/Classification/labels_all_phases_NEW.csv"
df_label = pd.read_csv(label_path)
print(df_label.Patient[0], df_label.Patient[101])

# images and segmentations
data_dir = "/trinity/home/agoedhart/Data/Scans"
img_name = ['ph2', 'ph1', 'T2']
images = sorted(glob.glob(os.path.join(data_dir,'*', img_name[0] + '_bbox.nii.gz')))
images1 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[1] + '_bbox.nii.gz')))
images2 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[2] + '_bbox.nii.gz')))
segmentations = sorted(glob.glob(os.path.join(data_dir,'*', img_name[0] + '_bbox_seg.nii.gz')))
segmentations1 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[1] + '_bbox_seg.nii.gz')))
segmentations2 = sorted(glob.glob(os.path.join(data_dir,'*', img_name[2] + '_bbox_seg.nii.gz')))

# data dictionary
data = []
labels = []
for i in range(0,len(images)):
    name = images[i].split(os.sep)[-2]
    labels.append(int(df_label[df_label["Patient"] == name]["Malignant"][i]))

#labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()  #for BCEWithLogitsLoss only

for i in range(0,len(images)):   
    data.append({"img": images[i], "seg": segmentations[i], 
                 "img_ph1": images1[i], "seg_ph1": segmentations1[i], 
                 "img_T2": images2[i], "seg_T2": segmentations2[i], 
                 "label": labels[i]})

print(images[0], images[101])
print('Labels:', labels)
#################### Method choices ################################
# Binary classification
n_classes = 1
# Use segmentations as input
use_seg = True
# Use data augmentation
data_aug = True
# Use other phases and sequences
use_T2 = False
use_ph1 = False
# Choose architecture and pre-training
architecture = ['DenseNet', 'ResNet', 'Classifier'][1]
# Choose loss functions and optimizer
lossfunc = ['BCEWithLogitsLoss', 'BCELoss', 'CrossEntropyLoss'][0]
# Choose random seed
random_seed = 42
# Choose learning rate and weight decay
optimizer = 'Adam'
lr = 1e-4
wd = 0 #1e-5
wf = 1
drop = 0.25
bs = 2 # batch size
pre = True
frozen = False
# Fold 
n=0
print('Fold:', n+1)
# Choose max. epochs
max_epochs = 1000
# Name of plot figure
#exp_name = '_RN10_pre_unfrozen'
exp_name = '_RN10_no_pretrain'
fig_name = '/trinity/home/agoedhart/Data/Plots/Masked/plots_CV_' + str(n+1) + exp_name + '.png'
csv_path = '/trinity/home/agoedhart/Data/Results/Resnet10/Data_fold_' + str(n+1) + exp_name + '.csv'
print('Experiment' + exp_name)
###################### Pretraining weights ###############################
pre_path = "/trinity/home/agoedhart/Data/Pretrained/resnet_10_23dataset.pth"
###################### Data Transforms ###############################
if use_seg == True:
    print('Masking used')
    
# Transform for masking
class Maskingd(MapTransform):
    def __init__(self, keys, gt_keys):
        super().__init__(keys)
        self.gt_keys = gt_keys
    def __call__(self, data):
        for key, gt_key in zip(self.keys, self.gt_keys):
            data[key] = data[key] * (data[gt_key] > 0)
        return data
    
if __name__ == '__main__':

    if use_T2 == False and use_ph1 == False and data_aug == True: 
        print('Only portal venous')
        in_channels = 1
        train_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"]),
                    EnsureChannelFirstd(keys=["img", "seg"]),
                    Orientationd(keys=["img", "seg"], axcodes="RAS"),
                    # Preprocessing
                    ScaleIntensityd(keys=["img"]),
                    # Data augmentation
                    RandZoomd(keys=["img", "seg"], prob = 0.3, min_zoom=1.0, max_zoom=2.0),
                    RandRotated(keys=["img", "seg"], range_z = 0.35, mode=("bilinear", "nearest"), prob = 0.3),
                    RandFlipd(keys=["img", "seg"], prob = 0.5),
                    RandGaussianNoised(keys=["img"], prob = 0.5, std = 0.05),
                    Maskingd(["img"],["seg"]),                
                    #ConcatItemsd(keys=["img", "seg"], name="img"),
                    #ToTensord(keys=["img", "seg"])                
                    ToTensord(keys=["img"])
                ]
            )
        val_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"]),
                    EnsureChannelFirstd(keys=["img", "seg"]),
                    Orientationd(keys=["img", "seg"], axcodes="RAS"),
                    ScaleIntensityd(keys=["img"]),                
                    Maskingd(["img"],["seg"]),                
                    #ConcatItemsd(keys=["img", "seg"], name="img"),
                    #ToTensord(keys=["img", "seg"])
                    ToTensord(keys=["img"])
                ]
            )


    if use_T2 == False and use_ph1 == True and data_aug == True: 
        print('ph1 was also used as input')
        print('Data augmentation was used')
        in_channels = 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ph1", "seg", "seg_ph1"]),
                EnsureChannelFirstd(keys=["img", "img_ph1", "seg", "seg_ph1"]),
                Orientationd(keys=["img", "img_ph1", "seg", "seg_ph1"], axcodes="RAS"),
                # Preprocessing
                ScaleIntensityd(keys=["img", "img_ph1"]),
                # Data augmentation
                RandZoomd(keys=["img", "img_ph1", "seg", "seg_ph1"], prob = 0.3, min_zoom=1.0, max_zoom=2.0),
                RandRotated(keys=["img", "img_ph1", "seg", "seg_ph1"], range_z = 0.35,
                           mode=("bilinear", "bilinear", "nearest", "nearest"), prob = 0.3),
                RandFlipd(keys=["img", "img_ph1", "seg", "seg_ph1"], prob = 0.5),
                RandGaussianNoised(keys=["img", "img_ph1"], prob = 0.5, std = 0.05),
                Maskingd(["img", "img_ph1"],
                         ["seg", "seg_ph1"]),
                ConcatItemsd(keys=["img", "img_ph1"], name="img"),
                ToTensord(keys=["img", "img_ph1"])
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "img_ph1", "seg", "seg_ph1"]),
                EnsureChannelFirstd(keys=["img", "img_ph1", "seg", "seg_ph1"]),
                Orientationd(keys=["img", "img_ph1", "seg", "seg_ph1"], axcodes="RAS"),
                # Preprocessing           
                ScaleIntensityd(keys=["img", "img_ph1"]),          
                Maskingd(["img", "img_ph1"],
                         ["seg", "seg_ph1"]),
                ConcatItemsd(keys=["img", "img_ph1"], name="img"),
                ToTensord(keys=["img", "img_ph1"])
            ]
        )

    ################################# Train and Val loader ###############################   
    # split train and val
    #train, val = sklearn.model_selection.train_test_split(data, test_size = 0.2, random_state=random_seed, stratify=labels)

    # create a training data loader
    #train_ds = Dataset(data=train, transform=train_transforms)
    #train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # create a validation data loader
    #val_ds = Dataset(data=val, transform=val_transforms)
    #val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    ############################## K-fold Cross Validation ###############################

    # split train, val and test set
    test = []

    random.Random(random_seed).shuffle(data)
    cv_data = data
    
    # Images and labels for stratified k-fold splits
    crossval_img = [cv_data[i]['img'] for i in range(len(cv_data))]
    crossval_label = [cv_data[i]['label'] for i in range(len(cv_data))]
    #crossval_label = [cv_data[i]['label'].argmax(dim=0) for i in range(len(cv_data))]

    
    # K-fold cross validation (stratified)
    k = 5
    skf = StratifiedKFold(n_splits=k)
    #sss = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=0)
    skf.get_n_splits(crossval_img, crossval_label)
    #sss.get_n_splits(crossval_img,crossval_label)

    # List of train and validation loaders
    train_cv = []
    val_cv = []

    # Weighted sampling
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, bs)

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

    # Number of samples
    print('Training samples:', len(train))
    print('Validation samples:', len(val))
    #print('# test:', len(test))

    # Check train_loader and create image
    #examples = iter(train_loader)         
    #samples = []
    #for i in range(25):
    #    samples.append(next(examples))    
   
    #for i in range(25):
    #    a = samples[i]['img']
    #    l = samples[i]['label'].item()
    #    f = samples[i]['img_meta_dict']['filename_or_obj'][0][-19:-16]
    #    b = a.numpy()
    #    c = b[0,0,:,:,48]
    #    plt.figure("train_data", (12, 12))
    #    plt.subplot(5,5,i+1)
    #    plt.title(str(f) + ',' + str(l))
    #    plt.axis('off')
    #    plt.suptitle("Training data sample")
    #    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    #    plt.imshow(c, cmap="gray", origin="lower")
    #plt.savefig('/trinity/home/agoedhart/Data/Plots/training_data.png')

    ##################################### Classifier  ############################################
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

    # Print choices
    print('Architecture:', architecture)
    print('# input channels:', in_channels)
    print('# output channels:', n_out)
    print('Loss function:', lossfunc) 
    print('Pre-trained:', pre)
    print('Optimizer:', optimizer)
    print('learning rate:', lr)
    print('weight decay:', wd)
    print('batch size:', bs)
   
    


    ##################################### Create a model ####################################
    if architecture == 'DenseNet': 
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels = in_channels, out_channels = n_out).to(device)
    if architecture == 'ResNet': 
        model = resnet10(n_input_channels= in_channels, num_classes=1, widen_factor = wf)
        print('Widen factor:', wf)
        if pre == True:
            weights = torch.load(pre_path)['state_dict']
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
            if frozen == True:
                for name, para in model.named_parameters():
                    para.requires_grad = False
                    #model.layer4[0].conv1.weight.requires_grad = True
                    #model.layer4[0].bn1.weight.requires_grad = True
                    #model.layer4[0].conv2.weight.requires_grad = True
                    #model.layer4[0].bn2.weight.requires_grad = True
                    #model.layer4[0].downsample[0].weight.requires_grad = True
                    #model.layer4[0].downsample[1].weight.requires_grad = True
                    model.fc.weight.requires_grad = True
    if architecture == 'Classifier':
        model = Classifier(in_shape=[1, 192, 160, 96], classes=1, channels=(2, 4, 8, 16), 
        strides=(2, 2, 2, 2), kernel_size=3, num_res_units=1, dropout = drop)
        print('Dropout:', drop)
  
    # Send to GPU
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

    # Loss and accuracy for all folds
    train_loss_cv = []
    val_loss_cv = []
    val_acc_cv = []
    train_acc_cv = []
    best_metrics = []
    best_models = []

    # Fold
    train_loader = train_cv[n]
    val_loader = val_cv[n]


    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    metric_values_train = []
    
    auc_values = []
    acc_values = []
    f1_values = []
    tn_values = []
    tp_values = []
    fn_values = []
    fp_values = []
    test_true_values = []
    test_prob_values = []
    
    auc_train_values = []
    acc_train_values = []
    f1_train_values = []
    tn_train_values = []
    tp_train_values = []
    fn_train_values = []
    fp_train_values = []
    train_true_values = []
    train_prob_values = []
    
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
        train_pred = []
        train_true = []

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['label'].to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            #print(labels)
            # Training loss
            loss = loss_function(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        
            # Training accuracy
            value_train = torch.eq(outputs.argmax(dim=1), labels) # See if output is equal to label
            metric_count_train += len(value_train) # Number of predictions
            num_correct_train += value_train.sum().item() # Total number of correct predictions

            # For AUC
            train_sigmoid = sigmoid(outputs).squeeze()
            #print('train_sig:', train_sigmoid.cpu().detach().numpy())
            train_pred.append(train_sigmoid.cpu().detach().numpy())
            train_true.append(labels.cpu().numpy())
        
        train_true = np.concatenate(train_true)
        train_true = [x for x in train_true]
        if len(train_true) == 81:
          train_pred[-1] = [train_pred[-1]]
        train_pred = np.concatenate(train_pred)
        train_pred = [x for x in train_pred]

    
        # Training loss
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #Training accuracy
        metric_train = accuracy_score(train_true, [round(x) for x in train_pred])
        metric_values_train.append(metric_train) # Fill accuracy list every epoch

        # Training metrics: consufion matrix, AUC, accuracy, f1_score
        train_auc_mean =  roc_auc_score(train_true, train_pred) 
        print('Train:')
        #print('Labels:', [int(x) for x in train_true])
        #print('Prob:', [x for x in train_pred])
        #print('Predic:', [round(x) for x in train_pred])
        print(confusion_matrix(train_true, [round(x) for x in train_pred], labels = [0,1])) 
        auc_train_values.append(train_auc_mean)    
        acc_train_values.append(accuracy_score(train_true, [round(x) for x in train_pred]))
        f1_train_values.append(f1_score(train_true, [round(x) for x in train_pred] , average = 'binary'))
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(train_true, [round(x) for x in train_pred]).ravel()
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
	    
            test_true = np.concatenate(test_true)
            test_true = [x for x in test_true]
            if len(test_true) == 21: #For batchsize 2,4,5,10
              test_pred[-1] = [test_pred[-1]]
            test_pred = np.concatenate(test_pred)
            test_pred = [x for x in test_pred]
                    
            # Validation accuracy
            metric = accuracy_score(test_true, [round(x) for x in test_pred])
            metric_values.append(metric)

            # Validation AUC
            val_auc_mean =  roc_auc_score(test_true, test_pred) 
            print('test prob:', [round(x,2) for x in test_pred])
            print('Labels:', [int(x) for x in test_true])
            print('Predic:', [round(x) for x in test_pred])
            print(confusion_matrix(test_true, [round(x) for x in test_pred], labels = [0,1])) 
            print('test acc:', metric) 
            print('test auc:', val_auc_mean)
            auc_values.append(val_auc_mean)
            acc_values.append(accuracy_score(test_true, [round(x) for x in test_pred]))
            f1_values.append(f1_score(test_true, [round(x) for x in test_pred], average = 'binary'))
            tn, fp, fn, tp = confusion_matrix(test_true, [round(x) for x in test_pred]).ravel()
            tn_values.append(tn)
            fp_values.append(fp)
            fn_values.append(fn)
            tp_values.append(tp)
            test_true_values.append(test_true)
            test_prob_values.append(test_pred)

            # Validation loss
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

                plt.savefig(fig_name)
                #################################################################################
                
                # Dataframe with values
                data_metrics = list(zip(epoch_loss_values, val_loss_values, acc_train_values, acc_values, auc_values, auc_train_values, f1_train_values, tn_train_values, tp_train_values, fn_train_values, fp_train_values, train_true_values, train_prob_values, f1_values, tn_values, tp_values, fn_values, fp_values, test_true_values, test_prob_values))
                df = pd.DataFrame(data_metrics, columns =['train loss', 'test loss', 'train acc', 'test acc', 'test auc', 'train auc', 'train f1', 'train tn', 'train tp','train fn','train fp','train true', 'train prob', 'test f1', 'test tn', 'test tp','test fn','test fp','test true', 'test prob'])
                df.to_csv(csv_path)  
                
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
        