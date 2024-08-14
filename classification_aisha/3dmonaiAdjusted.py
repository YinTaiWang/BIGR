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
from monai.transforms import RandRotated, RandGaussianNoised, RandFlipd, RandRotate90d
from monai.transforms import EnsureChannelFirstd, MapTransform, ScaleIntensityd
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

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy, AUROC
from torchsummary import summary
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# For adjusted Resnet:
from functools import partial
from typing import Any, Callable, List, Tuple, Type, Union
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option
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
   
for i in range(0,len(images)):   
    data.append({"img": images[i], "seg": segmentations[i], 
                 "img_ph1": images1[i], "seg_ph1": segmentations1[i], 
                 "img_T2": images2[i], "seg_T2": segmentations2[i], 
                 "label": labels[i]})

print(images[0], images[101])
print('Labels:', labels)
#################### Method choices ################################
# Use segmentations as input
use_seg = True
# Use data augmentation
data_aug = True
# Use other phases and sequences
use_T2 = False
use_ph1 = False
# Choose architecture and pre-training
architecture = ['DenseNet', 'ResNet', 'Classifier'][1]
pre = False
# Choose loss functions and optimizer
lossfunc = ['BCEWithLogitsLoss', 'BCELoss', 'CrossEntropyLoss'][2]
if lossfunc == 'BCEWithLogitsLoss':
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float() #for BCEWithLogitsLoss only
# Choose random seed
random_seed = 1
# Choose learning rate and weight decay
optimizer = 'Adam'
lr = 1e-4
wd = 0 #1e-5
wf = 1
# Binary classification
n_classes = 2
# Choose max. epochs
max_epochs = 300
# Name of plot figure
fig_name = '/trinity/home/agoedhart/Data/Plots/Masked/plots_Rn10adj_bs4.png'
#fig_name = '/trinity/home/agoedhart/Data/Plots/plots_test.png'
auc_fig = '/trinity/home/agoedhart/Data/Plots/auc_Rn10adj_bs4.png'
print(fig_name)
###################### Pretraining weights ###############################
pre_path = "/trinity/home/agoedhart/Data/Pretrained/resnet_10_23dataset.pth"



########################## Adjusted Resnet10 ##################################
print("No batchnormalization and half width")
__all__ = [
    "ResNet",
    "ResNetBlock",
    "resnet10adj",
]


def get_inplanes():
    #return [64, 128, 512, 512]
    return [8, 16, 16, 32]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        #self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


        
class ResNet(nn.Module):

    def __init__(
        self,
        block: Union[Type[Union[ResNetBlock]], str],
        layers: List[int],
        block_inplanes: List[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[Tuple[int], int] = 7,
        conv1_t_stride: Union[Tuple[int], int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1,
        num_classes: int = 2,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ) -> None:

        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        #n_size = self._get_conv_output(input_shape)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1,1)
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: Type[Union[ResNetBlock]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = self.fc1(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x
        
def _resnet(
    arch: str,
    block: Type[Union[ResNetBlock]],
    layers: List[int],
    block_inplanes: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, bias_downsample=not pretrained, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
        )
    return model


def resnet10adj(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10adj", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)



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
        in_channels = 1
        train_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg"]),
                    EnsureChannelFirstd(keys=["img", "seg"]),
                    Orientationd(keys=["img", "seg"], axcodes="RAS"),
                    # Preprocessing
                    ScaleIntensityd(keys=["img"]),
                    # Data augmentation
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
                RandRotated(keys=["img", "img_ph1", "seg", "seg_ph1"], range_z = 0.35,
                           mode=("bilinear", "bilinear", "nearest", "nearest"), prob = 0.3),
                RandFlipd(keys=["img", "img_ph1", "seg", "seg_ph1"], prob = 0.5),
                RandGaussianNoised(keys=["img", "img_ph1"], prob = 1.0, std = 0.05),
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
    #crossval, test = sklearn.model_selection.train_test_split(data, test_size = 0.13, random_state=random_seed, stratify=labels)
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

    for i, (train_index, val_index) in enumerate(skf.split(crossval_img, crossval_label)):
        # Split in folds
        train = [cv_data[j] for j in train_index]
        val = [cv_data[m] for m in val_index]
        # create a training data loader
        train_ds = Dataset(data=train, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory)
        # create a validation data loader
        val_ds = Dataset(data=val, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, pin_memory=pin_memory)
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
    #for i in range(16):
    #    samples.append(next(examples))    
   
    #for i in range(16):
    #    a = samples[i]['img']
    #    l = samples[i]['label'].item()
    #    b = a.numpy()
    #    c = b[0,0,:,:,48]
    #    plt.figure("train_data", (10, 10))
    #    plt.subplot(4,4,i+1)
    #    plt.title(l)
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


    # Create a model
    if architecture == 'DenseNet': 
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels = in_channels, out_channels = n_out).to(device)
    if architecture == 'ResNet': 
        model = resnet10adj(pretrained = pre, n_input_channels= in_channels, num_classes=n_classes)
        print('Widen factor:', wf)
        if pre == True:
            weights = torch.load(pre_path)['state_dict']
            model.load_state_dict(weights)
    if architecture == 'Classifier':
        model = Classifier(in_shape=[1, 192, 160, 96], classes=2, channels=(8, 16, 32, 64, 128), strides=(2, 2, 2, 2), kernel_size=3, num_res_units=2)
  
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
            #print(inputs.size())
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            #print(labels)
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
            test_pred = []
            test_true = []
            for val_data in val_loader:
                val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)

                    # Validation accuracy
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()

                    # For AUC
                    val_sigmoid = sigmoid(val_outputs)
                    test_pred.append(val_sigmoid.cpu().detach().numpy())
                    test_true.append(val_labels.cpu().numpy())

                    #print(val_outputs)
                    #print(val_labels)

                    # Validation loss
                    loss2 = loss_function(val_outputs, val_labels) # Calculate loss
                    val_loss += loss2.item() # Set loss value
	    
            # Validation accuracy
            metric = num_correct / metric_count
            metric_values.append(metric)

            # Validation AUC
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            val_auc_mean =  roc_auc_score(test_true, test_pred[:,1]) 
            #print('val_auc_mean:', val_auc_mean)
            auc_values.append(val_auc_mean)
     

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

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


    #################### Check #############################################
    if lossfunc == 'CrossEntropyLoss':
        print('CrossEntropyLoss function check')
        print('Output:', outputs)
        print('Label:', labels)
        print('Softmax:', softmax(outputs))
        print('-log(softmax):', -torch.log(softmax(outputs))) # Cross Entropy Loss formula
        print('CE loss:', loss_function(outputs,labels)) # Cross Entropy Loss Pytorch
    
    if lossfunc == 'BCEWithLogitsLoss':
        print('BCEWithLogitsLoss function check')
        print('Output:', outputs)
        print('Label:', labels)
        print('Sigmoid:', sigmoid(outputs))
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

    # AUC plot
    print('AUC values:', auc_values)
    plt.figure("auc", (5,5))
    plt.title("Validation AUC")
    x = [i + 1 for i in range(len(auc_values))]
    y = auc_values
    plt.plot(x, y, color="b", label = 'val')
    plt.ylim([0, 1.1])
    plt.xlim([0, max_epochs])
    plt.savefig(auc_fig)