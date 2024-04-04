import os
import sys
import csv
import glob
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import monai
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.utils import set_determinism
from transforms.set_transforms import training_transforms, post_transfroms

from model.douwe_unet import UNet
from model.monai_unet import MOANI_UNet
from model.groupreg_unet import GroupReg_UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from model.utils import Sanity_check, write_csv

####################
##   Settings     ##
####################
TEST_MODE = False # use only a small part of data
MODEL = "Douwe" # Douwe / MONAI / GroupReg
MAX_EPOCHS = 1000
LR = 0.01
LR_SCHEDULER = True
KERNEL_SIZE = (3, 3, 3)
ACCUMULATE_GRAD_BATCHES = 1

TRAIN_SCALE = 0.5
VAL_SCALE = 0.5
VAL_INTERVAL = 1 # always set to 1 to avoid some issue in plotting
CHECK_INTERVAL = 50
SLIDING_WINDOW = False

print("##############")
print("## Settings ##")
print("##############")
print(f"network: {MODEL}; kernel_size: {KERNEL_SIZE}")
print(f"scale in training set: {TRAIN_SCALE}; scale in validation set: {VAL_SCALE}")
print(f"learning rate: {LR}; learning rate scheduler: {LR_SCHEDULER}")
print(f"training epochs: {MAX_EPOCHS}; accumulate grad batch: {ACCUMULATE_GRAD_BATCHES}\n")

################################## MAIN ##################################
def main():
    
    ####################
    ##      Setup     ##
    ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    set_determinism(seed=0)
    
    script_dir = os.getcwd()
    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False

    if GPU_cluster:
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_resampled_images_newseg"
        seg3D_dir = "/data/scratch/r098906/BLT_radiomics/segs_new_resampled"
        model_folder = os.path.join(script_dir, "..", "BLT", "data", "BLT_radiomics", "segnet")
    else:
        img4D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/4D_resampled_images_newseg"
        seg3D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_resampled"
        model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "segnet")
    
    model_dir = os.path.join(model_folder, attempt)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"GPU cluster: {GPU_cluster}; Device: {device}")
    print(f"The results will be saved at: {model_dir}")
    
    ####################
    ### Load data
    # images and segmentations
    images = sorted(glob.glob(os.path.join(img4D_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg3D_dir, "*.nii.gz")))
    
    #test
    if TEST_MODE:
        images = images[:3]
        segs = segs[:3]
    
    data_dicts = [{"image": image, "seg": seg} for image, seg in zip(images, segs)]

    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\n\n")

    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=1)
    train_files, val_files = [data_dicts[i] for i in train_idx], [data_dicts[i] for i in val_idx]
        
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = training_transforms(seed=0)
    val_transforms = training_transforms(validation=True)
    post_pred = post_transfroms()
    post_label = post_transfroms(label=True)
    
    ####################
    ##   Dataloader   ##
    ####################
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    
    ###############
    ##   Model   ##
    ###############
    
    if MODEL == "Douwe":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            strides=(1,)*(len(KERNEL_SIZE)),
            kernel_size = KERNEL_SIZE,
            upsample_kernel_size = (1,)*(len(KERNEL_SIZE)),
            activation = 'LRELU',
            normalisation = 'instance'
            ).to(device)
    elif MODEL == "MONAI":
        model = MOANI_UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128),
            strides=(1,1,1),
            num_res_units=0,
            norm='instance'
            ).to(device)
    elif MODEL == "GroupReg":
        model = GroupReg_UNet(
            in_channels = 1, 
            out_channels = 2, 
            dim = 3, 
            depth = 3, 
            initial_channels = 32, 
            normalization = True
            ).to(device)
    else:
        raise ValueError("The model only supports the following options: 'Douwe', 'MONAI', or 'GroupReg'.")
    
    ema = EMA(
        model,
        beta = 0.9999, #0.9999,   # exponential moving average factor
        update_after_step = 5,    # only after this number of .update() calls will it start updating
        update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    ema.update()
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    if LR_SCHEDULER:
        lambda_poly = lambda epoch: (1 - epoch / MAX_EPOCHS) ** 0.9
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_poly)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    

    ###############
    ##   Train   ##
    ###############
    print("##############")
    print("## Training ##")
    print("##############")
    best_metric = -1
    best_metric_epoch = -1
    history = {'train_epoch_loss': [],
        'val_epoch_loss': [],
        'train_metric_values': [],
        'val_metric_values': []
    }
    
    start_time = time.time()
    for epoch in range(MAX_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0
        epoch_start_time = time.time()
        for x, train_data in enumerate(train_loader):
            step += 1
            image, seg = (
                train_data["image"].to(device),
                train_data["seg"].to(device),
            )
            
            if TRAIN_SCALE < 1:
                image = F.interpolate(image[:,1:2,:,:,:], 
                                         scale_factor = TRAIN_SCALE, 
                                         align_corners = True, 
                                         mode = 'trilinear', 
                                         recompute_scale_factor = False)
                seg = F.interpolate(seg.to(torch.float32), 
                                       scale_factor = TRAIN_SCALE, 
                                       mode ='nearest', 
                                       recompute_scale_factor = False)
            else:
                image = image[:,1:2,:,:,:]
                seg = seg
                
            optimizer.zero_grad()
            outputs = model(image)
            
            # dice metric
            post_tr_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            post_tr_seg = [post_label(i) for i in decollate_batch(seg)]
            dice_metric(y_pred=post_tr_outputs, y=post_tr_seg)
            
            # loss
            loss = loss_function(outputs, seg)
            loss.backward()
            
            # sanity check
            if (epoch + 1) % CHECK_INTERVAL == 0:
                check_dir = os.path.join(model_dir, 'train_check')
                if not os.path.exists(check_dir):
                    print("Create 'train_check' folder for validation output.")
                    os.makedirs(check_dir, exist_ok=True)
                save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                Sanity_check.check_seg(outputs = outputs, 
                                        data = (image, seg), 
                                        save_dir = save_dir)
            
            # update weight & gradient accumulation
            if (step + 1) % ACCUMULATE_GRAD_BATCHES == 0:
                optimizer.step()
                ema.update()
                print("optimizer update!")
            
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        if LR_SCHEDULER:
            scheduler.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Current Learning Rate: {current_lr}")
        
        epoch_loss /= step
        metric = dice_metric.aggregate().item() # aggregate the final mean dice result
        dice_metric.reset()
        history['train_epoch_loss'].append(epoch_loss)
        history['train_metric_values'].append(metric)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_epoch_loss = 0
            val_step = 0
            with torch.no_grad():
                for y, val_data in enumerate(val_loader):
                    val_step += 1
                    val_image, val_seg = (
                        val_data["image"].to(device),
                        val_data["seg"].to(device),
                    )
                    
                    if VAL_SCALE < 1:
                        val_image = F.interpolate(val_image[:,1:2,:,:,:], 
                                                scale_factor = VAL_SCALE, 
                                                align_corners = True, 
                                                mode = 'trilinear', 
                                                recompute_scale_factor = False)
                        val_seg = F.interpolate(val_seg.to(torch.float32), 
                                            scale_factor = VAL_SCALE, 
                                            mode ='nearest', 
                                            recompute_scale_factor = False)
                    else:
                        val_image = val_image[:,1:2,:,:,:]
                        val_seg = val_seg
                    
                    # Sliding window inference
                    if SLIDING_WINDOW:
                        roi_size = (50,50,50)
                        sw_batch_size = 1
                        val_outputs = sliding_window_inference(val_image, roi_size, sw_batch_size, ema)
                    else:
                        val_outputs = ema(val_image)
                    
                    # metric
                    post_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    post_val_seg = [post_label(i) for i in decollate_batch(val_seg)]
                    dice_metric(y_pred=post_val_outputs, y=post_val_seg)
                    
                    # loss
                    val_loss = loss_function(val_outputs, val_seg)
                    val_epoch_loss += val_loss.item()
                    
                    # Sanity check
                    if (epoch + 1) % CHECK_INTERVAL == 0:
                        check_dir = os.path.join(model_dir, 'val_check')
                        if not os.path.exists(check_dir):
                            print("Create 'check' folder for validation output.")
                            os.makedirs(check_dir, exist_ok=True)
                        save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                        Sanity_check.check_seg(outputs = val_outputs, 
                                                data = (val_image, val_seg), 
                                                save_dir = save_dir)
                    
                val_epoch_loss /= val_step
                metric = dice_metric.aggregate().item() # aggregate the final mean dice result
                dice_metric.reset() # reset the status for next validation round
                
                history['val_epoch_loss'].append(val_epoch_loss)
                history['val_metric_values'].append(metric)
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "try_unet.pth"))
                    print("saved new best metric model")
                    
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print("-"*20)
    print("-"*20)
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    print(f"--- {(time.time()-start_time):.4f} seconds ---")
    
    save_dir = os.path.join(model_dir, 'training_log.csv')
    write_csv(history, save_dir)
    
    ###################
    ##   Plot loss   ##
    ###################
    # Plot the loss and metric
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Epoch Average Loss")
    x = [i + 1 for i in range(len(history['train_epoch_loss']))]
    y_tr = history['train_epoch_loss']
    y_val = history['val_epoch_loss']
    axes[0].set_xlabel("Epoch")
    axes[0].plot(x, y_tr, label='train')
    axes[0].plot(x, y_val, label='val')
    axes[0].legend()
    axes[1].set_title("Val Mean Dice")
    x = [VAL_INTERVAL * (i + 1) for i in range(len(history['train_metric_values']))]
    y_tr = history['train_metric_values']
    y_val = history['val_metric_values']
    axes[1].set_xlabel("Epoch")
    axes[1].plot(x, y_tr, label='train')
    axes[1].plot(x, y_val, label='val')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'loss_metric.png'))
    plt.close(fig)

if __name__ == "__main__":
    main()