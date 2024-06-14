import os
import sys
import glob
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split

import monai
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.utils import set_determinism
from transforms.set_transforms import training_transforms, post_transfroms

from model.douwe_unet import UNet
from model.monai_unet import MOANI_UNet
from model.groupreg_unet import GroupReg_UNet
from ema_pytorch import EMA
from Uninet.loss_function.loss import NCC, Grad
from model.spatial_transformer import SpatialTransformer
from model.utils import Sanity_check, write_csv

####################
##   Settings     ##
####################
TEST_MODE = False # use only a small part of data
MODEL = "Douwe" # Douwe / MONAI / GroupReg
MAX_EPOCHS = 100
LR = 0.001
LR_SCHEDULER = True
KERNEL_SIZE = (3, 3, 3)
SMOOTH_W = 1e-1

TRAIN_SCALE = 0.5
VAL_SCALE = 0.5
VAL_INTERVAL = 1 # always set to 1 to avoid some issue in plotting
CHECK_INTERVAL = 10

print("##############")
print("## Settings ##")
print("##############")
print(f"network: {MODEL}; kernel_size: {KERNEL_SIZE}")
print(f"scale in training set: {TRAIN_SCALE}; scale in validation set: {VAL_SCALE}")
print(f"learning rate: {LR}; learning rate scheduler: {LR_SCHEDULER}")
print(f"training epochs: {MAX_EPOCHS}; Weight of smooth loss: {SMOOTH_W}\n")

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
        model_folder = os.path.join(script_dir, "..", "BLT", "data", "BLT_radiomics", "regnet")
    else:
        img4D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/4D_resampled_images_newseg"
        seg3D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_resampled"
        model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "regnet")
    
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
            in_channels=4,      # n
            out_channels=12,    # n*dim
            strides=(1,)*(len(KERNEL_SIZE)),
            kernel_size = KERNEL_SIZE,
            upsample_kernel_size = (1,)*(len(KERNEL_SIZE)),
            activation = 'LRELU',
            normalisation = 'instance'
            ).to(device)
    elif MODEL == "MONAI":
        model = MOANI_UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=12,
            channels=(32, 64, 128),
            strides=(1,1,1),
            num_res_units=0,
            norm='instance'
            ).to(device)
    elif MODEL == "GroupReg":
        model = GroupReg_UNet(
            in_channels = 4, 
            out_channels = 12, 
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
    
    ncc = NCC()
    grad = Grad(penalty='l2')
    optimizer = torch.optim.Adam(model.parameters(), LR)
    if LR_SCHEDULER:
        lambda_poly = lambda epoch: (1 - epoch / MAX_EPOCHS) ** 0.9
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_poly)
    

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
        epoch_simi_score = 0
        step = 0
        epoch_start_time = time.time()
        for x, train_data in enumerate(train_loader):
            step += 1
            
            image, seg = (
                train_data["image"].to(device),
                train_data["seg"].to(device),
            )
            
            if TRAIN_SCALE < 1:
                ori_image = image
                ori_image_shape = image.shape[2:]
                
                image = F.interpolate(image, 
                                        scale_factor = TRAIN_SCALE, 
                                        align_corners = True, 
                                        mode = 'trilinear', 
                                        recompute_scale_factor = False)
                image_shape = image.shape[2:]
                ori_seg = seg
                seg = F.interpolate(seg.to(torch.float32), 
                                        scale_factor = TRAIN_SCALE, 
                                        mode ='nearest', 
                                        recompute_scale_factor = False)
                
            else:
                image = image
                image_shape = image.shape[2:]
                seg = seg
                
                
            optimizer.zero_grad()
            outputs = model(image)
            
            # spatial transformer
            if TRAIN_SCALE < 1:
                scaled_disp_t2i = torch.squeeze(outputs, 0).reshape(4, 3, *image_shape) # (n, dim, shape)
                disp_t2i = torch.nn.functional.interpolate(scaled_disp_t2i, 
                                                            size = ori_image_shape, 
                                                            mode = 'trilinear', 
                                                            align_corners = True)
            else:
                disp_t2i = torch.squeeze(outputs, 0).reshape(4, 3, *image_shape)
            
            # spatial_transformer = SpatialTransformer(ori_image_shape).to(device)
            spatial_transformer = SpatialTransformer(dim=3).to(device)
            warped_input_image = spatial_transformer(ori_image, disp_t2i)
            template = torch.mean(warped_input_image, dim=1, keepdim=True)
            simi_loss = ncc(warped_input_image, template)
            smooth_loss = grad(disp_t2i)
            
            # plot to check the output
            if (epoch + 1) % CHECK_INTERVAL == 0:
                check_dir = os.path.join(model_dir, 'train_check')
                if not os.path.exists(check_dir):
                    print("Create 'train_check' folder for validation output.")
                    os.makedirs(check_dir, exist_ok=True)
                save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                Sanity_check.check_reg(outputs = warped_input_image, 
                                        data = (ori_image, ori_seg), 
                                        save_dir = save_dir)
                        
            # metric
            simi_score = -(simi_loss.item())
            epoch_simi_score += simi_score
            
            # loss
            total_loss = 0.
            total_loss += simi_loss
            total_loss += smooth_loss*SMOOTH_W
            
            total_loss.backward()
            optimizer.step()
            ema.update()
            epoch_loss += total_loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {total_loss.item():.4f}")
            print(f"simi. loss {simi_loss.item():.4f}, smooth loss {smooth_loss.item():.4f}")
        if LR_SCHEDULER:
            scheduler.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Current Learning Rate: {current_lr}")
        
        epoch_loss /= step
        metric = epoch_simi_score / step
        history['train_epoch_loss'].append(epoch_loss)
        history['train_metric_values'].append(metric)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")
        
        if (epoch + 1) % VAL_INTERVAL == 0:
                model.eval()
                val_epoch_simi_score = 0
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
                            ori_val_image = val_image
                            ori_val_image_shape = val_image.shape[2:]
                            val_image = F.interpolate(val_image, 
                                                    scale_factor = VAL_SCALE, 
                                                    align_corners = True, 
                                                    mode = 'trilinear', 
                                                    recompute_scale_factor = False)
                            val_image_shape = val_image.shape[2:]
                            
                            ori_val_seg = val_seg
                            val_seg = F.interpolate(val_seg.to(torch.float32), 
                                                    scale_factor = VAL_SCALE, 
                                                    mode ='nearest', 
                                                    recompute_scale_factor = False)
                        else:
                            val_image = val_image
                            val_image_shape = val_image.shape[2:]
                            val_seg = val_seg
                        
                        val_outputs = ema(val_image)
                        
                        # spatial transformer
                        if VAL_SCALE < 1:
                            val_scaled_disp_t2i = torch.squeeze(val_outputs, 0).reshape(4, 3, *val_image_shape) # (n, dim, shape)
                            val_disp_t2i = torch.nn.functional.interpolate(val_scaled_disp_t2i, 
                                                                        size = ori_val_image_shape, 
                                                                        mode = 'trilinear', 
                                                                        align_corners = True)
                        else:
                            val_disp_t2i = torch.squeeze(val_outputs, 0).reshape(4, 3, *val_image_shape)
                        # spatial_transformer = SpatialTransformer(ori_val_image_shape).to(device)
                        spatial_transformer = SpatialTransformer(dim=3).to(device)
                        val_warped_input_image = spatial_transformer(ori_val_image, val_disp_t2i)
                        val_template = torch.mean(val_warped_input_image, dim=1, keepdim=True)
                        val_simi_loss = ncc(val_warped_input_image, val_template)
                        val_smooth_loss = grad(val_disp_t2i)
                        
                        # loss
                        val_total_loss = 0.
                        val_total_loss += val_simi_loss
                        val_total_loss += val_smooth_loss*SMOOTH_W
                        val_epoch_loss += val_total_loss.item()
                        
                        if (epoch + 1) % CHECK_INTERVAL == 0:
                            # plot to check the output
                            check_dir = os.path.join(model_dir, 'val_check')
                            if not os.path.exists(check_dir):
                                print("Create 'check' folder for validation output.")
                                os.makedirs(check_dir, exist_ok=True)
                            save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                            Sanity_check.check_reg(outputs = val_warped_input_image, 
                                                data = (ori_val_image, ori_val_seg), 
                                                save_dir = save_dir)
                        
                        # metric
                        val_simi_score = -(val_simi_loss.item())
                        val_epoch_simi_score += val_simi_score
                        
                    val_epoch_loss /= val_step
                    metric = val_epoch_simi_score / val_step
                    
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
    axes[1].set_title("Val Mean NCC")
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