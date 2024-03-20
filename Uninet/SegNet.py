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
from monai.transforms import (
    Compose, EnsureChannelFirstd, CropForegroundd, LoadImaged,
    NormalizeIntensityd, Orientationd, AsDiscrete,
    # augmentaion
    RandRotated, RandZoomd, RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandAdjustContrastd, RandFlipd, ToTensord)

# from monai.networks.nets import UNet
from model.douweunet import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import StepLR
from ema_pytorch import EMA

####################
##   Settings     ##
####################
train_scale = 0.5
val_scale = 0.5
lr = 1e-4
lr_scheduler = False
max_epochs = 100
val_interval = 1
# num_res_units = 0

print("##############")
print("## Settings ##")
print("##############")
print(f"scale in training set: {train_scale}; scale in validation set: {val_scale}")
# print(f"num_res_units: {num_res_units}")
print(f"learning rate: {lr}; learning rate scheduler: {lr_scheduler}")
print(f"training epochs: {max_epochs}\n")

###################
##   Functions   ##
###################
def write_csv(dictionary, save_dir):
    '''
    Args:
        dictionary: a dictionary containing the loss and metric values
        save_dir: directory to save the CSV file
    '''
    with open(save_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        # Find the maximum length of the data to set the number of epochs
        max_length = max(len(v) for v in dictionary.values())
        # Write the header
        writer.writerow(['Epoch'] + list(dictionary.keys()))

        # Write data for each epoch
        for i in range(max_length):
            row = [i + 1]  # Epoch number
            for key in dictionary.keys():
                try:
                    # Try to add the value for this epoch, if it exists
                    row.append(dictionary[key][i])
                except IndexError:
                    # If the value doesn't exist for this metric at this epoch, add a blank
                    row.append('')
            writer.writerow(row)

    print(f"{save_dir} created")

def find_first_slice_with_label(data, label=1):
    slices = data.shape[-1]
    for i in range(slices):
        if label in data[..., i]:
            return i
    return None
def check_output(outputs, data, save_dir):
    image = data[0].cpu()
    seg = data[1].cpu()
    
    first_slice_index = find_first_slice_with_label(seg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_title(f"Image slice {first_slice_index}")
    axes[0].imshow(image[0, 0, :, :, first_slice_index], cmap="gray")
    axes[1].set_title(f"Mask slice {first_slice_index}")
    axes[1].imshow(seg[0, 0, :, :, first_slice_index])
    axes[2].set_title(f"Output slice {first_slice_index}")
    axes[2].imshow(torch.argmax(outputs, dim=1).detach().cpu()[0, :, :, first_slice_index])
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close(fig)

################################## MAIN ##################################
def main():
    
    ####################
    ##      Setup     ##
    ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%H:%M")
    set_determinism(seed=0)

    script_dir = os.getcwd()
    model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "u-net")
    model_dir = os.path.join(model_folder, attempt)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"The results will be saved at: {model_dir}")

    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False

    if GPU_cluster:
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_resampled_images_newseg"
        seg3D_dir = "/data/scratch/r098906/BLT_radiomics/segs_new_resampled"
    else:
        img4D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/4D_resampled_images_newseg"
        seg3D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_resampled"

    ####################
    ### Load data
    # images and segmentations
    images = sorted(glob.glob(os.path.join(img4D_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg3D_dir, "*.nii.gz")))
    
    #test
    # images = images[:3]
    # segs = segs[:3]
    
    data_dicts = [{"image": image, "seg": seg} for image, seg in zip(images, segs)]

    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\n\n")

    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=1)
    train_files, val_files = [data_dicts[i] for i in train_idx], [data_dicts[i] for i in val_idx]
        
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            CropForegroundd(keys=["image", "seg"], source_key="image"),
            Orientationd(keys=["image", "seg"], axcodes="RAS"),
            
            # Data augmentation
            RandRotated(
                keys=["image", "seg"],
                range_x=180,
                range_y=180,
                mode=("bilinear", "nearest"),
                align_corners=(True, None),
                prob=0.2,
            ),
            RandZoomd(
                keys=["image", "seg"],
                min_zoom=0.7,
                max_zoom=1.4,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.2,
            ),
            RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.2,
            ),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
            RandAdjustContrastd(keys=["image"], gamma=(0.65, 1.5), prob=0.15),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[0], prob=0.5
            ),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[1], prob=0.5
            ),
            RandFlipd(
                keys=["image", "seg"], spatial_axis=[2], prob=0.5
            ),
            ToTensord(keys=["image", "seg"])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "seg"]),
            EnsureChannelFirstd(keys=["image", "seg"]),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            CropForegroundd(keys=["image", "seg"], source_key="image"),
            Orientationd(keys=["image", "seg"], axcodes="RAS"),
        ]
    )
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
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
    model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    strides=(1, 1, 1),
    kernel_size = (3, 3, 3),
    upsample_kernel_size = (1, 1, 1),
    activation = 'LRELU',
    normalisation = 'instance'
    ).to(device)
    
    ema = EMA(
        model,
        beta = 0.9999, #0.9999,   # exponential moving average factor
        update_after_step = 5,    # only after this number of .update() calls will it start updating
        update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    ema.update()
    
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    if lr_scheduler:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
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
        'metric_values': []
    }
    
    start_time = time.time()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        epoch_start_time = time.time()
        for batch_data in train_loader:
            step += 1
            image, seg = (
                batch_data["image"].to(device),
                batch_data["seg"].to(device),
            )
            
            if train_scale < 1:
                image = F.interpolate(image[:,1:2,:,:,:], 
                                         scale_factor = train_scale, 
                                         align_corners = True, 
                                         mode = 'trilinear', 
                                         recompute_scale_factor = False)
                seg = F.interpolate(seg.to(torch.float32), 
                                       scale_factor = train_scale, 
                                       mode ='nearest', 
                                       recompute_scale_factor = False)
            else:
                image = image[:,1:2,:,:,:]
                seg = seg
                
            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_function(outputs, seg)
            loss.backward()
            optimizer.step()
            ema.update()
            if lr_scheduler:
                scheduler.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        history['train_epoch_loss'].append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            val_step = 0
            with torch.no_grad():
                for i, val_data in enumerate(val_loader):
                    val_step += 1
                    val_image, val_seg = (
                        val_data["image"].to(device),
                        val_data["seg"].to(device),
                    )
                    
                    if val_scale < 1:
                        val_image = F.interpolate(val_image[:,1:2,:,:,:], 
                                                scale_factor = val_scale, 
                                                align_corners = True, 
                                                mode = 'trilinear', 
                                                recompute_scale_factor = False)
                        val_seg = F.interpolate(val_seg.to(torch.float32), 
                                            scale_factor = val_scale, 
                                            mode ='nearest', 
                                            recompute_scale_factor = False)
                    else:
                        val_image = val_image[:,1:2,:,:,:]
                        val_seg = val_seg
                    roi_size = (50,50,50)
                    sw_batch_size = 1
                    # val_outputs = sliding_window_inference(val_image, roi_size, sw_batch_size, model)
                    val_outputs = sliding_window_inference(val_image, roi_size, sw_batch_size, ema)
                    
                    # save the loss
                    val_loss = loss_function(val_outputs, val_seg)
                    val_epoch_loss += val_loss.item()
                    
                    # plot to check the output
                    check_dir = os.path.join(model_dir, 'check')
                    if not os.path.exists(check_dir):
                        print("Create 'check' folder for validation output.")
                        os.makedirs(check_dir, exist_ok=True)
                    save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{i+1}.png")
                    check_output(outputs = val_outputs, 
                                 data = (val_image, val_seg), 
                                 save_dir = save_dir)
                    
                    # check dice metric
                    post_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    post_val_seg = [post_label(i) for i in decollate_batch(val_seg)]
                    dice_metric(y_pred=post_val_outputs, y=post_val_seg)

                val_epoch_loss /= val_step
                metric = dice_metric.aggregate().item() # aggregate the final mean dice result
                dice_metric.reset() # reset the status for next validation round
                
                history['val_epoch_loss'].append(val_epoch_loss)
                history['metric_values'].append(metric)
                
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
    
    ##################
    ##   Plotting   ##
    ##################
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
    x = [val_interval * (i + 1) for i in range(len(history['metric_values']))]
    y = history['metric_values']
    axes[1].set_xlabel("Epoch")
    axes[1].plot(x, y)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'loss_metric.png'))
    plt.close(fig)

if __name__ == "__main__":
    main()