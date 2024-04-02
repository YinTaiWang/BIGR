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
from Uninet.transforms.set_transforms import training_transforms, post_transfroms

# from monai.networks.nets import UNet
from model.douweunet import UNet
from model.loss import NCC, Grad
from model.spatial_transformer import SpatialTransformer
import model.util, utils.structure
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA

####################
##   Settings     ##
####################
TRAIN_SCALE = 0.5
VAL_SCALE = 0.5
LR = 0.01
LR_SCHEDULER = True
MAX_EPOCHS = 200
KERNEL_SIZE = (3, 3, 3)
SMOOTH_W = 1e-3
val_interval = 1
check_interval = 10


print("##############")
print("## Settings ##")
print("##############")
print(f"network: Douwe's improved UNet, kernel_size: {KERNEL_SIZE}")
print(f"scale in training set: {TRAIN_SCALE}; scale in validation set: {VAL_SCALE}")
print(f"learning rate: {LR}; learning rate scheduler: {LR_SCHEDULER}")
print(f"training epochs: {MAX_EPOCHS}\n")

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

def find_first_and_last_slice_with_label(data, label=1):
    first_slice = None
    last_slice = None
    slices = data.shape[-1]
    
    for i in range(slices):
        if label in data[..., i]:
            if first_slice is None:
                first_slice = i  # Found the first slice with the label
            last_slice = i  # Update last slice with the label at each find
    
    if first_slice is not None:
        return [first_slice, last_slice]
    else:
        return None  # Return None if the label is not found in any slice


def check_output(outputs, data, save_dir):
    image = data[0].cpu()
    seg = data[1].cpu()
    
    first_and_last_slice = find_first_and_last_slice_with_label(seg)
    middle_slice = round(sum(first_and_last_slice) / len(first_and_last_slice))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_title(f"Image slice {middle_slice}")
    axes[0].imshow(image[0, 0, :, :, middle_slice], cmap="gray")
    axes[1].set_title(f"Mask slice {middle_slice}")
    axes[1].imshow(seg[0, 0, :, :, middle_slice])
    axes[2].set_title(f"Output slice {middle_slice}")
    axes[2].imshow(torch.argmax(outputs, dim=1).detach().cpu()[0, :, :, middle_slice])
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close(fig)

################################## MAIN ##################################
def main():
    
    ####################
    ##      Setup     ##
    ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%H%M%S")
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
    spatial_transformer = SpatialTransformer()
    

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
        for batch_data in train_loader:
            step += 1
            epoch_ncc = 0
            
            image, seg = (
                batch_data["image"].to(device),
                batch_data["seg"].to(device),
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
            warped_input_image = spatial_transformer(ori_image, disp_t2i)
            template = torch.mean(warped_input_image, dim=1, keepdim=True)
            simi_loss = ncc(warped_input_image, template)
            smooth_loss = grad(disp_t2i)
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
        metric = epoch_simi_score / len(train_loader)
        history['train_epoch_loss'].append(epoch_loss)
        history['train_metric_values'].append(metric)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")
        
        if (epoch + 1) % val_interval == 0:
                model.eval()
                val_epoch_simi_score = 0
                val_epoch_loss = 0
                val_step = 0
                with torch.no_grad():
                    for i, val_data in enumerate(val_loader):
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
                            val_image_shape = val_image.shape
                            val_seg = F.interpolate(val_seg.to(torch.float32), 
                                                    scale_factor = VAL_SCALE, 
                                                    mode ='nearest', 
                                                    recompute_scale_factor = False)
                        else:
                            val_image = val_image
                            val_image_shape = val_image.shape
                            val_seg = val_seg
                        
                        val_outputs = ema(val_image)
                        
                        # spatial transformer
                        if TRAIN_SCALE < 1:
                            val_scaled_disp_t2i = torch.squeeze(val_outputs, 0).reshape(4, 3, *val_image_shape) # (n, dim, shape)
                            val_disp_t2i = torch.nn.functional.interpolate(val_scaled_disp_t2i, 
                                                                        size = ori_val_image_shape, 
                                                                        mode = 'trilinear', 
                                                                        align_corners = True)
                        else:
                            val_disp_t2i = torch.squeeze(val_outputs, 0).reshape(4, 3, *val_image_shape)
                        val_warped_input_image = spatial_transformer(ori_val_image, val_disp_t2i)
                        val_template = torch.mean(val_warped_input_image, dim=1, keepdim=True)
                        val_simi_loss = ncc(val_warped_input_image, val_template)
                        val_smooth_loss = grad(val_disp_t2i)
                        
                        # loss
                        val_total_loss = 0.
                        val_total_loss += val_simi_loss
                        val_total_loss += val_smooth_loss*SMOOTH_W
                        val_epoch_loss += val_total_loss.item()
                        
                        # if (epoch + 1) % check_interval == 0:
                        #     # plot to check the output
                        #     check_dir = os.path.join(model_dir, 'check')
                        #     if not os.path.exists(check_dir):
                        #         print("Create 'check' folder for validation output.")
                        #         os.makedirs(check_dir, exist_ok=True)
                        #     save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{i+1}.png")
                        #     check_output(outputs = val_outputs, 
                        #                 data = (val_image, val_seg), 
                        #                 save_dir = save_dir)
                        
                        # metric
                        val_simi_score = -(val_simi_loss.item())
                        val_epoch_simi_score += val_simi_score

                    val_epoch_loss /= val_step
                    val_metric = val_epoch_simi_score / val_step
                    
                    history['val_epoch_loss'].append(val_epoch_loss)
                    history['val_metric_values'].append(val_metric)
                    
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
    x = [val_interval * (i + 1) for i in range(len(history['train_metric_values']))]
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