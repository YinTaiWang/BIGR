import os
import re
import glob
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import monai
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from model.unet import UNet
from utils.functions import *
from utils.utils import Sanity_check, write_csv, plot_cv
from utils.set_transforms import training_transforms, post_transfroms

from ema_pytorch import EMA
from sklearn.model_selection import KFold


################
##  Settings  ##
################
TEST_MODE = False # use only a small part of data
VALIDATE_MODEL = True # perform the validation set

CONFIG = {
    # Model setup
    'max_epochs': 2,
    'fold': 2,
    'lr': 0.01,
    'lr_scheduler': True,
    'channels': (32, 64, 128),
    # Scale factor for downscaling the input images and segmentations for training.
    'train_scale': 0.5,  
    'val_scale': 0.5,
    # Model stablizer
    'accumulate_grad_batches': 4,
    # weight
    'weight_s': 0.8,
    # Sanity checks
    'check_train': True,
    'check_tr_interval': 1,
    'check_val': True,
    'check_val_interval': 1,
}

print("=== CONFIG ===")
max_key_length = max(len(key) for key in CONFIG.keys())
print(f"TASK: Segmentation")
for key, value in CONFIG.items():
    print(f'{key.upper():<{max_key_length + 3}}{value}')

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
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/images_preprocessed"
        seg3D_dir = "/data/scratch/r098906/BLT_radiomics/segs_new_preprocessed"
        model_folder = os.path.join(script_dir, "..", "BLT", "data", "BLT_radiomics", "segnet")
    else:
        img4D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/images_preprocessed"
        seg3D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_preprocessed"
        model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "segnet")
    
    model_dir = os.path.join(model_folder, attempt)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"GPU cluster: {GPU_cluster}; Device: {device}")
    print(f"The results will be saved at: {model_dir}")
    
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = training_transforms(seed=0)
    val_transforms = training_transforms(validation=True)
    post_pred = post_transfroms()
    post_label = post_transfroms(label=True)
    
    #################
    ##  Load data  ##
    #################
    images = sorted(glob.glob(os.path.join(img4D_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg3D_dir, "*.nii.gz")))
    pattern = r"_(\d+)\.nii\.gz"
    phases = [re.search(pattern, file).group(1) for file in sorted(os.listdir(seg3D_dir)) if re.search(pattern, file)]
    
    #test
    if TEST_MODE:
        images = images[:10]
        segs = segs[:10]
        phases = phases[:10]
    
    data_dicts = [{"image": image, "seg": seg, "phase":phase} for image, seg, phase in zip(images, segs, phases)]
    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\n")

    ##########################
    ##   Cross-Validation   ##
    ##########################
    print("##############")
    print("## Training ##")
    print("##############")
    CV_train_history = list()
    CV_val_history = list()
    best_metrics = list()
    
    indices = np.arange(len(images))
    kf = KFold(n_splits=CONFIG['fold'], random_state=1, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold+1}:")
        print(f"  Train: index={train_idx}")
        print(f"  Test:  index={val_idx}\n")
        train_files, val_files = [data_dicts[i] for i in train_idx], [data_dicts[i] for i in val_idx]
        
        ####################
        ##   Dataloader   ##
        ####################
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

        ###############
        ##   Model   ##
        ###############
        model = UNet(
            spatial_dims = 3,
            in_channels = 4,
            out_channels = 8,
            channels = CONFIG['channels'],
            act = "LRELU",
            norm = "instance"
        ).to(device)
        divisor = 2 ** (len(CONFIG['channels'])-1)
        
        ema = EMA(
            model,
            beta = 0.9999, #0.9999,   # exponential moving average factor
            update_after_step = 5,    # only after this number of .update() calls will it start updating
            update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
            )
        ema.update()
        
        # loss function
        loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])
        if CONFIG['lr_scheduler']:
            lambda_poly = lambda epoch: (1 - epoch / CONFIG['max_epochs']) ** 0.9
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_poly)
        # metric
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        

        ###############
        ##   Train   ##
        ###############
        best_metric = -1
        best_metric_epoch = -1
        train_history = {'total_loss': [],
                        'metric_dice': [],}
        val_history  = {'total_loss': [],
                    'metric_dice': []}
        
        start_time = time.time()
        for epoch in range(CONFIG['max_epochs']):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{CONFIG['max_epochs']}")
            model.train()
            epoch_loss = 0
            step = 0
            epoch_start_time = time.time()
            
            for x, train_data in enumerate(train_loader):
                step += 1
                image, seg, phase = (
                    train_data["image"].to(device),
                    train_data["seg"].to(device),
                    train_data["phase"]
                )
                
                # Data preprocess:
                # padding to fit the unet strided conv
                # scaling to fit the memory limit
                padded_image = add_padding_to_divisible(image, divisor)
                padded_seg = add_padding_to_divisible(seg, divisor)
                phase = int(phase[0])
                
                # Model predict
                outputs = model(padded_image) # [1, 8, *img_shape]
            
                # Loss
                batch_loss = 0
                pair = outputs.shape[1]/2
                for c_phase, c in enumerate(range(0, outputs.shape[1], 2)):
                    phase_loss = loss_function(outputs[:,c:c+2,...], padded_seg)
                    if c_phase != phase:
                        phase_loss = phase_loss * CONFIG['weight_s']
                    batch_loss += phase_loss
                batch_loss /= pair
                epoch_loss += batch_loss.item() # save it to epoch_loss
                
                # Backpropagation
                batch_loss.backward()
                
                # Update parameters // Gradient accumulation // EMA
                if step % CONFIG['accumulate_grad_batches'] == 0:
                    optimizer.step()
                    ema.update()
                    optimizer.zero_grad() # clear gradients
                
                # Metric
                post_tr_outputs = [
                    post_pred(i)
                    for x in range(0, outputs.shape[1], 2)
                    for i in decollate_batch(outputs[:, x:x+2, :, :, :])
                ]
                post_tr_seg = [post_label(i) for i in decollate_batch(padded_seg)] * 4
                dice_metric(y_pred=post_tr_outputs, y=post_tr_seg)
                
                # Sanity check
                if CONFIG['check_train'] and ((epoch + 1) % CONFIG['check_tr_interval']) == 0:
                    # create path to save the file
                    check_dir = os.path.join(model_dir, f'train_check_fold{fold+1}')
                    if not os.path.exists(check_dir):
                        print(f"Create train_check_fold{fold+1} folder for validation output.")
                        os.makedirs(check_dir, exist_ok=True)
                    # execute
                    save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                    Sanity_check.check_seg4d(outputs = outputs, 
                                            data = (padded_image, padded_seg), 
                                            save_dir = save_dir)
                    
                # print
                if step % 20 == 0: # only print loss every 20 batch    
                    print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {batch_loss.item():.4f}")
            
            if CONFIG['lr_scheduler']:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    print(f"Current Learning Rate: {current_lr}")
            
            epoch_loss /= step
            metric = dice_metric.aggregate().item() # aggregate the final mean dice result
            dice_metric.reset()
            
            train_history['total_loss'].append(epoch_loss)
            train_history['metric_dice'].append(metric)
            print(f"Training -- epoch {epoch + 1} average loss: {epoch_loss:.4f}, average dice: {metric:.4f}")
            

            if VALIDATE_MODEL:
                model.eval()
                val_epoch_loss = 0
                val_step = 0
                with torch.no_grad():
                    for y, val_data in enumerate(val_loader):
                        val_step += 1
                        val_image, val_seg, val_phase = (
                            val_data["image"].to(device),
                            val_data["seg"].to(device),
                            val_data["phase"]
                        )
                        
                        # Data preprocess:
                        # padding to fit the unet strided conv
                        # scaling to fit the memory limit
                        padded_val_image = add_padding_to_divisible(val_image, divisor)
                        padded_val_seg = add_padding_to_divisible(val_seg, divisor)
                        val_phase = int(val_phase[0])
                        
                        # Model predict
                        val_outputs = ema(padded_val_image)
                        
                        # Loss
                        val_batch_loss = 0
                        val_pair = val_outputs.shape[1]/2
                        for c_phase, c in enumerate(range(0, val_outputs.shape[1], 2)):
                            val_phase_loss = loss_function(val_outputs[:,c:c+2,...], padded_val_seg)
                            if c_phase != val_phase:
                                val_phase_loss = val_phase_loss * CONFIG['weight_s']
                            val_batch_loss += val_phase_loss
                        val_batch_loss /= val_pair
                        val_epoch_loss += val_batch_loss.item()  # save it to epoch_loss
                        
                        # Metric
                        post_val_outputs = [
                            post_pred(i)
                            for x in range(0, val_outputs.shape[1], 2)
                            for i in decollate_batch(val_outputs[:, x:x+2, :, :, :])
                        ]
                        post_val_seg = [post_label(i) for i in decollate_batch(padded_val_seg)] * 4
                        dice_metric(y_pred=post_val_outputs, y=post_val_seg)
                        
                        # Sanity check
                        if CONFIG['check_val'] and ((epoch + 1) % CONFIG['check_val_interval'] == 0):
                            # create path to save the file
                            check_dir = os.path.join(model_dir, f'val_check_fold{fold+1}')
                            if not os.path.exists(check_dir):
                                print(f"Create 'val_check_fold{fold+1}' folder for validation output.")
                                os.makedirs(check_dir, exist_ok=True)
                            # execute
                            save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                            Sanity_check.check_seg4d(outputs = val_outputs, 
                                                    data = (padded_val_image, padded_val_seg), 
                                                    save_dir = save_dir)
                    
                    val_epoch_loss /= val_step # calculate average epoch loss
                    metric = dice_metric.aggregate().item() # aggregate the final mean dice result
                    dice_metric.reset() # reset the status for next validation round
                    
                    val_history['total_loss'].append(val_epoch_loss)
                    val_history['metric_dice'].append(metric)
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(model_dir, f"seg_fold{fold+1}.pth"))
                        print("saved new best metric model")
                        
                    print(
                        f"Validation -- epoch {epoch + 1} average loss: {val_epoch_loss:.4f}, average dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
                    print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")
            if step == len(train_ds):
                torch.save(model.state_dict(), os.path.join(model_dir, f"seg_fold{fold+1}_final.pth"))
                print("saved the final model")
        
        print("+"*30)
        print(f"Fold {fold+1} train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
        print(f"--- {(time.time()-start_time):.4f} seconds ---")
        best_metrics.append(best_metric)
        
        save_dir = os.path.join(model_dir, f"training_log_fold{fold+1}.csv")
        write_csv(train_history, save_dir)
        save_dir = os.path.join(model_dir, f"validation_log_fold{fold+1}.csv")
        write_csv(val_history, save_dir)
        print(" ")
        CV_train_history.append(train_history)
        CV_val_history.append(val_history)
        # plot
        save_dir = os.path.join(model_dir, f"loss_metric.png")
        plot_cv(CV_train_history, CV_val_history, model_dir)
    
    best_fold_metric = max(best_metrics)
    best_fold_metric_idx = best_metrics.index(best_fold_metric) + 1
    print(" ")
    print("="*20)
    print(f"All training are completed, best_metric: {best_fold_metric:.4f} " f"at fold: {best_fold_metric_idx}")

if __name__ == "__main__":
    main()