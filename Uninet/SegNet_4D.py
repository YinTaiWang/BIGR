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
from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from model.unet import UNet
from utils.functions import *
from utils.sanity_check import *
from utils.transforms import *

from ema_pytorch import EMA
from sklearn.model_selection import KFold


################
##  Settings  ##
################
TEST_MODE = False # use only a small part of data
VALIDATE_MODEL = True # perform the validation set

CONFIG = {
    # Model setup
    'max_epochs': 500,
    'fold': 5,
    'lr': 0.01,
    'lr_scheduler': True,
    'filters': [4, 8, 16, 32, 64, 128],
    'kernels': [[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    'strides': [[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2]],
    # Model stablizer
    'accumulate_grad_batches': 4,
    # Weight
    'weight_s': 0.8, # weight for no ground truth ones
    # Sanity checks
    'check_train': False,
    'check_tr_interval': 1,
    'check_val': True,
    'check_val_interval': 100,
}

print("=== CONFIG ===")
max_key_length = max(len(key) for key in CONFIG.keys())
print(f"TASK: Segmentation")
for key, value in CONFIG.items():
    print(f'{key.upper():<{max_key_length + 3}}{value}')

################################## MAIN ##################################
def main():
    SAVE_ONE_EXAMPLE = False
    ####################
    ##      Setup     ##
    ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    set_determinism(seed=4294967295) # use the default seed from monai
    
    script_dir = os.getcwd()
    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False

    if GPU_cluster:
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/images_preprocessed_nobc"
        seg3D_dir = "/data/scratch/r098906/BLT_radiomics/segs_new_preprocessed"
        model_folder = "/data/scratch/r098906/BLT_results/segnet"
    else:
        img4D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/images_preprocessed_nobc"
        seg3D_dir = "c:/Users/Yin/Desktop/Vrije_courses/internship/codes/data/BLT_radiomics/image_files/segs_new_preprocessed"
        model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "segnet")
    
    model_dir = model_folder + "/" + attempt
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nGPU cluster: {GPU_cluster}; Device: {device}")
    print("\n=== Data directory ===")
    print(f"IMAGES: {img4D_dir}")
    print(f"SEGMENTATION: {seg3D_dir}")
    print(f"RESULTS: {model_dir}")
    
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = training_transforms(seed=0)
    val_transforms = training_transforms(validation=True)
    
    #################
    ##  Load data  ##
    #################
    images = sorted(glob.glob(os.path.join(img4D_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg3D_dir, "*.nii.gz")))
    pattern = r"_([0-9]+)_([0-9]+)\.nii\.gz"
    patients, phases = [], []
    for file in sorted(os.listdir(seg3D_dir)):
        match = re.search(pattern, file)
        if match:
            patients.append(match.group(1))
            phases.append(match.group(2))
    
    #test
    if TEST_MODE:
        images = images[:10]
        segs = segs[:10]
        patients = patients[:10]
        phases = phases[:10]
        
    
    data_dicts = [{"image": image, "seg": seg, "patient":patient, "phase":phase} 
                  for image, seg, patient, phase in zip(images, segs, patients, phases)]
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
    kf = KFold(n_splits=CONFIG['fold'], random_state=12345, shuffle=True)
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
        model = DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=8,
            kernel_size=CONFIG['kernels'],
            strides=CONFIG["strides"],
            upsample_kernel_size=CONFIG["strides"][1:],
            filters=CONFIG['filters'],
            norm_name="instance",
            act_name="leakyrelu",
            deep_supervision=False,
            deep_supr_num=2
            ).to(device)
        divisor = 2 ** (len(CONFIG['kernels']))
        
        # exponential moving average
        ema = EMA(
            model,
            beta = 0.9999, #0.9999,   # exponential moving average factor
            update_after_step = 5,    # only after this number of .update() calls will it start updating
            update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
            )
        ema.update()
        # loss function
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        # optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG['lr'],
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        # lr scheduler
        if CONFIG['lr_scheduler']:
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (1 - epoch / CONFIG['max_epochs']) ** 0.9,
            )
        # metric function
        # dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        ###############
        ##   Train   ##
        ###############
        best_metric = -1
        best_metric_epoch = -1
        train_history = {'total_loss': [],
                         'metric_dice': [],}
        val_history = {'total_loss': [],
                       'metric_dice': []}
        
        start_time = time.time()
        for epoch in range(CONFIG['max_epochs']):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{CONFIG['max_epochs']}")
            model.train()
            epoch_loss = 0
            epoch_dice = 0
            step = 0
            epoch_start_time = time.time()

            for x, train_data in enumerate(train_loader):
                step += 1
                image, seg, patient, phase = (
                    train_data["image"].to(device),
                    train_data["seg"].to(device),
                    train_data["patient"],
                    train_data["phase"]
                )

                # CHECK MEM
                # mem = torch.cuda.memory_allocated(device)
                # print(f'After load data {x}: {(mem/2**30):.2f} GB')
                
                # Preprocess:
                image = add_padding_to_divisible(image, divisor)
                seg = add_padding_to_divisible(seg, divisor)
                patient = patient[0]
                phase = int(phase[0])
                
                # Model predict
                outputs = model(image) # [1, 8, *img_shape]
            
                # Loss
                batch_loss = 0
                pair = outputs.shape[1]/2
                for c_phase, c in enumerate(range(0, outputs.shape[1], 2)):
                    phase_loss = loss_function(outputs[:,c:c+2,...], seg)
                    if c_phase != phase:
                        phase_loss = phase_loss * CONFIG['weight_s']
                    batch_loss += phase_loss
                batch_loss /= pair
                epoch_loss += batch_loss.item() # save it to epoch_loss
                
                # CHECK MEM
                # mem = torch.cuda.memory_allocated(device)
                # print(f'After load data {x}: {(mem/2**30):.2f} GB')
                
                ################### TRAIN ONLY ###################
                # Backpropagation
                batch_loss.backward()
                # Update parameters // Gradient accumulation // EMA
                if step % CONFIG['accumulate_grad_batches'] == 0:
                    optimizer.step()
                    ema.update()
                    optimizer.zero_grad() # clear gradients
                ################### TRAIN ONLY ###################
                
                # CHECK MEM
                # mem = torch.cuda.memory_allocated(device)
                # print(f'After load data {x}: {(mem/2**30):.2f} GB')
                
                if SAVE_ONE_EXAMPLE and (batch_loss.item() < 0.5):
                    torch.save(outputs, "outputs.pth")
                    torch.save(seg, "seg.pth")
                    SAVE_ONE_EXAMPLE = False
                    
                # Metric
                metric, method, post_outputs = best_post_processing_finder(outputs, seg)
                epoch_dice += metric
                # print(f"post processing: {method}")
                
                # Sanity check
                if CONFIG['check_train'] and ((epoch + 1) % CONFIG['check_tr_interval']) == 0:
                    # create path to save the file
                    check_dir = os.path.join(model_dir, f'train_check_fold{fold+1}')
                    if not os.path.exists(check_dir):
                        print(f"Create train_check_fold{fold+1} folder for validation output.")
                        os.makedirs(check_dir, exist_ok=True)
                    # execute
                    save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                    Sanity_check.check_seg4d(outputs = post_outputs, 
                                            data = (image, seg, patient, phase), 
                                            save_dir = save_dir)
            
                # print
                if step % 40 == 0: # only print loss every 20 batch    
                    print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {batch_loss.item():.4f}")

            # CHECK MEM
            # mem = torch.cuda.memory_allocated(device)
            # print(f'End of the epoch {x}: {(mem/2**30):.2f} GB')
                
            if CONFIG['lr_scheduler']:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    print(f"Current Learning Rate: {current_lr}")
            
            epoch_loss /= step
            epoch_dice /= step
            train_history['total_loss'].append(epoch_loss)
            train_history['metric_dice'].append(epoch_dice)
            print(f"Training -- epoch {epoch + 1} average loss: {epoch_loss:.4f}, average dice: {epoch_dice:.4f}")
            
            

            if VALIDATE_MODEL:
                model.eval()
                val_epoch_loss = 0
                val_epoch_dice = 0
                val_step = 0
                with torch.no_grad():
                    for y, val_data in enumerate(val_loader):
                        val_step += 1
                        val_image, val_seg, val_patient, val_phase = (
                            val_data["image"].to(device),
                            val_data["seg"].to(device),
                            val_data["patient"],
                            val_data["phase"]
                        )
                        
                        # Data preprocess:
                        val_image = add_padding_to_divisible(val_image, divisor)
                        val_seg = add_padding_to_divisible(val_seg, divisor)
                        val_patient = val_patient[0]
                        val_phase = int(val_phase[0])
                        
                        # Model predict
                        val_outputs = ema(val_image)
                        
                        # Loss
                        val_batch_loss = 0
                        val_pair = val_outputs.shape[1]/2
                        for c_phase, c in enumerate(range(0, val_outputs.shape[1], 2)):
                            val_phase_loss = loss_function(val_outputs[:,c:c+2,...], val_seg)
                            if c_phase != val_phase:
                                val_phase_loss = val_phase_loss * CONFIG['weight_s']
                            val_batch_loss += val_phase_loss
                        val_batch_loss /= val_pair
                        val_epoch_loss += val_batch_loss.item()  # save it to epoch_loss
            
                        # Metric
                        val_metric, val_method, post_val_outputs = best_post_processing_finder(val_outputs, val_seg)
                        val_epoch_dice += val_metric
                        print(f"post processing (val): {val_method}")
                        
                        # Sanity check
                        if CONFIG['check_val'] and ((epoch + 1) % CONFIG['check_val_interval'] == 0):
                            # create path to save the file
                            check_dir = os.path.join(model_dir, f'val_check_fold{fold+1}')
                            if not os.path.exists(check_dir):
                                print(f"Create 'val_check_fold{fold+1}' folder for validation output.")
                                os.makedirs(check_dir, exist_ok=True)
                            # execute
                            save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                            Sanity_check.check_seg4d(outputs = post_val_outputs, 
                                                    data = (val_image, val_seg, val_patient, val_phase), 
                                                    save_dir = save_dir)
                    
                    val_epoch_loss /= val_step # calculate average epoch loss
                    val_epoch_dice /= val_step
                    
                    val_history['total_loss'].append(val_epoch_loss)
                    val_history['metric_dice'].append(val_epoch_dice)
                    
                    if val_epoch_dice > best_metric:
                        best_metric = val_epoch_dice
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(model_dir, f"seg_fold{fold+1}.pth"))
                        print("saved new best metric model")
                        
                    print(
                        f"Validation -- epoch {epoch + 1} average loss: {val_epoch_loss:.4f}, average dice: {val_epoch_dice:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
                    print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")
        
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
    
    mean_metric = np.mean(best_metrics)
    print(" ")
    print("="*20)
    print(f"All training are completed, average metric of all folds: {mean_metric:.4f}")

if __name__ == "__main__":
    main()