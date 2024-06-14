import os
import re
import glob
import time
import numpy as np
from datetime import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from sklearn.model_selection import KFold

from monai.networks.nets import DynUNet
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric

from loss_function import DiceCELoss
from utils.sanity_check import Sanity_check
from utils.functions import *
from utils.file_and_folder_operations import load_json
from utils.transforms import training_transforms, best_post_processing_finder
from utils.results_handler import write_csv, plot_one_fold_results, plot_cross_validation_results
from utils.tasks_by_id import set_results_path_by_id


################
##  Settings  ##
################
TEST_MODE = False # use only a small part of data
VALIDATE_MODEL = True # perform the validation set

################################## MAIN ##################################
def main(args):
    
    ####################
    ##      Setup     ##
    ####################
    SAVE_ONE_EXAMPLE = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    set_determinism(seed=4294967295) # use the default seed from monai
    
    root_path = args.root_path
    task_id = str(args.task_id)
    PreprocessTask_dir, ResultsTask_dir = set_results_path_by_id(root_path, task_id)
    img_dir = os.path.join(PreprocessTask_dir, "imagesTr")
    seg_dir = os.path.join(PreprocessTask_dir, "labelsTr")
    
    model_dir = ResultsTask_dir + "/" + attempt
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        
    plans = load_json(os.path.join(PreprocessTask_dir, "plans.json"))
    CONFIG = plans['model_configs']
    print("=== CONFIG ===")
    max_key_length = max(len(key) for key in CONFIG.keys())
    print(f"TASK: Segmentation")
    for key, value in CONFIG.items():
        print(f'{key.upper():<{max_key_length + 3}}{value}')
    
    print(f"\nDevice: {device}")
    print("\n=== Data directory ===")
    print(f"IMAGES: {img_dir}")
    print(f"SEGMENTATION: {seg_dir}")
    print(f"RESULTS: {model_dir}")
    
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = training_transforms(seed=4294967295)
    val_transforms = training_transforms(validation=True)
    
    #################
    ##  Load data  ##
    #################
    images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
    pattern = r"_([0-9]+)_([0-9]+)\.nii\.gz"
    patients, phases = [], []
    for file in sorted(os.listdir(seg_dir)):
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
        train_files, val_files = [data_dicts[i] for i in train_idx], [data_dicts[i] for i in val_idx]
        print(f"\nFold {fold+1}:")
        print(f"  Train: {[data['patient'] for data in train_files]}")
        print(f"  Test:  {[data['patient'] for data in val_files]}\n")
        ####################
        ##   Dataloader   ##
        ####################
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], num_workers=2)

        ###############
        ##   Model   ##
        ###############
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=CONFIG['kernels'],
            strides=CONFIG["strides"],
            upsample_kernel_size=CONFIG["strides"][1:],
            filters=CONFIG['filters'],
            norm_name="instance",
            act_name="leakyrelu",
            deep_supervision=True,
            deep_supr_num=CONFIG['deep_supervision']
            ).to(device)
        divisors = calculate_reduction_factors(CONFIG["strides"])
        
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
            nesterov=True)
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
                phase = int(phase[0])
                image = add_padding_to_divisible(image, divisors)
                seg = add_padding_to_divisible(seg, divisors)
                patient = patient[0]
                
                # Model predict
                outputs = model(image) # [1, 8, *img_shape]
            
                # Loss
                # loss = loss_function(outputs, seg)
                loss = _compute_loss(outputs, seg, loss_function, CONFIG['deep_supervision_weights'])
                epoch_loss += loss.item() # save it to epoch_loss
                
                # CHECK MEM
                # mem = torch.cuda.memory_allocated(device)
                # print(f'After load data {x}: {(mem/2**30):.2f} GB')
                
                ################### TRAIN ONLY ###################
                # Backpropagation
                loss.backward()
                # Update parameters // Gradient accumulation // EMA
                if step % CONFIG['accumulate_grad_batches'] == 0:
                    optimizer.step()
                    ema.update()
                    optimizer.zero_grad() # clear gradients
                ################### TRAIN ONLY ###################
                
                # CHECK MEM
                # mem = torch.cuda.memory_allocated(device)
                # print(f'After load data {x}: {(mem/2**30):.2f} GB')
                
                if SAVE_ONE_EXAMPLE and (loss.item() < 0.5):
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
                    Sanity_check.check_seg3d(outputs = post_outputs, 
                                            data = (image, seg, patient, phase), 
                                            save_dir = save_dir)
                
                # print
                if step % 40 == 0: # only print loss every 20 batch    
                    print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

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
                        val_image = add_padding_to_divisible(val_image, divisors)
                        val_seg = add_padding_to_divisible(val_seg, divisors)
                        val_patient = val_patient[0]
                        val_phase = int(val_phase[0])
                        
                        # Model predict
                        val_outputs = ema(val_image)
                        
                        # Loss
                        # val_loss = loss_function(val_outputs, val_seg)
                        val_loss = _compute_loss(val_outputs, val_seg, loss_function, CONFIG['deep_supervision_weights'])
                        val_epoch_loss += val_loss.item()  # save it to epoch_loss
            
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

            progress_dir = os.path.join(model_dir, 'progress')
            if not os.path.exists(progress_dir):
                print(f"Create 'progress' folder.")
                os.makedirs(progress_dir, exist_ok=True)
            save_dir = os.path.join(progress_dir, f"fold{fold+1}_progress.png")
            plot_one_fold_results(train_history, val_history, save_dir)
            
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
        plot_cross_validation_results(CV_train_history, CV_val_history, model_dir)
    
    mean_metric = np.mean(best_metrics)
    print(" ")
    print("="*20)
    print(f"All training are completed, average metric of all folds: {mean_metric:.4f}")
    
def _compute_loss(outputs, labels, loss_function, supervision_weights):
        if len(outputs.size()) - len(labels.size()) == 1:
            outputs = torch.unbind(outputs, dim=1)
            loss = sum(
                [
                    supervision_weights[i] * loss_function(output, labels)
                    for i, output in enumerate(outputs)
                ]
            )
        else:
            loss = loss_function(outputs, labels)

        return loss

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", "--root_path", type=str, help="path to root folder, where contains raw_data, preprocessed, and results folders")
    parser.add_argument("-t", "--task_id", type=int, help="task ID")

    args = parser.parse_args()
    main(args)