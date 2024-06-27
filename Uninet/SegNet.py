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
from utils.save_results import write_csv, plot_one_fold_results, plot_cross_validation_results
from utils.tasks_by_id import set_results_path_by_id

################
##  Settings  ##
################
TEST_MODE = True # use only a small part of data
VALIDATE_MODEL = True # perform the validation set

################################## MAIN ##################################
def main(args):
    
    ####################
    ##      Setup     ##
    ####################
    run_fold = args.fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%m%d-%M%S")
    set_determinism(seed=4294967295) # use the default seed from monai
    
    root_path = args.root_path
    task_id = str(args.task_id)
    PreprocessTask_dir, ResultsTask_dir = set_results_path_by_id(root_path, task_id)
    img_dir = os.path.join(PreprocessTask_dir, "imagesTr")
    seg_dir = os.path.join(PreprocessTask_dir, "labelsTr")
    
    model_dir = os.path.join(ResultsTask_dir, f"{attempt}_fold_{run_fold}")
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
        if fold == args.fold:
            print(f"\nFold {fold}:")
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
                spatial_dims        =3,
                in_channels         =CONFIG['n_channel'],
                out_channels        =CONFIG['n_channel'] * 2,
                kernel_size         =CONFIG['kernels'],
                strides             =CONFIG["strides"],
                upsample_kernel_size=CONFIG["strides"][1:],
                filters             =CONFIG['filters'],
                norm_name           ="instance",
                act_name            ="leakyrelu",
                deep_supervision    =CONFIG['deep_supervision'],
                deep_supr_num       =CONFIG['deep_supervision_num']
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
            
            # optimizer
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=CONFIG['lr_initial'],
                momentum=0.99,
                weight_decay=3e-5,
                nesterov=True)
            
            # lr scheduler
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (1 - epoch / CONFIG['max_epochs']) ** 0.9)
                
            # loss function
            loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
            # metric function
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            
            ###############
            ##   Train   ##
            ###############
            best_metric, best_metric_epoch = -1, -1
            train_history = {'total_loss': [],
                            'metric_dice': [],}
            val_history = {'total_loss': [],
                        'metric_dice': []}
            
            start_time = time.time()
            for epoch in range(CONFIG['max_epochs']):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{CONFIG['max_epochs']}")
                model.train()
                step, epoch_loss, epoch_dice = 0, 0, 0
                epoch_start_time = time.time()
                # print("divisor: ", divisors)
                
                for x, train_data in enumerate(train_loader):
                    step += 1
                    image, seg, patient, phase = (
                        train_data["image"].to(device),
                        train_data["seg"].to(device),
                        train_data["patient"],
                        train_data["phase"]
                    )
                    
                    # CHECK MEM
                    # print(image.shape)
                    # mem = torch.cuda.memory_allocated(device)
                    # print(f'After load data {x+1}: {(mem/2**30):.2f} GB')
                    
                    # Preprocess
                    phase = int(phase[0])
                    image = add_padding_to_divisible(image, divisors)
                    seg = add_padding_to_divisible(seg, divisors)
                    patient = patient[0]
                    
                    # CHECK MEM
                    print(image.shape)
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After padding {x+1}: {(mem/2**30):.2f} GB')
                    
                    # Model predict
                    outputs = model(image) # [1, (1 + deep_supervision_num), out_channel, *img_shape]
                    
                    # CHECK MEM
                    print(outputs.shape)
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After output {x+1}: {(mem/2**30):.2f} GB')
                    
                    # Loss
                    if CONFIG['n_channel'] == 1: # 3D
                        if CONFIG['deep_supervision']:
                            loss = _compute_loss(outputs=outputs, 
                                                labels=seg, 
                                                loss_function=loss_function,
                                                supervision_weights=CONFIG['deep_supervision_weights'])
                        else:
                            loss = _compute_loss(outputs=outputs, 
                                                labels=seg, 
                                                loss_function=loss_function)
                    else: # 4D
                        if CONFIG['deep_supervision']:
                            loss = _compute_loss(outputs=outputs, 
                                                labels=seg, 
                                                loss_function=loss_function,
                                                supervision_weights=CONFIG['deep_supervision_weights'],
                                                phase=phase,
                                                weight_s=CONFIG['weight_s'])
                        else:
                            loss = _compute_loss(outputs=outputs, 
                                                labels=seg, 
                                                loss_function=loss_function,
                                                phase=phase,
                                                weight_s=CONFIG['weight_s'])
                    epoch_loss += loss.item() # save it to epoch_loss
                    
                    # CHECK MEM
                    # mem = torch.cuda.memory_allocated(device)
                    # print(f'After calculate loss {x+1}: {(mem/2**30):.2f} GB')
                    
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
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After backpropagation {x+1}: {(mem/2**30):.2f} GB')
                    
                    # Metric
                    if CONFIG['deep_supervision']:
                        metric, method, post_outputs = best_post_processing_finder(outputs[:,0,:,:,:,:], seg, dice_metric)
                    else:
                        metric, method, post_outputs = best_post_processing_finder(outputs, seg, dice_metric)
                    epoch_dice += metric
                    
                    # Sanity check
                    if ((epoch + 1) % CONFIG['check_tr_interval']) == 0:
                        
                        # create path to save the file
                        check_dir = os.path.join(model_dir, f'train_check_fold{fold}') 
                        if not os.path.exists(check_dir):
                            print(f"Create train_check_fold{fold} folder for validation output.")
                            os.makedirs(check_dir, exist_ok=True)
                        
                        # execute
                        save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                        # 3D
                        if CONFIG['n_channel'] == 1:
                            Sanity_check.check_seg3d(outputs = post_outputs, 
                                                    data = (image, seg, patient, phase), 
                                                    save_dir = save_dir)
                        # 4D
                        else: 
                            Sanity_check.check_seg4d(outputs = post_outputs, 
                                                data = (image, seg, patient, phase), 
                                                save_dir = save_dir)
                    
                    # print
                    if step % 40 == 0: # only print loss every n batch    
                        print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
                            f"train_loss: {loss.item():.4f}")
                
                    del image, seg, patient, phase, outputs, post_outputs
                    torch.cuda.empty_cache()
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'CLEAN {x+1}: {(mem/2**30):.2f} GB')
                    
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
                    val_step, val_epoch_loss, val_epoch_dice = 0, 0, 0
                    with torch.no_grad():
                        for y, val_data in enumerate(val_loader):
                            val_step += 1
                            image, seg, patient, phase = (
                                val_data["image"].to(device),
                                val_data["seg"].to(device),
                                val_data["patient"],
                                val_data["phase"]
                            )
                            
                            # Preprocess
                            image = add_padding_to_divisible(image, divisors)
                            seg = add_padding_to_divisible(seg, divisors)
                            patient = patient[0]
                            phase = int(phase[0])
                            
                            # CHECK MEM
                            print(image.shape)
                            mem = torch.cuda.memory_allocated(device)
                            print(f'After padding (val) {y+1}: {(mem/2**30):.2f} GB')
                            
                            # Model predict
                            outputs = ema(image)
                            
                            # CHECK MEM
                            print(outputs.shape)
                            mem = torch.cuda.memory_allocated(device)
                            print(f'After output (val) {y+1}: {(mem/2**30):.2f} GB')
                            
                            # Loss
                            if CONFIG['n_channel'] == 1: # 3D
                                if CONFIG['deep_supervision']:
                                    loss = _compute_loss(outputs=outputs, 
                                                        labels=seg, 
                                                        loss_function=loss_function,
                                                        supervision_weights=CONFIG['deep_supervision_weights'])
                                else:
                                    loss = _compute_loss(outputs=outputs, 
                                                        labels=seg, 
                                                        loss_function=loss_function)
                            else: # 4D
                                if CONFIG['deep_supervision']:
                                    loss = _compute_loss(outputs=outputs, 
                                                        labels=seg, 
                                                        loss_function=loss_function,
                                                        supervision_weights=CONFIG['deep_supervision_weights'],
                                                        phase=phase,
                                                        weight_s=CONFIG['weight_s'])
                                else:
                                    loss = _compute_loss(outputs=outputs, 
                                                        labels=seg, 
                                                        loss_function=loss_function,
                                                        phase=phase,
                                                        weight_s=CONFIG['weight_s'])
                            
                            val_epoch_loss += loss.item()  # save it to epoch_loss
                
                            # Metric
                            if CONFIG['deep_supervision']:
                                metric, method, post_outputs = best_post_processing_finder(outputs[:,0,:,:,:,:], seg, dice_metric)
                            else:
                                metric, method, post_outputs = best_post_processing_finder(outputs, seg, dice_metric)
                            val_epoch_dice += metric
                            print(f"post processing (val): {method}")
                            
                            # Sanity check
                            if ((epoch + 1) % CONFIG['check_val_interval'] == 0):
                                # create path to save the file
                                check_dir = os.path.join(model_dir, f'val_check_fold{fold}')
                                if not os.path.exists(check_dir):
                                    print(f"Create 'val_check_fold{fold}' folder for validation output.")
                                    os.makedirs(check_dir, exist_ok=True)
                                # execute
                                save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                                if CONFIG['n_channel'] == 1: # 3D
                                    Sanity_check.check_seg3d(outputs = post_outputs, 
                                                            data = (image, seg, patient, phase), 
                                                            save_dir = save_dir)
                                else: # 4D
                                    Sanity_check.check_seg4d(outputs = post_outputs, 
                                                            data = (image, seg, patient, phase), 
                                                            save_dir = save_dir)
                                    
                                del image, seg, patient, phase, outputs, post_outputs
                                torch.cuda.empty_cache()
                                # CHECK MEM
                                mem = torch.cuda.memory_allocated(device)
                                print(f'CLEAN (val) {y+1}: {(mem/2**30):.2f} GB')
                        
                        val_epoch_loss /= val_step # calculate average epoch loss
                        val_epoch_dice /= val_step
                        
                        val_history['total_loss'].append(val_epoch_loss)
                        val_history['metric_dice'].append(val_epoch_dice)
                        
                        if val_epoch_dice > best_metric:
                            best_metric = val_epoch_dice
                            best_metric_epoch = epoch + 1
                            torch.save(model.state_dict(), os.path.join(model_dir, f"seg_fold{fold}.pth"))
                            print("saved new best metric model")
                            
                        print(
                            f"Validation -- epoch {epoch + 1} average loss: {val_epoch_loss:.4f}, average dice: {val_epoch_dice:.4f}"
                            f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
                        print(f"-- {(time.time()-epoch_start_time):.4f} seconds --")

                save_dir = os.path.join(model_dir, f"progress.png")
                plot_one_fold_results(train_history, val_history, save_dir)
                
            torch.save(model.state_dict(), os.path.join(model_dir, f"seg_fold{fold}_final.pth"))
            print("saved the final model")
            
            print("+"*30)
            print(f"Fold {fold} train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
            print(f"--- {(time.time()-start_time):.4f} seconds ---")
            best_metrics.append(best_metric)
            
            save_dir = os.path.join(model_dir, f"training_log_fold{fold}.csv")
            write_csv(train_history, save_dir)
            save_dir = os.path.join(model_dir, f"validation_log_fold{fold}.csv")
            write_csv(val_history, save_dir)
            print(" ")
            CV_train_history.append(train_history)
            CV_val_history.append(val_history)
            # plot
            plot_cross_validation_results(CV_train_history, CV_val_history, model_dir)
    
    mean_metric = np.mean(best_metrics)
    print(" ")
    print("="*20)
    print(f"All training are completed, average metric of all folds: {mean_metric:.4f}")


def _compute_loss(outputs, labels, loss_function, supervision_weights=None, phase=None, weight_s=None):
    def compute_pair_loss(output, labels, num_channels, weight_s):
        pair_loss = 0
        num_pairs = num_channels // 2

        for pair_index, channel in enumerate(range(0, num_channels, 2)):
            current_pair_loss = loss_function(output[:, channel:channel+2, ...], labels)
            
            # If current pair index is not the gt phase, apply scaling weight
            if pair_index != phase:
                current_pair_loss *= weight_s

            pair_loss += current_pair_loss

        # Average the loss over all pairs
        return pair_loss / num_pairs
    
    # If deep supervision is performed
    if len(outputs.size()) - len(labels.size()) == 1:
        outputs = torch.unbind(outputs, dim=1)  # Unbind batch into list of tensors
        losses = []

        for i, output in enumerate(outputs):
            num_channels = output.size()[1]

            if num_channels > 2:
                loss = compute_pair_loss(output, labels, num_channels, weight_s)
            else:
                loss = loss_function(output, labels)

            # Apply supervision weights to the calculated pair loss
            weighted_loss = supervision_weights[i] * loss
            losses.append(weighted_loss)

        total_loss = sum(losses)
    
    # No deep supervision
    else:
        num_channels = outputs.size()[1]

        if num_channels > 2:
            total_loss = compute_pair_loss(outputs, labels, num_channels, weight_s)
        else:
            total_loss = loss_function(outputs, labels)

    return total_loss


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", "--root_path", type=str, help="path to root folder, where contains raw_data, preprocessed, and results folders")
    parser.add_argument("-t", "--task_id", type=int, help="task ID")
    parser.add_argument("-f", "--fold", type=int, help="run fold [0, 1, 2, 3, 4]")

    args = parser.parse_args()
    main(args)