import os
import re
import gc
import glob
import time
import numpy as np
from datetime import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from ema_pytorch import EMA
from sklearn.model_selection import KFold

import monai
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric

from model.dynunet_uni import DynUniNet
from model.spatial_transformer import SpatialTransformer
from loss_function import DiceCELoss, ConsistencyLoss, NCC, Grad
from utils.functions import *
from utils.sanity_check import Sanity_check
from utils.file_and_folder_operations import load_json
from utils.transforms import training_transforms, best_post_processing_finder
from utils.tasks_by_id import set_results_path_by_id
from utils.save_results import write_csv, plot_one_fold_results, plot_cross_validation_results

################
##  Settings  ##
################
TEST_MODE = False # use only a small part of data
VALIDATE_MODEL = True # perform the validation set

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    return device

def setup_path_and_plans(root, ID, run_fold):
    # Path
    PreprocessTask_dir, ResultsTask_dir = set_results_path_by_id(root, ID)
    img_dir = os.path.join(PreprocessTask_dir, "imagesTr")
    seg_dir = os.path.join(PreprocessTask_dir, "labelsTr")
    
    attempt = datetime.now().strftime("%m%d-%M%S")
    model_dir = os.path.join(ResultsTask_dir, f"{attempt}_uninet_fold_{run_fold}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Plans
    plans = load_json(os.path.join(img_dir, "..", "plans.json"))
    CONFIG = plans['model_configs']
    
    print("\n=== CONFIG ===")
    print("Task: Joint segmentation and registration")
    max_key_length = max(len(key) for key in CONFIG.keys())
    for key, value in CONFIG.items():
        print(f'{key.upper():<{max_key_length + 3}}{value}')
        
    print("\n=== Data directory ===")
    print(f"IMAGES: {img_dir}")
    print(f"SEGMENTATION: {seg_dir}")
    print(f"RESULTS: {model_dir}")
    
    return img_dir, seg_dir, model_dir, CONFIG

def load_data(img_dir, seg_dir, test_mode):
    images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    segs = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))
    pattern = r"_([0-9]+)_([0-9]+)\.nii\.gz"
    patients, phases = [], []
    for file in sorted(os.listdir(seg_dir)):
        match = re.search(pattern, file)
        if match:
            patients.append(match.group(1))
            phases.append(match.group(2))
    
    if test_mode:
        images = images[:10]
        segs = segs[:10]
        patients = patients[:10]
        phases = phases[:10]
        
    data_dicts = [{"image": image, "seg": seg, "patient":patient, "phase":phase} 
                  for image, seg, patient, phase in zip(images, segs, patients, phases)]
    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\n")
    return data_dicts

def setup_model(config, device):
    model = DynUniNet(
        spatial_dims        =3,
        in_channels         =config['n_channel'],
        out_channels        =[config['n_channel'] * 2, config['n_channel'] * 3],
        kernel_size         =config['kernels'],
        strides             =config["strides"],
        upsample_kernel_size=config["strides"][1:],
        filters             =config['filters'],
        norm_name           ="instance",
        act_name            ="leakyrelu",
        deep_supervision    =config['deep_supervision'],
        deep_supr_num       =config['deep_supervision_num']
        ).to(device)
    divisors = calculate_reduction_factors(config["strides"])
    
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
        lr=config['lr_initial'],
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )
    
    # lr scheduler
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - epoch / config['max_epochs']) ** 0.9,
    )
    
    # spatial transformer
    spatial_transformer = SpatialTransformer(dim=3).to(device)
    
    # loss functions
    loss_functions = {}
    loss_functions['dice'] = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_functions['consis'] = ConsistencyLoss()
    loss_functions['ncc'] = NCC(win=11)
    loss_functions['grad'] = Grad(penalty='l2')
    
    # metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    return model, divisors, ema, optimizer, scheduler, spatial_transformer, loss_functions, dice_metric


def outputs_handler(outputs_seg, outputs_reg, image, seg, spatial_transformer, dice_metric, config):
    # Calculate `displacement field` using outputs_reg
    disp_t2i = torch.squeeze(outputs_reg, 0).reshape(config['n_channel'], 3, *outputs_reg.shape[2:])
    
    # We also need to warp the predicted segmetations, so first postprocess it
    if config['deep_supervision']:
        metric_dice, _, post_outputs_seg = best_post_processing_finder(outputs_seg[:,0,:,:,:,:], seg, dice_metric)
    else:
        metric_dice, _, post_outputs_seg = best_post_processing_finder(outputs_seg, seg, dice_metric)

    # `post_outputs_seg` is a list, we need to modify it to a tensor and same dimension as image
    post_seg_tensor = torch.stack([t[1] for t in post_outputs_seg])
    post_seg_tensor = post_seg_tensor.unsqueeze(0)
    
    # Warping with spatial transformer
    warped_image = spatial_transformer(image, disp_t2i)
    warped_post_seg = spatial_transformer(post_seg_tensor, disp_t2i)
    
    return warped_image, warped_post_seg, disp_t2i, metric_dice
 

def _create_history_dict():
    history = {'total_loss': [],
                'loss_dice': [],
                'loss_consis': [],
                'loss_simi': [],
                'loss_smooth': [],
                'metric_dice': [],
                'metric_ncc': []}
    return history

def _compute_losses(warped_image, warped_seg, disp_t2i, outputs_seg,
                    labels, phase,
                    loss_functions, config):
    
    def compute_pair_loss(outputs, labels, weight_s, dice_loss_function):
        pair_loss = 0
        num_pairs = outputs.size()[1] // 2

        for pair_index, channel in enumerate(range(0, outputs.size()[1], 2)):
            current_pair_loss = dice_loss_function(outputs[:, channel:channel+2, ...], labels)
            
            # If current pair index is not the gt phase, apply scaling weight
            if pair_index != phase:
                current_pair_loss *= weight_s

            pair_loss += current_pair_loss

        # Average the loss over all pairs
        return pair_loss / num_pairs
    
    def compute_dice_loss(outputs_seg, labels, loss_function):
        # If deep supervision is performed
        if len(outputs_seg.size()) - len(labels.size()) == 1:
            outputs_seg = torch.unbind(outputs_seg, dim=1)  # Unbind batch into list of tensors
            losses = []

            for i, output in enumerate(outputs_seg):
                loss = compute_pair_loss(output, labels, config['weight_s'], loss_function)
                # Apply supervision weights to the calculated pair loss
                weighted_loss = config['deep_supervision_weights'][i] * loss
                losses.append(weighted_loss)

            total_loss = sum(losses)
        
        # No deep supervision
        else:
            total_loss = compute_pair_loss(outputs_seg, labels, config['weight_s'], loss_function)

        return total_loss
    
    loss_dice = compute_dice_loss(outputs_seg, labels, loss_functions['dice'])
    loss_consis = loss_functions['consis'](warped_seg)
    loss_simi = loss_functions['ncc'](warped_image, torch.mean(warped_image, dim=1, keepdim=True))
    loss_smooth = loss_functions['grad'](disp_t2i)
    ncc_metric = -loss_simi
    
    total_loss = loss_dice + (loss_simi + (config['weight_r'] * loss_smooth)) + loss_consis 
    losses = [total_loss, loss_dice, loss_consis, loss_simi, loss_smooth, ncc_metric]
    return total_loss, losses

def _update_epoch_stats(stats, losses):
    keys = list(stats.keys())
    for i, key in enumerate(keys):
        if key == 'metric_dice':
            continue
        else: #total_loss, loss_dice, loss_consis, loss_simi, loss_smooth, ncc_metric
            stats[key] += losses[i].item()
    return stats

################################## MAIN ##################################
def main(args):
    
    ####################
    ##      Setup     ##
    ####################
    run_fold = args.fold
    device = setup_device()
    img_dir, seg_dir, model_dir, CONFIG = setup_path_and_plans(args.root_path, str(args.task_id), run_fold)
    assert CONFIG['n_channel'] > 1, "The script is for joint segmentation and registration. The number of channel should be more than one."
    set_determinism(seed=4294967295)
    
    ####################
    ##   Transforms   ##
    ####################
    train_transforms = training_transforms(seed=4294967295)
    val_transforms = training_transforms(validation=True)
    
    #################
    ##  Load data  ##
    #################
    print("\n===============")
    print("##### Training #####")
    data_dicts = load_data(img_dir, seg_dir, TEST_MODE)
    CV_train_history, CV_val_history = [], []
    best_metrics = []
    indices = np.arange(len(data_dicts))
    
    ##########################
    ##   Cross-Validation   ##
    ##########################
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

            model, divisors, ema, optimizer, scheduler, spatial_transformer, loss_functions, dice_metric = setup_model(CONFIG, device)
        
            ###############
            ##   Train   ##
            ###############
            best_loss, best_loss_epoch = 1, -1
            train_history, val_history = _create_history_dict(), _create_history_dict()
            
            start_time = time.time()
            for epoch in range(CONFIG['max_epochs']):
                print("-" * 10)
                print(f"epoch {epoch + 1}/{CONFIG['max_epochs']}")
                model.train()
                step = 0
                stats = {
                    'total_loss': 0,
                    'loss_dice': 0, 'loss_consis': 0, 'loss_simi': 0, 'loss_smooth': 0, 
                    'metric_ncc': 0, 'metric_dice': 0
                }
                
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
                    print("======NEW IMG====== ", image.shape)
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After load data {x+1}: {(mem/2**30):.2f} GB')
                    # old_img_shape = np.array(image.shape[2:])
                    
                    # Preprocess
                    image = add_padding_to_divisible(image, divisors)
                    seg = add_padding_to_divisible(seg, divisors)
                    patient = patient[0]
                    phase = int(phase[0])
                    # print("Difference: ", np.array(image.shape[2:]) - old_img_shape)
                    
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After padding {x+1}: {(mem/2**30):.2f} GB')
                        
                    # Model prediction
                    outputs_seg, outputs_reg = model(image)
                    
                    
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After prediction {x+1}: {(mem/2**30):.2f} GB')
                    
                    warped_image, warped_post_seg, disp_t2i, metric_dice = outputs_handler(
                        outputs_seg, outputs_reg, image, seg, spatial_transformer, dice_metric, CONFIG)
                    
                    # CLEAN
                    del outputs_reg
                    torch.cuda.empty_cache()
                    
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After post output {x+1}: {(mem/2**30):.2f} GB')
                    
                    # Sanity check
                    if (epoch+1 == 1) or ((epoch + 1) % CONFIG['check_tr_interval']) == 0:
                        check_dir = os.path.join(model_dir, f'train_check_fold{fold}')
                        # create path to save the file
                        if not os.path.exists(check_dir):
                            print(f"Create 'train_check_fold{fold}' folder for training output.")
                            os.makedirs(check_dir, exist_ok=True)
                        save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{x+1}.png")
                        Sanity_check.check_joint4d(outputs = (warped_image, warped_post_seg), 
                                                data = (image, seg, patient, phase), 
                                                save_dir = save_dir)
                    
                    # Losses
                    if CONFIG['deep_supervision']:
                        total_loss, losses = _compute_losses(warped_image, warped_post_seg, disp_t2i, outputs_seg,
                                                            seg, phase,
                                                            loss_functions, CONFIG)
                    else:
                        total_loss, losses = _compute_losses(warped_image, warped_post_seg, disp_t2i, outputs_seg,
                                                            seg, phase,
                                                            loss_functions, CONFIG)
                    # Update result
                    stats['metric_dice'] += metric_dice
                    stats = _update_epoch_stats(stats, losses)
                    
                    # CLEAN
                    del image, seg, patient, phase
                    del warped_image, warped_post_seg, disp_t2i, outputs_seg
                    torch.cuda.empty_cache()
                    
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'CLEAN {x+1}: {(mem/2**30):.2f} GB')
                    
                    ################### TRAIN ONLY ###################
                    # Backpropagation
                    total_loss.backward()
                    # Update parameters // Gradient accumulation // EMA
                    if step % CONFIG['accumulate_grad_batches'] == 0:
                        optimizer.step()
                        ema.update()
                        optimizer.zero_grad() # clear gradients
                    ################### TRAIN ONLY ###################
                    
                    # CHECK MEM
                    mem = torch.cuda.memory_allocated(device)
                    print(f'After backpropagation {x}: {(mem/2**30):.2f} GB')
                    
                    # Print info every n batch 
                    if step % 20 == 0:  
                        print(f"{step}/{len(train_ds) // train_loader.batch_size}")
                        losses_var = ['total_loss', 'loss_dice', 'loss_consis', 'loss_simi', 'loss_smooth']
                        max_key_length = max(len(var) for var in losses_var)
                        for i, var in enumerate(losses_var):
                            print(f'{var.upper():<{max_key_length + 3}}{losses[i]:.4f}')
                        print("---")
                    
                scheduler.step()
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    print(f"Current Learning Rate: {current_lr}")
                        
                # calculate mean losses for this epoch
                print(stats)
                print("step: ", step)
                for key in stats.keys():
                    stats[key] /= step
                print('--average--')
                print(stats)
                
                # append to history
                for stat_key in stats:
                    train_history[stat_key].append(stats[stat_key])
                
                print(
                    f"Epoch {epoch + 1}, avg total loss: {stats['total_loss']:.4f}"
                    f"\ntraining time: {(time.time()-epoch_start_time):.4f} seconds")

                
                ####################
                ##   Validation   ##
                ####################
                if VALIDATE_MODEL:
                    model.eval()
                    val_stats = {
                        'total_loss': 0,
                        'loss_dice': 0, 'loss_consis': 0, 'loss_simi': 0, 'loss_smooth': 0, 
                        'metric_ncc': 0, 'metric_dice': 0}
                    val_step = 0
                    
                    with torch.no_grad():
                        for y, val_data in enumerate(val_loader):
                            val_step += 1
                            image, seg, patient, phase = (
                                val_data["image"].to(device),
                                val_data["seg"].to(device),
                                val_data["patient"],
                                val_data["phase"]
                            )
                            
                            print("======NEW VAL IMG====== ", image.shape)
                            mem = torch.cuda.memory_allocated(device)
                            print(f'After load data {x+1}: {(mem/2**30):.2f} GB')
                            
                            # Preprocess
                            image = add_padding_to_divisible(image, divisors)
                            seg = add_padding_to_divisible(seg, divisors)
                            patient = patient[0]
                            phase = int(phase[0])
                            
                            # Model prediction
                            outputs_seg, outputs_reg = ema(image)
                            
                            # CHECK MEM
                            mem = torch.cuda.memory_allocated(device)
                            print(f'After prediction {x+1}: {(mem/2**30):.2f} GB')
                            
                            warped_image, warped_post_seg, disp_t2i, metric_dice = outputs_handler(
                                outputs_seg, outputs_reg, image, seg, spatial_transformer, dice_metric, CONFIG)

                            del outputs_reg
                            torch.cuda.empty_cache()
                            
                        
                            # Sanity check
                            if (epoch+1 == 1) or ((epoch + 1) % CONFIG['check_val_interval']) == 0:
                                check_dir = os.path.join(model_dir, f'val_check_fold{fold}')
                                # create path to save the file
                                if not os.path.exists(check_dir):
                                    print(f"Create 'val_check_fold{fold}' folder for validation output.")
                                    os.makedirs(check_dir, exist_ok=True)
                                save_dir = os.path.join(check_dir, f"epoch{epoch+1}_{y+1}.png")
                                Sanity_check.check_joint4d(outputs = (warped_image, warped_post_seg), 
                                                data = (image, seg, patient, phase), 
                                                save_dir = save_dir)
                            
                            # Losses
                            if CONFIG['deep_supervision']:
                                total_loss, losses = _compute_losses(warped_image, warped_post_seg, disp_t2i, outputs_seg,
                                                                    seg, phase,
                                                                    loss_functions, CONFIG)
                            else:
                                total_loss, losses = _compute_losses(warped_image, warped_post_seg, disp_t2i, outputs_seg,
                                                                    seg, phase,
                                                                    loss_functions, CONFIG)
                            val_stats['metric_dice'] += metric_dice
                            val_stats = _update_epoch_stats(val_stats, losses)
                            
                            # Print info every n batch
                            if val_step % 20 == 0:  
                                print(f"{val_step}/{len(val_ds) // val_loader.batch_size}")
                                losses_var = ['total_loss', 'loss_dice', 'loss_consis', 'loss_simi', 'loss_smooth']
                                max_key_length = max(len(var) for var in losses_var)
                                for i, var in enumerate(losses_var):
                                    print(f'{var.upper():<{max_key_length + 3}}{losses[i]:.4f}')
                                print("---")
                        
                        # calculate mean losses for this epoch
                        print(val_stats)
                        print("step: ", val_step)
                        for key in val_stats.keys():
                            val_stats[key] /= val_step
                        print('--val average--')
                        print(val_stats)
                        
                        # append to history
                        for stat_key in val_stats:
                            val_history[stat_key].append(val_stats[stat_key])
                    
                        if val_stats['total_loss'] < best_loss:
                            best_loss = val_stats['total_loss']
                            best_loss_epoch = epoch + 1
                            torch.save(model.state_dict(), os.path.join(model_dir, f"joint_fold{fold}.pth"))
                            print("Save the new best loss model!")
                        # if metric > best_metric:
                        #     best_metric = metric
                        #     best_metric_epoch = epoch + 1
                        #     torch.save(model.state_dict(), os.path.join(model_dir, f"model_fold{fold+1}.pth"))
                        #     print("saved new best metric model")
                            
                        print(
                            f"\nValidation -- epoch: {epoch + 1}, average total loss: {val_stats['total_loss']:.4f}"
                            f"\nBest loss: {best_loss:.4f} at epoch: {best_loss_epoch}"
                        )
                
                plot_one_fold_results(train_history, val_history, model_dir)
                save_dir = os.path.join(model_dir, f"training_log_fold{fold}.csv")
                write_csv(train_history, save_dir)
                save_dir = os.path.join(model_dir, f"validation_log_fold{fold}.csv")
                write_csv(val_history, save_dir)
            
            torch.save(model.state_dict(), os.path.join(model_dir, f"joint_fold{fold}_final.pth"))
            print("saved the final model")
            
            print("+"*30)
            print(f"Fold {fold+1} train completed, best_loss: {best_loss:.4f} " f"at epoch: {best_loss_epoch}")
            print(f"--- {(time.time()-start_time):.4f} seconds ---")
            best_metrics.append(best_loss)
            
            # save the logs
            
            # append to CV history
            CV_train_history.append(train_history)
            CV_val_history.append(val_history)
            # plot
            plot_cross_validation_results(CV_train_history, CV_val_history, model_dir)
        
    best_fold_metric = min(best_metrics)
    best_fold_metric_idx = best_metrics.index(best_fold_metric) + 1
    print(" ")
    print("="*20)
    print(f"All training are completed. Best losses from each fold: ")
    for number in best_metrics:
        print(f"{number:.4f}")
    print(f"Best_loss: {best_fold_metric:.4f} " f"at fold: {best_fold_metric_idx}")










if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-root", "--root_path", type=str, help="path to root folder, where contains raw_data, preprocessed, and results folders")
    parser.add_argument("-t", "--task_id", type=int, help="task ID")
    parser.add_argument("-f", "--fold", type=int, help="run fold [0, 1, 2, 3, 4]")
    
    args = parser.parse_args()
    main(args)