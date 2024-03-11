import os
import sys
import csv
import logging
import argparse
from datetime import datetime
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

import monai
from monai.data import ImageDataset, DataLoader
from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirst, Compose, Orientation, NormalizeIntensity,
    Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,ToTensor)
from ema_pytorch import EMA

from utils import multiclass_metric, plotting

####################
###  Parse_args  ###
####################
def parse_args():
    """
        Parses inputs from the commandline.
        :return: inputs as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Setting to run DenseNet BLT_radiomics model")
    parser.add_argument("-k", "--kfold", help="The value of kfold. Default = 5.",
                        default=5, required=False)
    parser.add_argument("-e", "--max_epochs", help="Number of max epochs. Default = 100",
                        default=100, required=False)
    parser.add_argument("-l", "--learning_rate", help="Learning rate. Default = 1e-4", 
                        default=1e-4,required=False)
    parser.add_argument("-r", "--random_state", help="Seed for splitting data. Default = 1", 
                        default=1,required=False)

    return parser.parse_args()

####################
###  Functions   ###
####################
def create_empty_predict_lists(device):
    y = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y_pred_argmax = torch.tensor([], dtype=torch.long, device=device)
    return y, y_pred, y_pred_argmax

def empty_history_dict():
    history_dict = {
    "epoch_loss": [],
    "Accuracy": [],
    "Sensitivity": [],
    "Specificity": [],
    "Precision": [],
    "NPV": [],
    "F1_score": [],
    "AUC": [],
    "AverageAccuracy": [],
    "cm": []
    }
    return history_dict

####################
## Save data
def append_metrics(dictionary, Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, cm):
    dictionary["Accuracy"].append(Accuracy)
    dictionary["Sensitivity"].append(Sensitivity)
    dictionary["Specificity"].append(Specificity)
    dictionary["Precision"].append(Precision)
    dictionary["NPV"].append(NPV)
    dictionary["F1_score"].append(F1_score)
    dictionary["AUC"].append(AUC)
    dictionary["AverageAccuracy"].append(AverageAccuracy)
    dictionary["cm"].append(cm)
    return dictionary

def write_csv(dictionary, fold, split, model_dir):
    '''
    Args:
        dictionary: a dictionary containing the loss and metric values
        fold: the k-fold number now it is
        split: tr, ts, or val
    '''
    # Convert dictionary to rows of data
    rows = []
    epochs = range(1, len(next(iter(dictionary.values()))) + 1)
    for epoch in epochs:
        row = [epoch]
        for key in dictionary:
            row.append(dictionary[key][epoch - 1])
        rows.append(row)

    # Write to CSV
    csv_path = os.path.join(model_dir, "csv")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    filename = f"{split}_fold{fold}.csv"
    file_dir = os.path.join(csv_path, filename)
    with open(file_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        # Writing headers
        headers = ['Epoch'] + [key for key in dictionary.keys()]
        writer.writerow(headers)
        # Writing data
        writer.writerows(rows)
    print(f"{filename} created")         
    
################################## MAIN ##################################
def main():
    
    ####################
    ##      Setup     ##
    ####################
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%m-%d-%M")
    set_determinism(seed=0)
    
    ####################
    ## Process arguments
    args = parse_args()
    kfold = int(args.kfold)
    random_state = int(args.random_state)
    learning_rate = float(args.learning_rate)
    max_epochs = int(args.max_epochs)

    print("-"*30)
    print("\n## Settings -- BLT_radiomics (Binary)")
    print(f"Files will be save at: densenet_binary/{attempt}")
    print(f"kfold: {kfold}; random_state: {random_state}")
    print(f"learning rate: {learning_rate}; max_epochs: {max_epochs}")
    
    ####################
    ### Set path, change if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(script_dir, "..", "data", "BLT_radiomics", "models", "densenet_binary")
    model_dir = os.path.join(model_folder, attempt)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False
    
    if GPU_cluster:
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/images"
        seg4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/segs"
        label_dir = os.path.join(script_dir, "..", "data", "labels_all_phases_NEW.csv")
    else:
        img4D_dir = os.path.join(script_dir, "..", "data", "BLT_radiomics", "image_files", "4D_new_registered", "images")
        seg4D_dir = os.path.join(script_dir, "..", "data", "BLT_radiomics", "image_files", "4D_new_registered", "segs")
        label_dir = os.path.join(script_dir, "..", "data", "BLT_radiomics", "labels_all_phases_NEW.csv")
    

    ####################
    ### Load data
    # images and segmentations
    images = []     # image files
    for file in os.listdir(img4D_dir):
        if file.endswith(".nii.gz"):
            images.append(os.path.join(img4D_dir, file))
    segs = []       # segmentation files
    for file in os.listdir(seg4D_dir):
        if file.endswith(".nii.gz"):
            segs.append(os.path.join(seg4D_dir, file))
    
    # label
    labels = pd.read_csv(label_dir)["Malignant"].to_numpy()
    labels = torch.from_numpy(labels)
    num_class = len(np.unique(labels, axis=0))
    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\nNumber of class: {num_class}.\n\n")

    ####################
    ##   Transforms   ##
    ####################
    train_img_transforms = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        NormalizeIntensity(),
        # Data augmentation
        RandZoom(prob = 0.3, min_zoom=1.0, max_zoom=1.2),
        RandRotate(range_z=0.35, prob=0.3),
        RandFlip(prob = 0.5),
        RandGaussianNoise(prob=0.5, std=0.05),
        ToTensor()])

    train_seg_transforms = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        # Data augmentation
        RandZoom(prob=0.3, min_zoom=1.0, max_zoom=1.2),
        RandRotate(range_z=0.35, prob=0.3),
        RandFlip(prob = 0.5),
        ToTensor()])

    val_img_transforms = Compose([
        EnsureChannelFirst(),
        NormalizeIntensity(),
        ToTensor()])
    val_seg_transforms = Compose([
        EnsureChannelFirst(),
        ToTensor()])
    

    ######################
    ## Cross validation ##
    ######################
    # lists to store data from the cv 
    metrics_best = list()
    metrics_final = list()
    models_best = list()
    models_final = list()
    models_finalEMA = list()
    models_bestEMA = list()
    train_history_cv = list()
    val_history_cv = list() 
    
    ALL_y_ts, ALL_y_ts_bm_pred, ALL_y_ts_bm_pred_argmax = create_empty_predict_lists(device)
    _, ALL_y_ts_fm_pred, ALL_y_ts_fm_pred_argmax = create_empty_predict_lists(device)
    
    skf = StratifiedKFold(n_splits=kfold)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(images, labels)):
        print("-"*30)
        print(f"Fold: {fold+1}/{kfold}")
        train_val_x, test_x = [images[i] for i in train_val_idx], [images[i] for i in test_idx]
        train_val_seg, test_seg = [segs[i] for i in train_val_idx], [segs[i] for i in test_idx]
        train_val_y, test_y = [labels[i] for i in train_val_idx], [labels[i] for i in test_idx]
        
        indices = np.arange(len(train_val_x))
        train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=random_state, stratify=train_val_y)
        train_x = [train_val_x[i] for i in train_idx]
        train_seg = [train_val_seg[i] for i in train_idx]
        train_y = [train_val_y[i] for i in train_idx]
        val_x = [train_val_x[i] for i in val_idx]
        val_seg = [train_val_seg[i] for i in val_idx]
        val_y = [train_val_y[i] for i in val_idx]
        
        ####################
        ## Create dataloader
        train_ds = ImageDataset(image_files=train_x, seg_files=train_seg, labels=train_y,
                            transform=train_img_transforms, seg_transform=train_seg_transforms)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
        val_ds = ImageDataset(image_files=val_x, seg_files=val_seg, labels=val_y, 
                            transform=val_img_transforms, seg_transform=val_seg_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
        test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                            transform=val_img_transforms, seg_transform=val_seg_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
    
    
        ####################
        ## Create Model, Loss, Optimizer
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_class, norm='instance').to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        use_ema = True
        max_epochs = max_epochs
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        
        if use_ema:
            ema = EMA(
                model,
                beta = 0.9999, #0.9999,   # exponential moving average factor
                update_after_step = 5,    # only after this number of .update() calls will it start updating
                update_every = 5,         # how often to actually update, to save on compute (updates every 10th .update() call)
                )
            ema.update()

        ######################
        ##  Training
        train_history = empty_history_dict()
        val_history = empty_history_dict()
        test_history_bestmodel = empty_history_dict()
        test_history_finalmodel = empty_history_dict()

        for epoch in range(max_epochs):
            print(" ")
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            print(f"### Training")
            model.train()
            
            epoch_loss = 0
            step = 0
            y, y_pred, y_pred_argmax = create_empty_predict_lists(device)

            for batch_data in train_loader:
                step += 1
                train_images, train_segs, train_labels = (
                    batch_data[0].to(device), 
                    batch_data[1].to(device), 
                    batch_data[2].to(device))
                masked_train_images = torch.cat((train_images,train_segs[:,0:1,:,:]), dim=1) 
                
                # forward and backward pass
                optimizer.zero_grad()
                train_pred = model(masked_train_images)
                train_pred_argmax = train_pred.argmax(dim=1)
                loss = loss_function(train_pred, train_labels)
                loss.backward()
                optimizer.step()
                ema.update()
                epoch_loss += loss.item()
                
                epoch_len = len(train_ds) // train_loader.batch_size
                if (step % 20 == 0) or (step == epoch_len): # only print every 20 and the last steps
                    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                
                # append the predicted values and calculate the metrics
                y = torch.cat([y, train_labels], dim=0)
                y_pred = torch.cat([y_pred, train_pred], dim=0)
                y_pred_argmax = torch.cat([y_pred_argmax, train_pred_argmax], dim=0)
            Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, cm = multiclass_metric.performance_multiclass(y, y_pred_argmax, y_pred)
            train_history = append_metrics(train_history, Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, cm)
            
            epoch_loss /= step
            train_history["epoch_loss"].append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            
            ####################
            ## Validation
            if (epoch + 1) % val_interval == 0:
                print(f"\n### Validation")
                model.eval()
                
                v_epoch_loss = 0
                v_step = 0
                y_val, y_val_pred, y_val_pred_argmax = create_empty_predict_lists(device)
                
                with torch.no_grad():
                    for val_data in val_loader:
                        v_step += 1
                        val_images, val_segs, val_labels = (
                            val_data[0].to(device),
                            val_data[1].to(device),
                            val_data[2].to(device))
                        masked_val_images = torch.cat((val_images,val_segs[:,0:1,:,:]), dim=1)
                        
                        # forward
                        val_pred = model(masked_val_images)
                        if use_ema:
                            val_pred = ema(masked_val_images)
                        val_pred_argmax = val_pred.argmax(dim=1)
                        v_loss = loss_function(val_pred, val_labels)
                        v_epoch_loss += v_loss.item()

                        # append the predicted values and calculate the metrics
                        y_val = torch.cat([y_val, val_labels], dim=0)
                        y_val_pred = torch.cat([y_val_pred, val_pred], dim=0)
                        y_val_pred_argmax = torch.cat([y_val_pred_argmax, val_pred_argmax], dim=0)
                    vAccuracy, vSensitivity, vSpecificity, vPrecision, vNPV, vF1_score, vAUC, vAverageAccuracy, vcm = multiclass_metric.performance_multiclass(y_val, y_val_pred_argmax, y_val_pred)
                    val_history = append_metrics(val_history, vAccuracy, vSensitivity, vSpecificity, vPrecision, vNPV, vF1_score, vAUC, vAverageAccuracy, vcm)
                    
                    v_epoch_loss /= v_step
                    val_history["epoch_loss"].append(v_epoch_loss)
                    
                    if vAUC > best_metric:
                        best_metric = vAUC
                        best_metric_epoch = epoch + 1
                        best_model = model
                        best_model_ema = ema.ema_model
                    print(f"Current epoch: {epoch+1}, AUC: {vAUC:.4f}")
                    print(f"Sensitivity: {vSensitivity:.4f}, Specificity: {vSpecificity:.4f}")
                    print(f"Best AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
            
        ####################
        ## Testing
        print(f"\n### Testing")
        best_model.eval()
        model.eval()

        y_ts, y_ts_bm_pred, y_ts_bm_pred_argmax = create_empty_predict_lists(device)
        _, y_ts_fm_pred, y_ts_fm_pred_argmax = create_empty_predict_lists(device)
        
        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_segs, test_labels = (
                    test_data[0].to(device),
                    test_data[1].to(device),
                    test_data[2].to(device))
                masked_test_images = torch.cat((test_images,test_segs[:,0:1,:,:]), dim=1)
                
                ts_bestmodel_pred = best_model(masked_test_images)
                ts_bestmodel_pred_argmax = ts_bestmodel_pred.argmax(dim=1)
                ts_finalmodel_pred = model(masked_test_images)
                ts_finalmodel_pred_argmax = ts_finalmodel_pred.argmax(dim=1)
                
                # append the value to the list
                y_ts = torch.cat([y_ts, test_labels], dim=0)
                y_ts_bm_pred = torch.cat([y_ts_bm_pred, ts_bestmodel_pred], dim=0)
                y_ts_bm_pred_argmax = torch.cat([y_ts_bm_pred_argmax, ts_bestmodel_pred_argmax], dim=0)
                y_ts_fm_pred = torch.cat([y_ts_fm_pred, ts_finalmodel_pred], dim=0)
                y_ts_fm_pred_argmax = torch.cat([y_ts_fm_pred_argmax, ts_finalmodel_pred_argmax], dim=0)
                
            ALL_y_ts = torch.cat([ALL_y_ts, y_ts], dim=-1)
            ALL_y_ts_bm_pred = torch.cat([ALL_y_ts_bm_pred, y_ts_bm_pred], dim=-2)
            ALL_y_ts_bm_pred_argmax = torch.cat([ALL_y_ts_bm_pred_argmax, y_ts_bm_pred_argmax], dim=-1)
            ALL_y_ts_fm_pred = torch.cat([ALL_y_ts_fm_pred, y_ts_fm_pred], dim=-2)
            ALL_y_ts_fm_pred_argmax = torch.cat([ALL_y_ts_fm_pred_argmax, y_ts_fm_pred_argmax], dim=-1)
            
            test_history_bestmodel["epoch_loss"].append(None)
            test_history_finalmodel["epoch_loss"].append(None)
            tsbmAccuracy, tsbmSensitivity, tsbmSpecificity, tsbmPrecision, tsbmNPV, tsbmF1_score, tsbmAUC, tsbmAverageAccuracy, tsbmcm = multiclass_metric.performance_multiclass(y_ts, y_ts_bm_pred_argmax, y_ts_bm_pred)
            tsfmAccuracy, tsfmSensitivity, tsfmSpecificity, tsfmPrecision, tsfmNPV, tsfmF1_score, tsfmAUC, tsfmAverageAccuracy, tsfmcm = multiclass_metric.performance_multiclass(y_ts, y_ts_fm_pred_argmax, y_ts_fm_pred)
            test_history_bestmodel = append_metrics(test_history_bestmodel, tsbmAccuracy, tsbmSensitivity, tsbmSpecificity, tsbmPrecision, tsbmNPV, tsbmF1_score, tsbmAUC, tsbmAverageAccuracy, tsbmcm)
            test_history_finalmodel = append_metrics(test_history_finalmodel, tsfmAccuracy, tsfmSensitivity, tsfmSpecificity, tsfmPrecision, tsfmNPV, tsfmF1_score, tsfmAUC, tsfmAverageAccuracy, tsfmcm)
            
            print("-"*10)
            print(f"Training and testing for fold {fold+1} is complete.")
            print(f"\nBest metric model evaluate on testing set; AUC: {tsbmAUC:4f}")
            print(f"Sensitivity: {tsbmSensitivity:.4f}, Specificity: {tsbmSpecificity:.4f}")
            print(f"\nFinal model evaluate on testing set; AUC: {tsfmAUC:4f}")
            print(f"Sensitivity: {tsfmSensitivity:.4f}, Specificity: {tsfmSpecificity:.4f}")
        
        # save the data of final epoch
        print("\n## Save model and history")
        models_final.append(model)
        metrics_final.append(vAUC)
        models_best.append(best_model)
        metrics_best.append(best_metric)
        models_finalEMA.append(ema.ema_model)
        models_bestEMA.append(best_model_ema)
        train_history_cv.append(train_history)
        val_history_cv.append(val_history)
        write_csv(train_history, fold+1, "tr", model_dir)
        write_csv(val_history, fold+1, "val", model_dir)
        write_csv(test_history_bestmodel, fold+1, "ts_best", model_dir)
        write_csv(test_history_finalmodel, fold+1, "ts_final", model_dir)
        plotting.plot_cv(train_history_cv, val_history_cv, model_dir)
                  
        print("\n\n")    
    
    # save all the models
    for m in range(len(models_best)):
        file_name = f"model_best_fold_{str(m+1)}.pth"
        save_model = os.path.join(model_dir, file_name)
        torch.save(models_best[m].state_dict(), save_model)
    for m in range(len(models_final)):
        file_name = f"model_final_fold_{str(m+1)}.pth"
        save_model = os.path.join(model_dir, file_name)
        torch.save(models_final[m].state_dict(), save_model)
    for m in range(len(models_finalEMA)):
        file_name = f"model_finalEMA_fold_{str(m+1)}.pth"
        save_model = os.path.join(model_dir, file_name)
        torch.save(models_final[m].state_dict(), save_model)
    for m in range(len(models_bestEMA)):
        file_name = f"model_bestEMA_fold_{str(m+1)}.pth"
        save_model = os.path.join(model_dir, file_name)
        torch.save(models_final[m].state_dict(), save_model)
    
    fig_path = os.path.join(model_dir, "fig")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        
    plotting.plot_cm_roc(ALL_y_ts, ALL_y_ts_bm_pred_argmax, ALL_y_ts_bm_pred, model_name="best", save_path=fig_path)
    plotting.plot_cm_roc(ALL_y_ts, ALL_y_ts_fm_pred_argmax, ALL_y_ts_fm_pred, model_name="final", save_path=fig_path)
    

if __name__ == "__main__":
    main()
