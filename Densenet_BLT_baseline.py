import os
import sys
import glob
import logging
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import itk
import SimpleITK as sitk
import monai
from monai.data import ImageDataset, DataLoader, decollate_batch
from monai.metrics import ROCAUCMetric
from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirst, Compose, Activations, AsDiscrete,
    Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,ToTensor)

import itk
import SimpleITK as sitk
import monai
from monai.data import ImageDataset, DataLoader, decollate_batch
from monai.metrics import ROCAUCMetric
from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirst, Compose, Activations, AsDiscrete,
    Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,ToTensor)

class EMA:
    def __init__(self, model):
        self.model = model
        self.decay = 1 / 10
        self.shadow = {} # follow the updates of the model parameters
        self.backup = {} # store the original model parameters
        self.num_updates = 0

        # Initialize shadow weights and backup weights with the model's initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self.backup[name] = param.data.clone()

    def update(self):
        # Update the number of updates
        self.num_updates += 1

        # Calculate the decay rate based on the current number of updates
        decay = (1 + self.num_updates) / (10 + self.num_updates)

        # Update the shadow weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                    self.shadow[name] = new_average

    def apply_shadow(self):
        # Apply the shadow weights to the model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # Restore the original model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

####################
### Functions
def check_dataoverlap(train_x, val_x, test_x):
    '''
    Args:
        train_x: a list of training data
        val_x: a list of validation data
        test_x: a list of testing data
    Return:
        print if the data has overlapping elements.
    '''
    # Check dataoverlap
    overlap = set(train_x) & set(val_x) & set(test_x)

    # Check if there is any overlap
    if overlap:
        print(f"The following elements are overlapped: {overlap}")
    else:
        print("The data have no overlapping elements.")
        
def create_empty_predict_lists(device):
    y = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y_pred_argmax = torch.tensor([], dtype=torch.long, device=device)
    return y, y_pred, y_pred_argmax

def calaulate_auc(y, y_pred, y_trans, y_pred_trans):
    '''
    Args:
        y: a tensor containing true labels
        y_pred: a tensor containing prdicted values without argmax
        y_trans: transformation setting for y
        y_pred_trans: transformation setting for y_pred
    Return:
        a float of auc value
    '''
    auc_metric = ROCAUCMetric()
    y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
    y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
    auc_metric(y_pred_act, y_onehot)
    auc_value = auc_metric.aggregate()
    auc_metric.reset()
    del y_pred_act, y_onehot
    return auc_value

def calaulate_metric(y:list, y_pred:list):
    '''
    Args:
        y: a list contatin true labels
        y_pred: a list containing prdicted values after argmax
    Return:
        metric: a tuple containing 4 values of TP, FP, FN, TN
        avg_sensitivity: average sensitivity value
        avg_specificity: average specificity value
    '''
    # Move tensors to CPU if they are on GPU
    if y.is_cuda:
        y = y.cpu()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu()
        
    cm = confusion_matrix(y, y_pred)
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (FP + FN + TP)
    # Calculate Sensitivity(Recall) and Specificity for each class
    sensitivity = TP / (TP + FN)
    avg_sensitivity = sum(sensitivity)/len(sensitivity)
    specificity = TN / (TN + FP)
    avg_specificity = sum(specificity)/len(specificity)
    return (TP, FP, FN, TN), avg_sensitivity, avg_specificity
      

def main():
    ####################
    ### Setup
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%j-%M")
    set_determinism(seed=0)

    ####################
    ### Set path, change if needed
    script_dir = os.getcwd()
    model_dir = os.path.join(script_dir, "data", "BLT_radiomics", "models", "baseline")
    img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_old_NP/images"
    seg4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_old_NP/segs"
    # img4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_old_NP", "images")
    # seg4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_old_NP", "segs")

    ####################
    ### Load the data
    # image files
    images = []
    for file in os.listdir(img4D_dir):
        if file.endswith(".nii.gz"):
            images.append(os.path.join(img4D_dir, file))
        
    # segmentation files
    segs = []
    for file in os.listdir(seg4D_dir):
        if file.endswith(".nii.gz"):
            segs.append(os.path.join(seg4D_dir, file))
        
    # label data
    # label_dir = os.path.join(script_dir, "data", "BLT_radiomics", "labels_all_phases_NEW.csv")
    label_dir = os.path.join(script_dir, "data", "labels_all_phases_NEW.csv")
    labels = pd.read_csv(label_dir)["Pheno"].to_numpy()
    labels[labels == 4] = 2
    labels = torch.from_numpy(labels)
    num_class = len(np.unique(labels, axis=0))

    print(f"image data count: {len(images)}.\nsegmetation data count: {len(segs)}.\nnumber of class: {num_class}.")
    
    ####################
    ## Split data
    # Split dataset into train+val and test sets
    random_state = np.random.RandomState()
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(list(range(len(images))), labels))

    # Further split train+val into train and val sets
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)  # 0.25 x 0.8 = 0.2
    train_idx, val_idx = next(sss2.split(train_val_idx, [labels[i] for i in train_val_idx]))

    seed = random_state.get_state()[1][0]
    print(f"Random state seed used: {seed}")

    # Adjust indices to original dataset
    train_idx = [train_val_idx[i] for i in train_idx]
    val_idx = [train_val_idx[i] for i in val_idx]

    train_x = [images[i] for i in train_idx]
    train_seg = [segs[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]

    val_x = [images[i] for i in val_idx]
    val_seg = [segs[i] for i in val_idx]
    val_y = [labels[i] for i in val_idx]

    test_x = [images[i] for i in test_idx]
    test_seg = [segs[i] for i in test_idx]
    test_y = [labels[i] for i in test_idx]
    check_dataoverlap(train_x, val_x, test_x)
    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

    ####################
    ## Define transforms and dataloader
    # Define transforms
    train_transforms = Compose([
        EnsureChannelFirst(), Resize((78,78,31)),
        # Data augmentation
        RandRotate(range_z = 0.35, prob = 0.3), RandFlip(prob = 0.5), 
        RandGaussianNoise(std=0.05, prob=0.5),
        RandZoom(prob = 0.3, min_zoom=1.0, max_zoom=1.2), ToTensor()])
    val_transforms = Compose([EnsureChannelFirst(), Resize((78,78,31))])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    y_pred_trans = Compose([Activations(softmax=True)])

    # Define image dataset, data loader
    # check_ds = ImageDataset(image_files=images, seg_files=segs, labels=labels,
    #                         transform=train_transforms, seg_transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=17, num_workers=2, pin_memory=torch.cuda.is_available())
    # # check the data
    # im, seg, label = monai.utils.misc.first(check_loader)
    # print(type(im), im.shape, seg.shape, label)

    # create a data loader
    batch_size = 1
    train_ds = ImageDataset(image_files=train_x, seg_files=train_seg, labels=train_y,
                            transform=train_transforms, seg_transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ImageDataset(image_files=val_x, seg_files=val_seg, labels=val_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a test data loader
    test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
    
    ####################
    ## Create the model, loss function and optimizer
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=8, out_channels=num_class, norm='instance').to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    ema = EMA(model)

    ####################
    ## Training & validation
    # settings
    save_model = os.path.join(model_dir, attempt+".pth")
    if os.path.exists(save_model):
        raise ValueError("The model already exists")

    max_epochs = 100
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()

    # store the data
    train_history = {
        "epoch_loss": [],
        "AUC": [],
        "metric": [],
        "sensitivity": [],
        "specificity": []}

    val_history = {
        "AUC": [],
        "metric": [],
        "sensitivity": [],
        "specificity": []}

    for epoch in range(max_epochs):
        print("-" * 15)
        print(f"epoch {epoch + 1}/{max_epochs}")
        print("#"*3+" Training")
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
            masked_train_images = torch.cat((train_images,train_segs), dim=1) 
            
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
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            
            # append the predicted value
            y = torch.cat([y, train_labels], dim=0)
            y_pred = torch.cat([y_pred, train_pred], dim=0)
            y_pred_argmax = torch.cat([y_pred_argmax, train_pred_argmax], dim=0)
        
        auc_value = calaulate_auc(y, y_pred, y_trans, y_pred_trans)
        metric, avg_sensitivity, avg_specificity = calaulate_metric(y, y_pred_argmax)
        train_history["AUC"].append(auc_value)
        train_history["metric"].append(metric)
        train_history["sensitivity"].append(avg_sensitivity)
        train_history["specificity"].append(avg_specificity)
        
        epoch_loss /= step
        train_history["epoch_loss"].append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            print("#"*3+" Validation")
            ema.apply_shadow()
            model.eval()
            
            y_val, y_val_pred, y_val_pred_argmax = create_empty_predict_lists(device)
            
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_segs, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                        val_data[2].to(device))
                    masked_val_images = torch.cat((val_images,val_segs), dim=1)
                    
                    val_pred = model(masked_val_images)
                    val_pred_argmax = val_pred.argmax(dim=1)

                    # append value to the lists
                    y_val = torch.cat([y_val, val_labels], dim=0)
                    y_val_pred = torch.cat([y_val_pred, val_pred], dim=0)
                    y_val_pred_argmax = torch.cat([y_val_pred_argmax, val_pred_argmax], dim=0)
                
                val_auc_value = calaulate_auc(y_val, y_val_pred, y_trans, y_pred_trans)
                val_metric, val_avg_sensitivity, val_avg_specificity = calaulate_metric(y_val, y_val_pred_argmax)
                val_history["AUC"].append(val_auc_value)
                val_history["metric"].append(val_metric)
                val_history["sensitivity"].append(val_avg_sensitivity)
                val_history["specificity"].append(val_avg_specificity) 
                
                if val_auc_value > best_metric:
                    best_metric = val_auc_value
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_model)
                    print(f"Save new best metric model: {attempt}_new.pth")
                print(f"Current epoch: {epoch+1}, AUC: {val_auc_value:.4f}")
                print(f"Sensitivity: {val_avg_sensitivity:.4f}, Specificity: {val_avg_specificity:.4f}")
                print(f"Best AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
                writer.add_scalar("val_AUC", val_auc_value, epoch + 1)
                ema.restore()
    
    ####################
    ## Plotting the loss and training history
    # settings
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = [i + 1 for i in range(len(train_history["epoch_loss"]))]
    y = train_history["epoch_loss"]
    axes[0,0].plot(x, y)
    axes[0,0].set_title("Epoch Average Loss")
    axes[0,0].set_xlabel("epoch")

    x = [val_interval * (i + 1) for i in range(len(train_history["AUC"]))]
    y1 = train_history["AUC"]
    y2 = val_history["AUC"]
    axes[0,1].plot(x, y1, label="training")
    axes[0,1].plot(x, y2, label="validation")
    axes[0,1].legend()
    axes[0,1].set_title("AUC")
    axes[0,1].set_xlabel("epoch")
    axes[0,1].set_ylim(bottom=0);

    x = [val_interval * (i + 1) for i in range(len(train_history["sensitivity"]))]
    y1 = train_history["sensitivity"]
    y2 = val_history["sensitivity"]
    axes[1,0].plot(x, y1, label="training")
    axes[1,0].plot(x, y2, label="validation")
    axes[1,0].legend()
    axes[1,0].set_title("Sensitivity")
    axes[1,0].set_xlabel("epoch")
    axes[1,0].set_ylim(bottom=0);

    x = [val_interval * (i + 1) for i in range(len(train_history["specificity"]))]
    y1 = train_history["specificity"]
    y2 = val_history["specificity"]
    axes[1,1].plot(x, y1, label="training")
    axes[1,1].plot(x, y2, label="validation")
    axes[1,1].legend()
    axes[1,1].set_title("Specificity")
    axes[1,1].set_xlabel("epoch")
    axes[1,1].set_ylim(bottom=0);

    save_dir = os.path.join(model_dir, "snap", attempt + "_snap.png")
    plt.savefig(save_dir)
    
    ####################
    ## Testing
    model.load_state_dict(torch.load(os.path.join(model_dir, attempt+".pth"), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    ema.apply_shadow()
    model.eval()

    y_ts, y_ts_pred, y_ts_pred_argmax = create_empty_predict_lists(device)

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_segs, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2].to(device))
            masked_test_images = torch.cat((test_images,test_segs), dim=1)
            
            ts_pred = model(masked_test_images)
            ts_pred_argmax = ts_pred.argmax(dim=1)
            
            # append the value
            y_ts = torch.cat([y_ts, test_labels], dim=0)
            y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
            y_ts_pred_argmax = torch.cat([y_ts_pred_argmax, ts_pred_argmax], dim=0)
        
        ts_auc_value = calaulate_auc(y_ts, y_ts_pred, y_trans, y_pred_trans)
        ts_metric, ts_avg_sensitivity, ts_avg_specificity = calaulate_metric(y_ts, y_ts_pred_argmax)
        print(f"Model evaluate on testing set; AUC: {ts_auc_value}")
        print(f"Sensitivity: {ts_avg_sensitivity:4f}, Specificity: {ts_avg_specificity:4f}")
       
    ####################
    ## Results       
    TP, FP, FN, TN = ts_metric
    results = np.vstack((TP, FP, FN, TN)).T
    results_df = pd.DataFrame(results, columns=["TP", "FP", "FN", "TN"])
    results_df.index = ["HCA","FNH","HCC"]
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)

    # Adding these metrics to the DataFrame
    results_df["Recall"] = Recall
    results_df["Specificity"] = Specificity
    print("\n## metric")
    print(results_df)
    
    if y_ts.is_cuda:
        y_ts = y_ts.cpu()
    if y_ts_pred_argmax.is_cuda:
        y_ts_pred_argmax = y_ts_pred_argmax.cpu()
    
    print("\n## classification report")
    print(classification_report(y_ts, y_ts_pred_argmax, target_names=["HCA","FNH","HCC"], digits=3))
    
    # Creating  a confusion matrix,which compares the y_test and y_pred
    cm = confusion_matrix(y_ts, y_ts_pred_argmax)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    save_dir = os.path.join(model_dir, "snap", attempt+"_confusion.png")
    plt.savefig(save_dir)
    # plt.show()

if __name__ == "__main__":
    main()
