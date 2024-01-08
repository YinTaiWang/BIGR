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

def check_dataoverlap(train_x, val_x, test_x):
    # Check dataoverlap
    overlap = set(train_x) & set(val_x) & set(test_x)

    # Check if there is any overlap
    if overlap:
        print(f"The following elements are overlapped: {overlap}")
    else:
        print("The data have no overlapping elements.")
        

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    attempt = datetime.now().strftime("%Y-%j-%M")
    set_determinism(seed=0)

    ####################
    ## set path
    script_dir = os.getcwd()
    model_dir = os.path.join(script_dir, "data", "BLT_radiomics", "models")
    img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/images"
    seg4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/segs"
    # data_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files")
    # img4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "4D_old_registered", "images")
    # seg4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "4D_old_registered", "segs")

    ####################
    ## Load files
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

    # print(images)
    # print(segs)
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
    train_ds = ImageDataset(image_files=train_x, seg_files=train_seg, labels=train_y,
                            transform=train_transforms, seg_transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=15, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ImageDataset(image_files=val_x, seg_files=val_seg, labels=val_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=25, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a test data loader
    test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=25, num_workers=2, pin_memory=torch.cuda.is_available())
    
    ####################
    ## Create the model
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    auc_metric = ROCAUCMetric()
    ema = EMA(model)

    ####################
    ## Training
    # settings
    max_epochs = 100
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()

    # store the data
    history = {
        "epoch_loss_values": [],
        "tr_AUC_values": [],
        "ts_AUC_values": [],
        "ts_ACC_values": [],
        "lrs": []
    }

    for epoch in range(max_epochs):
        print("-" * 15)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            y = torch.tensor([], dtype=torch.long, device=device)
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            
            train_images, train_segs, train_labels = (
                batch_data[0].to(device), 
                batch_data[1].to(device), 
                batch_data[2].to(device))
            masked_train_images = torch.cat((train_images,train_segs[:,1:2,:,:]), dim=1) 
            
            optimizer.zero_grad()
            # print(f"current lr: {optimizer.param_groups[0]['lr']}")
            train_pred = model(masked_train_images)
            loss = loss_function(train_pred, train_labels)
            loss.backward()
            optimizer.step()
            # lrs.append(optimizer.param_groups[0]["lr"])
            # scheduler.step()
            ema.update()
            epoch_loss += loss.item()
            
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            
        # Calculate train AUC
            y = torch.cat([y, train_labels], dim=0)
            y_pred = torch.cat([y_pred, train_pred], dim=0)
        y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_value = auc_metric.aggregate()
        auc_metric.reset()
        del y_pred_act, y_onehot
        history["tr_AUC_values"].append(auc_value)
        
        epoch_loss /= step
        history["epoch_loss_values"].append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            print("-"*3+"Validate"+"-"*3)
            ema.apply_shadow()
            model.eval()
            
            num_correct = 0.0
            metric_count = 0
            
            with torch.no_grad():
                for val_data in val_loader:
                    y_val = torch.tensor([], dtype=torch.long, device=device)
                    y_val_pred = torch.tensor([], dtype=torch.float32, device=device)
                    
                    val_images, val_segs, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                        val_data[2].to(device))
                    masked_val_images = torch.cat((val_images,val_segs[:,1:2,:,:]), dim=1) # use mask
                    
                    val_pred = model(masked_val_images)
                    val_argmax = val_pred.argmax(dim=1)
                    val_acc = torch.eq(val_argmax, val_labels)
                    metric_count += len(val_acc)
                    num_correct += val_acc.sum().item()
                    
                # Calculate AUC
                    y_val = torch.cat([y_val, val_labels], dim=0)
                    y_val_pred = torch.cat([y_val_pred, val_pred], dim=0)
                y_val_onehot = [y_trans(i) for i in decollate_batch(y_val, detach=False)]
                y_val_pred_act = [y_pred_trans(i) for i in decollate_batch(y_val_pred)]
                auc_metric(y_val_pred_act, y_val_onehot)
                val_auc_value = auc_metric.aggregate()
                auc_metric.reset()
                del y_val_pred_act, y_val_onehot
                history["ts_AUC_values"].append(val_auc_value)
                
                # Calculate Accuracy
                val_acc_value = num_correct / metric_count
                history["ts_ACC_values"].append(val_acc_value)
                
                if val_auc_value > best_metric:
                    best_metric = val_auc_value
                    best_metric_epoch = epoch + 1
                    save_model = os.path.join(model_dir, attempt+"_new.pth")
                    torch.save(model.state_dict(), save_model)
                    print("saved new best metric model")
                print(f"Current epoch: {epoch+1}")
                print(f"AUC: {val_auc_value:.4f}; Accuracy: {val_acc_value:.4f}")
                print(f"Best AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
                writer.add_scalar("val_AUC", val_auc_value, epoch + 1)
                ema.restore()
        
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    writer.close()
        
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    writer.close()

    plt.figure("train", (25, 6))
    plt.subplot(1, 3, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(history["epoch_loss_values"]))]
    y = history["epoch_loss_values"]
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(1, 3, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(history["tr_AUC_values"]))]
    y1 = history["tr_AUC_values"]
    y2 = history["ts_AUC_values"]
    plt.xlabel("epoch")
    plt.plot(x, y1, label="training")
    plt.plot(x, y2, label="validation")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Val accuracy")
    x = [val_interval * (i + 1) for i in range(len(history["ts_ACC_values"]))]
    y = history["ts_ACC_values"]
    plt.xlabel("epoch")
    plt.plot(x, y)
    save_dir = os.path.join(model_dir, "snap", attempt+"_new_snap.png")
    plt.savefig(save_dir)
    plt.show()
    
    
    # Calculate mean and standard deviation for both training and test AUC
    mean_ts = np.mean(history["ts_AUC_values"])
    std_ts = np.std(history["ts_AUC_values"])
    mean_tr = np.mean(history["tr_AUC_values"])
    std_tr = np.std(history["tr_AUC_values"])

    # Create an array of x values
    x = range(len(history["ts_AUC_values"]))

    # Plot test AUC values
    plt.plot(x, history["ts_AUC_values"], label='Test AUC', color='#06592A')
    plt.fill_between(x, history["ts_AUC_values"] - std_ts, history["ts_AUC_values"] + std_ts, color='#06592A', alpha=0.2, label='Test Mean ± Std Dev')

    # Plot training AUC values
    plt.plot(x, history["tr_AUC_values"], label='Train AUC', color='#1F77B4')
    plt.fill_between(x, history["tr_AUC_values"] - std_tr, history["tr_AUC_values"] + std_tr, color='#1F77B4', alpha=0.2, label='Train Mean ± Std Dev')
    
    plt.axhline(y=0.5, color='black', linestyle='--', lw=0.8, label='Random')
    plt.xlim(0, len(history["ts_AUC_values"]))
    plt.ylim(bottom=0)
    plt.title('Training and Validation AUC')
    plt.legend()
    save_dir = os.path.join(model_dir, "snap", attempt + "_new_AUC_snap.png")
    plt.savefig(save_dir)
    plt.show()
    
    ####################
    ## Testing
    model.load_state_dict(torch.load(os.path.join(model_dir, attempt+"_new.pth"), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    ema.apply_shadow()
    with torch.no_grad():
        model.eval()
        y_ts_true_values = []
        y_ts_pred_values = []
        
        for test_data in test_loader:
            y_ts_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_ts = torch.tensor([], dtype=torch.long, device=device)
            test_images, test_segs, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2].to(device))
            
            masked_test_images = torch.cat((test_images,test_segs[:,1:2,:,:]), dim=1)
            ts_pred = model(masked_test_images)
            ts_pred_argmax = ts_pred.argmax(dim=1)
            
            for i in range(len(ts_pred_argmax)):
                y_ts_true_values.append(test_labels[i].item())
                y_ts_pred_values.append(ts_pred_argmax[i].item())
                
            y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
            y_ts = torch.cat([y_ts, test_labels], dim=0)
            
        y_ts_onehot = [y_trans(i) for i in decollate_batch(y_ts, detach=False)]
        y_ts_pred_act = [y_pred_trans(i) for i in decollate_batch(y_ts_pred)]
        auc_metric(y_ts_pred_act, y_ts_onehot)
        ts_auc_value = auc_metric.aggregate()
        print(f"Test AUC: {ts_auc_value}")
       
    ####################
    ## Results       
    print(classification_report(y_ts_true_values, y_ts_pred_values, target_names=["HCA","FNH","HCC"], digits=3))
    
    # Creating  a confusion matrix,which compares the y_test and y_pred
    cm = confusion_matrix(y_ts_true_values, y_ts_pred_values)
    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    save_dir = os.path.join(model_dir, "snap", attempt+"_confusion.png")
    plt.savefig(save_dir)
    plt.show()

if __name__ == "__main__":
    main()
