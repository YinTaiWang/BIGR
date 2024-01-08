import os
import sys
import glob
import shutil
import tempfile
import logging
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
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
    Resize, ScaleIntensity, RandRotate, RandFlip)



def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    attemp = datetime.now().strftime("%Y-%j-%M")
    set_determinism(seed=0)

    ####################
    ## set path
    script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files")
    result_dir = os.path.join(script_dir, "data", "BLT_radiomics", "models")
    img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_old_registered/images"
    seg4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_old_registered/segs"
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
    label_dir = os.path.join(script_dir, "data", "labels_all_phases_NEW.csv")
    # change label to numpy type
    labels = pd.read_csv(label_dir)["Pheno"].to_numpy()
    labels[labels == 4] = 2
    # labels = pd.read_csv(label_dir)[['HCA', 'FNH', 'HCC']].to_numpy()
    num_class = len(np.unique(labels, axis=0))
    print(f"image data count: {len(images)}.\nsegmetation data count: {len(segs)}.\nnumber of class: {num_class}")

    ####################
    ## Split data
    # Split dataset into train+val and test sets
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_val_idx, test_idx = next(sss1.split(list(range(len(images))), labels))

    # Further split train+val into train and val sets
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)  # 0.25 x 0.8 = 0.2
    train_idx, val_idx = next(sss2.split(train_val_idx, [labels[i] for i in train_val_idx]))

    # Adjust indices to original dataset
    train_idx = [train_val_idx[i] for i in train_idx]
    val_idx = [train_val_idx[i] for i in val_idx]

    # val_frac = 0.1
    # test_frac = 0.1
    # length = len(images)
    # indices = np.arange(length)
    # np.random.shuffle(indices)

    # test_split = int(test_frac * length)
    # val_split = int(val_frac * length) + test_split
    # test_idx = indices[:test_split]
    # val_idx = indices[test_split:val_split]
    # train_idx = indices[val_split:]

    train_x = [images[i] for i in train_idx]
    train_seg = [segs[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]

    val_x = [images[i] for i in val_idx]
    val_seg = [segs[i] for i in val_idx]
    val_y = [labels[i] for i in val_idx]

    test_x = [images[i] for i in test_idx]
    test_seg = [segs[i] for i in test_idx]
    test_y = [labels[i] for i in test_idx]

    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

    ####################
    ## Define transforms and dataloader
    # Define transforms
    train_transforms = Compose([
        ScaleIntensity(), EnsureChannelFirst(), Resize((78,78,31)),
        RandRotate(range_z = 0.35, prob = 0.8), RandFlip(prob = 0.5)])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((78,78,31))])
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    # Define image dataset, data loader
    check_ds = ImageDataset(image_files=images, seg_files=segs, labels=labels,
                            transform=train_transforms, seg_transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=17, num_workers=2, pin_memory=torch.cuda.is_available())
    # check the data
    im, seg, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, seg.shape, label)

    # create a data loader
    train_ds = ImageDataset(image_files=train_x, seg_files=train_seg, labels=train_y,
                            transform=train_transforms, seg_transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=17, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    val_ds = ImageDataset(image_files=val_x, seg_files=val_seg, labels=val_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=17, num_workers=2, pin_memory=torch.cuda.is_available())

    test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=17, num_workers=2, pin_memory=torch.cuda.is_available())
    
    # check loading
    class_counter = Counter()
    for images, segs, labels in train_loader:
        class_counter.update(labels.tolist())
    print("train", class_counter)
    class_counter = Counter()
    for images, segs, labels in val_loader:
        class_counter.update(labels.tolist())
    print("val", class_counter)
    class_counter = Counter()
    for images, segs, labels in test_loader:
        class_counter.update(labels.tolist())
    print("test", class_counter)
    
    ####################
    ## Create the model
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4 classes -> the out_channels should be 4
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=4).to(device)
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=8, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    auc_metric = ROCAUCMetric()

    ####################
    ## Training
    # start a typical PyTorch training
    max_epochs = 4
    val_interval = 2

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            train_images, train_segs, train_labels = (
                batch_data[0].to(device), 
                batch_data[1].to(device), 
                batch_data[2].to(device)
            )
            masked_train_images = torch.cat((train_images,train_segs), dim=1) # use the mask
            # train_labels = torch.argmax(train_labels, dim=1)
            
            optimizer.zero_grad()
            outputs = model(masked_train_images)
            loss = loss_function(outputs, train_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    val_images, val_segs, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                        val_data[2].to(device)
                    )
                    masked_val_images = torch.cat((val_images,val_segs), dim=1) # use mask
                    y_pred = torch.cat([y_pred, model(masked_val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    
                y_onehot_list = []
                y_pred_act_list = []
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                y_onehot_list.append(y_onehot)
                y_pred_act_list.append(y_pred_act)
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    model_dir = os.path.join(result_dir, attemp+".pth")
                    torch.save(model.state_dict(), model_dir)
                    print("saved new best metric model")
                print(f"Current epoch: {epoch+1} current accuracy: {result:.4f} ")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
                writer.add_scalar("val_accuracy", result, epoch + 1)
        
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    writer.close()

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    fig_dir = os.path.join(result_dir, attemp+"_snap.png")
    plt.savefig(fig_dir)
    
    ####################
    ## Testing
    model.load_state_dict(torch.load(model_dir, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_segs, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2].to(device)
            )
            masked_test_images = torch.cat((test_images,test_segs), dim=1)
            pred = model(masked_test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
       
    ####################
    ## Results       
    print(classification_report(y_true, y_pred, target_names=["HCA","FNH","HCC"], digits=3))
    
    # Creating  a confusion matrix,which compares the y_test and y_pred
    cm = confusion_matrix(y_true, y_pred)
    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    fig_dir = os.path.join(result_dir, attemp+"_confusion.png")
    plt.savefig(fig_dir)

if __name__ == "__main__":
    main()
