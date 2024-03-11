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
    EnsureChannelFirst, Compose, Activations, AsDiscrete,
    Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,ToTensor)
from ema_pytorch import EMA

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
def check_dataoverlap(train_x, val_x, test_x):
    '''
    Args:
        train_x: a list of training data
        val_x: a list of validation data
        test_x: a list of testing data
    Return:
        print if the data has overlapping elements.
    '''
    train_patients = list()
    val_patients = list()
    test_patients = list()
    
    for i in train_x:
        train_patients.append(int(i.split('.')[0].split("_")[-1]))
    for i in val_x:
        val_patients.append(int(i.split('.')[0].split("_")[-1]))
    for i in test_x:
        test_patients.append(int(i.split('.')[0].split("_")[-1]))
    
    overlapped = False
    numbers = list()
    for number in train_patients:
        if number in val_patients or number in test_patients:
            numbers.append(number)
            overlapped = True
    if overlapped:
        print("Data overplapped in different sets.")
        print(f"Overlapped patient IDs: {numbers}")
    else:
        print("No data overplapped in different sets.")
        print(f"Training set: {len(train_patients)}; Validation set: {len(val_patients)}; Test set: {len(test_patients)}")
        
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
## Metrics
def pairwise_auc(y_truth, y_score, class_i, class_j):
    # Filter out the probabilities for class_i and class_j
    y_score = [est[class_i] for ref, est in zip(y_truth, y_score) if ref in (class_i, class_j)]
    y_truth = [ref for ref in y_truth if ref in (class_i, class_j)]

    # Sort the y_truth by the estimated probabilities
    sorted_y_truth = [y for x, y in sorted(zip(y_score, y_truth), key=lambda p: p[0])]

    # Calculated the sum of ranks for class_i
    sum_rank = 0
    for index, element in enumerate(sorted_y_truth):
        if element == class_i:
            sum_rank += index + 1
    sum_rank = float(sum_rank)

    # Get the counts for class_i and class_j
    n_class_i = float(y_truth.count(class_i))
    n_class_j = float(y_truth.count(class_j))

    # If a class in empty, AUC is 0.0
    if n_class_i == 0 or n_class_j == 0:
        return 0.0

    # Calculate the pairwise AUC
    return (sum_rank - (0.5 * n_class_i * (n_class_i + 1))) / (n_class_i * n_class_j)

def multi_class_auc(y_truth, y_score):
    classes = np.unique(y_truth)

    # if any(t == 0.0 for t in np.sum(y_score, axis=1)):
    #     raise ValueError('No AUC is calculated, output probabilities are missing')

    pairwise_auc_list = [0.5 * (pairwise_auc(y_truth, y_score, i, j) +
                                pairwise_auc(y_truth, y_score, j, i)) for i in classes for j in classes if i < j]

    c = len(classes)
    return (2.0 * sum(pairwise_auc_list)) / (c * (c - 1))

def performance_multiclass(y_truth, y_prediction, y_score=None, beta=1):
    '''
    Multiclass performance metrics.

    y_truth and y_prediction should both be lists or tensors with the multiclass label of each object, e.g.

    y_truth = [0, 0, 0,	0, 0, 0, 2, 2, 1, 1, 2]         ### Groundtruth
    y_prediction = [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2]    ### Predicted labels
    y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ]  ### Normalized score per patient for all labels (three in this example)
        
    Calculation of accuracy accorading to formula suggested in CAD Dementia Grand Challege http://caddementia.grand-challenge.org
    and the TADPOLE challenge https://tadpole.grand-challenge.org/Performance_Metrics/
    Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py

    '''
    if y_truth.is_cuda:
        y_truth = y_truth.cpu()
    if y_prediction.is_cuda:
        y_prediction = y_prediction.cpu()
    
    cm = confusion_matrix(y_truth, y_prediction)

    # Determine no. of classes
    labels_class = np.unique(y_truth)
    n_class = len(labels_class)

    # Splits confusion matrix in true and false positives and negatives
    TP = np.zeros(shape=(1, n_class), dtype=int)
    FN = np.zeros(shape=(1, n_class), dtype=int)
    FP = np.zeros(shape=(1, n_class), dtype=int)
    TN = np.zeros(shape=(1, n_class), dtype=int)
    n = np.zeros(shape=(1, n_class), dtype=int)
    for i in range(n_class):
        TP[:, i] = cm[i, i]
        FN[:, i] = np.sum(cm[i, :])-cm[i, i]
        FP[:, i] = np.sum(cm[:, i])-cm[i, i]
        TN[:, i] = np.sum(cm[:])-TP[:, i]-FP[:, i]-FN[:, i]

    n = np.sum(cm)

    # Determine Accuracy
    Accuracy = (np.sum(TP))/n

    # BCA: Balanced Class Accuracy
    BCA = list()
    for i in range(n_class):
        BCAi = 1/2*(TP[:, i]/(TP[:, i] + FN[:, i]) + TN[:, i]/(TN[:, i] + FP[:, i]))
        BCA.append(BCAi)
    AverageAccuracy = np.mean(BCA)

    # Determine total positives and negatives
    P = TP + FN
    N = FP + TN

    # Calculation of sensitivity
    Sensitivity = TP/P
    Sensitivity = np.mean(Sensitivity)

    # Calculation of specifitity
    Specificity = TN/N
    Specificity = np.mean(Specificity)

    # Calculation of precision
    Precision = TP/(TP+FP)
    Precision = np.nan_to_num(Precision)
    Precision = np.mean(Precision)

    # Calculation of NPV
    NPV = TN/(TN+FN)
    NPV = np.nan_to_num(NPV)
    NPV = np.mean(NPV)

    # Calculation  of F1_Score
    F1_score = ((1+(beta**2))*(Sensitivity*Precision))/((beta**2)*(Precision + Sensitivity))
    F1_score = np.nan_to_num(F1_score)
    F1_score = np.mean(F1_score)

    # Calculation of Multi Class AUC according to classpy: https://bitbucket.org/bigr_erasmusmc/classpy/src/master/classpy/multi_class_auc.py
    if y_score is not None:
        if y_score.is_cuda:
            y_score = y_score.cpu()
        softmax = torch.nn.Softmax(dim=1)
        y_score_soft = softmax(y_score)
        AUC = multi_class_auc(y_truth, y_score_soft)
    else:
        AUC = None

    return Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, cm

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

####################
## Plotting
def get_mean_std(list_w_dicts, metric):
    metric_values = [fold[metric] for fold in list_w_dicts]
    metric_values = list(zip(*metric_values))

    means = [np.mean(epoch) for epoch in metric_values]
    stds = [np.std(epoch) for epoch in metric_values]
    return means, stds

def save_progress(train_history_cv, val_history_cv, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    tr_epoch_loss_m, tr_epoch_loss_std = get_mean_std(train_history_cv, metric="epoch_loss")
    v_epoch_loss_m, v_epoch_loss_std = get_mean_std(val_history_cv, metric='epoch_loss')
    tr_epochs = range(1, len(tr_epoch_loss_m) + 1)
    v_epochs = range(1, len(v_epoch_loss_m) + 1)
    axes[0,0].plot(tr_epochs, tr_epoch_loss_m, label=f'training')
    axes[0,0].plot(v_epochs, v_epoch_loss_m, label=f'validation')
    axes[0,0].fill_between(tr_epochs, np.array(tr_epoch_loss_m)-np.array(tr_epoch_loss_std), np.array(tr_epoch_loss_m)+np.array(tr_epoch_loss_std), alpha=0.3)
    axes[0,0].fill_between(v_epochs, np.array(v_epoch_loss_m)-np.array(v_epoch_loss_std), np.array(v_epoch_loss_m)+np.array(v_epoch_loss_std), alpha=0.3)
    axes[0,0].legend()
    axes[0,0].set_title("Epoch Average Loss")
    axes[0,0].set_xlabel("epoch")

    tr_AUC_m, tr_AUC_sd = get_mean_std(train_history_cv, metric="AUC")
    v_AUC_m, v_AUC_sd = get_mean_std(val_history_cv, metric='AUC')
    tr_epochs = range(1, len(tr_AUC_m) + 1)
    v_epochs = range(1, len(v_AUC_m) + 1)
    axes[0,1].plot(tr_epochs, tr_AUC_m, label=f'training')
    axes[0,1].plot(v_epochs, v_AUC_m, label=f'validation')
    axes[0,1].fill_between(tr_epochs, np.array(tr_AUC_m) - np.array(tr_AUC_sd), np.array(tr_AUC_m) + np.array(tr_AUC_sd), alpha=0.3)
    axes[0,1].fill_between(v_epochs, np.array(v_AUC_m) - np.array(v_AUC_sd), np.array(v_AUC_m) + np.array(v_AUC_sd), alpha=0.3)
    axes[0,1].legend()
    axes[0,1].set_title("AUC")
    axes[0,1].set_xlabel("epoch")
    axes[0,1].set_ylim(bottom=0);

    tr_Accuracy_m, tr_Accuracy_sd = get_mean_std(train_history_cv, metric="Accuracy")
    v_Accuracy_m, v_Accuracy_sd = get_mean_std(val_history_cv, metric='Accuracy')
    tr_epochs = range(1, len(tr_Accuracy_m) + 1)
    v_epochs = range(1, len(v_Accuracy_m) + 1)
    axes[0,2].plot(tr_epochs, tr_Accuracy_m, label=f'training')
    axes[0,2].plot(v_epochs, v_Accuracy_m, label=f'validation')
    axes[0,2].fill_between(tr_epochs, np.array(tr_Accuracy_m) - np.array(tr_Accuracy_sd), np.array(tr_Accuracy_m) + np.array(tr_Accuracy_sd), alpha=0.3)
    axes[0,2].fill_between(v_epochs, np.array(v_Accuracy_m) - np.array(v_Accuracy_sd), np.array(v_Accuracy_m) + np.array(v_Accuracy_sd), alpha=0.3)
    axes[0,2].legend()
    axes[0,2].set_title("Accuracy")
    axes[0,2].set_xlabel("epoch")
    axes[0,2].set_ylim(bottom=0, top=1);

    ## row 2
    tr_Sensitivity_m, tr_Sensitivity_sd = get_mean_std(train_history_cv, metric="Sensitivity")
    v_Sensitivity_m, v_Sensitivity_sd = get_mean_std(val_history_cv, metric='Sensitivity')
    tr_epochs = range(1, len(tr_Sensitivity_m) + 1)
    v_epochs = range(1, len(v_Sensitivity_m) + 1)
    axes[1,0].plot(tr_epochs, tr_Sensitivity_m, label=f'training')
    axes[1,0].plot(v_epochs, v_Sensitivity_m, label=f'validation')
    axes[1,0].fill_between(tr_epochs, np.array(tr_Sensitivity_m) - np.array(tr_Sensitivity_sd), np.array(tr_Sensitivity_m) + np.array(tr_Sensitivity_sd), alpha=0.3)
    axes[1,0].fill_between(v_epochs, np.array(v_Sensitivity_m) - np.array(v_Sensitivity_sd), np.array(v_Sensitivity_m) + np.array(v_Sensitivity_sd), alpha=0.3)
    axes[1,0].legend()
    axes[1,0].set_title("Sensitivity")
    axes[1,0].set_xlabel("epoch")
    axes[1,0].set_ylim(bottom=0, top=1);

    tr_Specificity_m, tr_Specificity_sd = get_mean_std(train_history_cv, metric="Specificity")
    v_Specificity_m, v_Specificity_sd = get_mean_std(val_history_cv, metric='Specificity')
    tr_epochs = range(1, len(tr_Specificity_m) + 1)
    v_epochs = range(1, len(v_Specificity_m) + 1)
    axes[1,1].plot(tr_epochs, tr_Specificity_m, label=f'training')
    axes[1,1].plot(v_epochs, v_Specificity_m, label=f'validation')
    axes[1,1].fill_between(tr_epochs, np.array(tr_Specificity_m) - np.array(tr_Specificity_sd), np.array(tr_Specificity_m) + np.array(tr_Specificity_sd), alpha=0.3)
    axes[1,1].fill_between(v_epochs, np.array(v_Specificity_m) - np.array(v_Specificity_sd), np.array(v_Specificity_m) + np.array(v_Specificity_sd), alpha=0.3)
    axes[1,1].legend()
    axes[1,1].set_title("Specificity")
    axes[1,1].set_xlabel("epoch")
    axes[1,1].set_ylim(bottom=0, top=1);

    tr_Precision_m, tr_Precision_sd = get_mean_std(train_history_cv, metric="Precision")
    v_Precision_m, v_Precision_sd = get_mean_std(val_history_cv, metric='Precision')
    tr_epochs = range(1, len(tr_Precision_m) + 1)
    v_epochs = range(1, len(v_Precision_m) + 1)
    axes[1,2].plot(tr_epochs, tr_Precision_m, label=f'training')
    axes[1,2].plot(v_epochs, v_Precision_m, label=f'validation')
    axes[1,2].fill_between(tr_epochs, np.array(tr_Precision_m) - np.array(tr_Precision_sd), np.array(tr_Precision_m) + np.array(tr_Precision_sd), alpha=0.3)
    axes[1,2].fill_between(v_epochs, np.array(v_Precision_m) - np.array(v_Precision_sd), np.array(v_Precision_m) + np.array(v_Precision_sd), alpha=0.3)
    axes[1,2].legend()
    axes[1,2].set_title("Precision")
    axes[1,2].set_xlabel("epoch")
    axes[1,2].set_ylim(bottom=0, top=1);

    ## row 3
    tr_NPV_m, tr_NPV_sd = get_mean_std(train_history_cv, metric="NPV")
    v_NPV_m, v_NPV_sd = get_mean_std(val_history_cv, metric='NPV')
    tr_epochs = range(1, len(tr_NPV_m) + 1)
    v_epochs = range(1, len(v_NPV_m) + 1)
    axes[2,0].plot(tr_epochs, tr_NPV_m, label=f'training')
    axes[2,0].plot(v_epochs, v_NPV_m, label=f'validation')
    axes[2,0].fill_between(tr_epochs, np.array(tr_NPV_m) - np.array(tr_NPV_sd), np.array(tr_NPV_m) + np.array(tr_NPV_sd), alpha=0.3)
    axes[2,0].fill_between(v_epochs, np.array(v_NPV_m) - np.array(v_NPV_sd), np.array(v_NPV_m) + np.array(v_NPV_sd), alpha=0.3)
    axes[2,0].legend()
    axes[2,0].set_title("NPV")
    axes[2,0].set_xlabel("epoch")
    axes[2,0].set_ylim(bottom=0, top=1);

    tr_F1_score_m, tr_F1_score_sd = get_mean_std(train_history_cv, metric="F1_score")
    v_F1_score_m, v_F1_score_sd = get_mean_std(val_history_cv, metric='F1_score')
    tr_epochs = range(1, len(tr_F1_score_m) + 1)
    v_epochs = range(1, len(v_F1_score_m) + 1)
    axes[2,1].plot(tr_epochs, tr_F1_score_m, label=f'training')
    axes[2,1].plot(v_epochs, v_F1_score_m, label=f'validation')
    axes[2,1].fill_between(tr_epochs, np.array(tr_F1_score_m) - np.array(tr_F1_score_sd), np.array(tr_F1_score_m) + np.array(tr_F1_score_sd), alpha=0.3)
    axes[2,1].fill_between(v_epochs, np.array(v_F1_score_m) - np.array(v_F1_score_sd), np.array(v_F1_score_m) + np.array(v_F1_score_sd), alpha=0.3)
    axes[2,1].legend()
    axes[2,1].set_title("F1_score")
    axes[2,1].set_xlabel("epoch")
    axes[2,1].set_ylim(bottom=0, top=1);

    tr_AverageAccuracy_m, tr_AverageAccuracy_sd = get_mean_std(train_history_cv, metric="AverageAccuracy")
    v_AverageAccuracy_m, v_AverageAccuracy_sd = get_mean_std(val_history_cv, metric='AverageAccuracy')
    tr_epochs = range(1, len(tr_AverageAccuracy_m) + 1)
    v_epochs = range(1, len(v_AverageAccuracy_m) + 1)
    axes[2,2].plot(tr_epochs, tr_AverageAccuracy_m, label=f'training')
    axes[2,2].plot(v_epochs, v_AverageAccuracy_m, label=f'validation')
    axes[2,2].fill_between(tr_epochs, np.array(tr_AverageAccuracy_m) - np.array(tr_AverageAccuracy_sd), np.array(tr_AverageAccuracy_m) + np.array(tr_AverageAccuracy_sd), alpha=0.3)
    axes[2,2].fill_between(v_epochs, np.array(v_AverageAccuracy_m) - np.array(v_AverageAccuracy_sd), np.array(v_AverageAccuracy_m) + np.array(v_AverageAccuracy_sd), alpha=0.3)
    axes[2,2].legend()
    axes[2,2].set_title("Average Accuracy")
    axes[2,2].set_xlabel("epoch")
    axes[2,2].set_ylim(bottom=0, top=1);

    save_dir = os.path.join(save_path, "snap_cv.png")
    plt.savefig(save_dir)
    
def plotting(y_truth, y_prediction, y_score, model_name:str, save_path, show=False):
    '''
    Plotting the confusion matrix and ROC plot
    
    Args:
        y_truth and y_prediction should both be lists or tensors with the multiclass label of each object, e.g.
    
        y_truth = [0, 0, 0,	0, 0, 0, 2, 2, 1, 1, 2]         ### Groundtruth
        y_prediction = [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2]    ### Predicted labels
        y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ]  ### Normalized score per patient for all labels (three in this example)
        model_name = (Best metric) or (Final model)
    '''
    # Set model name
    if "best" in model_name:
        model_name = "(Best metric)"
    elif "final" in model_name:
        model_name = "(Final metric)"
    else:
        raise ValueError("Only take 'best' or 'final'.")
    
    class_names = ["HCA","FNH","HCC"]
    
    # Confusion matrix
    cm = pd.DataFrame(confusion_matrix(y_truth, y_prediction), 
                    index=class_names, 
                    columns=class_names)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True)
    plt.title(f"Confusion Matrix {model_name}")
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    
    if model_name == "(Best metric)":
        save_dir = os.path.join(save_path, "cm_best_metric.png")
    elif model_name == "(Final metric)":
        save_dir = os.path.join(save_path, "cm_final_model.png")
    plt.savefig(save_dir)
    print("Confusion matrix is saved!")
    
    if show:
        plt.show()
    
    # Create ROC plot
    y_truth_binarized = label_binarize(y_truth, classes=[0, 1, 2])
    n_classes = y_truth_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_truth_binarized[:, i], [score[i] for score in y_score])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"Multi-class ROC {model_name}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    if model_name == "(Best metric)":
        save_dir = os.path.join(save_path, "ROC_best_metric.png")
    elif model_name == "(Final metric)":
        save_dir = os.path.join(save_path, "ROC_final_model.png")
    plt.savefig(save_dir)
    print("ROC plot is saved!")
    
    if show:
        plt.show()
         
    
################################## MAIN ##################################
def main():
    
    ####################
    ##      Setup     ##
    ####################
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempt = datetime.now().strftime("%Y-%j-%M")
    set_determinism(seed=0)
    
    ####################
    ## Process arguments
    args = parse_args()
    kfold = int(args.kfold)
    random_state = int(args.random_state)
    learning_rate = float(args.learning_rate)
    max_epochs = int(args.max_epochs)

    print("-"*30)
    print("\n## Settings -- BLT_radiomics")
    print(f"Files will be save at: densenet/{attempt}")
    print(f"kfold: {kfold}; random_state: {random_state}")
    print(f"learning rate: {learning_rate}; max_epochs: {max_epochs}")
    
    ####################
    ### Set path, change if needed
    script_dir = os.getcwd()
    model_folder = os.path.join(script_dir, "data", "BLT_radiomics", "models", "densenet")
    model_dir = os.path.join(model_folder, attempt)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if 'r098906' in script_dir:
        GPU_cluster = True
    else:
        GPU_cluster = False
    
    img4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_new_registered", "images")
    seg4D_dir = os.path.join(script_dir, "data", "BLT_radiomics", "image_files", "4D_new_registered", "segs")
    label_dir = os.path.join(script_dir, "data", "BLT_radiomics", "labels_all_phases_NEW.csv")
    if GPU_cluster:
        img4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/images"
        seg4D_dir = "/data/scratch/r098906/BLT_radiomics/4D_new_registered/segs"
        label_dir = os.path.join(script_dir, "data", "labels_all_phases_NEW.csv")
    

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
    labels = pd.read_csv(label_dir)["Pheno"].to_numpy()
    labels[labels == 4] = 2
    labels = torch.from_numpy(labels)
    num_class = len(np.unique(labels, axis=0))
    print(f"\nImage data count: {len(images)}.\nSegmetation data count: {len(segs)}.\nNumber of class: {num_class}.\n\n")

    ####################
    ##   Transforms   ##
    ####################
    train_transforms = Compose([
        EnsureChannelFirst(), Resize((78,78,31)),
        # Data augmentation
        RandRotate(range_z = 0.35, prob = 0.3), RandFlip(prob = 0.5), 
        RandGaussianNoise(std=0.05, prob=0.5),
        RandZoom(prob = 0.3, min_zoom=1.0, max_zoom=1.2), ToTensor()])
    val_transforms = Compose([EnsureChannelFirst(), Resize((78,78,31))])
    # y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    # y_pred_trans = Compose([Activations(softmax=True)])
    

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
        check_dataoverlap(train_x, val_x, test_x)
        
        ####################
        ## Create dataloader
        train_ds = ImageDataset(image_files=train_x, seg_files=train_seg, labels=train_y,
                            transform=train_transforms, seg_transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
        val_ds = ImageDataset(image_files=val_x, seg_files=val_seg, labels=val_y, 
                            transform=val_transforms, seg_transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
        test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                            transform=val_transforms, seg_transform=val_transforms)
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
            Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, cm = performance_multiclass(y, y_pred_argmax, y_pred)
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
                    vAccuracy, vSensitivity, vSpecificity, vPrecision, vNPV, vF1_score, vAUC, vAverageAccuracy, vcm = performance_multiclass(y_val, y_val_pred_argmax, y_val_pred)
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
            tsbmAccuracy, tsbmSensitivity, tsbmSpecificity, tsbmPrecision, tsbmNPV, tsbmF1_score, tsbmAUC, tsbmAverageAccuracy, tsbmcm = performance_multiclass(y_ts, y_ts_bm_pred_argmax, y_ts_bm_pred)
            tsfmAccuracy, tsfmSensitivity, tsfmSpecificity, tsfmPrecision, tsfmNPV, tsfmF1_score, tsfmAUC, tsfmAverageAccuracy, tsfmcm = performance_multiclass(y_ts, y_ts_fm_pred_argmax, y_ts_fm_pred)
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
        save_progress(train_history_cv, val_history_cv, model_dir)
                  
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
        
    plotting(ALL_y_ts, ALL_y_ts_bm_pred_argmax, ALL_y_ts_bm_pred, model_name="best", save_path=fig_path)
    plotting(ALL_y_ts, ALL_y_ts_fm_pred_argmax, ALL_y_ts_fm_pred, model_name="final", save_path=fig_path)
    
    ####################
    ##    Testing     ##
    ####################
    
    # print("-" * 35)
    # i_best = np.argmax(metrics_best) + 1
    # i_final =  np.argmax(metrics_final) + 1
    # print('Best metric for fold', i_best+1)
    # print('Best final metric for fold', i_final+1)
    
    # # Set path the store the figures
    # fig_path = os.path.join(model_dir, "fig")
    # if not os.path.exists(fig_path):
    #     os.makedirs(fig_path)
    # best_metric_model = f"model_bestEMA_fold_{i_best}.pth"
    # final_metric_model = f"model_finalEMA_fold_{i_final}.pth"
    
    # ### Best metric model ###
    # print(f"Test model:\tmodel_bestEMA_fold_{best_metric_model}")
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_class, norm='instance').to(device)
    # model.load_state_dict(torch.load(os.path.join(model_dir, best_metric_model), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    # model.eval()
    # y_ts, y_ts_pred, y_ts_pred_argmax = create_empty_predict_lists(device)
    # with torch.no_grad():
    #     for test_data in test_loader:
    #         test_images, test_segs, test_labels = (
    #             test_data[0].to(device),
    #             test_data[1].to(device),
    #             test_data[2].to(device))
    #         masked_test_images = torch.cat((test_images,test_segs[:,0:1,:,:]), dim=1)
            
    #         ts_pred = model(masked_test_images)
    #         ts_pred_argmax = ts_pred.argmax(dim=1)
            
    #         # append the predicted values and calculate the metrics
    #         y_ts = torch.cat([y_ts, test_labels], dim=0)
    #         y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
    #         y_ts_pred_argmax = torch.cat([y_ts_pred_argmax, ts_pred_argmax], dim=0)
        
    #     Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, metric = performance_multiclass(y_ts, y_ts_pred_argmax, y_ts_pred)
    #     print(f"\nBest metric model evaluate on testing set; AUC: {AUC:4f}")
    #     print(f"Sensitivity: {Sensitivity:.4f}, Specificity: {Specificity:.4f}")
    
    # plotting(y_ts, y_ts_pred_argmax, y_ts_pred, model_name="best", save_path=fig_path)
    
    # ### Final best model ###
    # print(f"Test model:\tmodel_finalEMA_fold_{final_metric_model}")
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_class, norm='instance').to(device)
    # model.load_state_dict(torch.load(os.path.join(model_dir, final_metric_model), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    # model.eval()
    # y_ts, y_ts_pred, y_ts_pred_argmax = create_empty_predict_lists(device)
    # with torch.no_grad():
    #     for test_data in test_loader:
    #         test_images, test_segs, test_labels = (
    #             test_data[0].to(device),
    #             test_data[1].to(device),
    #             test_data[2].to(device))
    #         masked_test_images = torch.cat((test_images,test_segs[:,0:1,:,:]), dim=1)
            
    #         ts_pred = model(masked_test_images)
    #         ts_pred_argmax = ts_pred.argmax(dim=1)
            
    #         # append the predicted values and calculate the metrics
    #         y_ts = torch.cat([y_ts, test_labels], dim=0)
    #         y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
    #         y_ts_pred_argmax = torch.cat([y_ts_pred_argmax, ts_pred_argmax], dim=0)
        
    #     Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, metric = performance_multiclass(y_ts, y_ts_pred_argmax, y_ts_pred)
    #     print(f"\nFinal model evaluate on testing set; AUC: {AUC:4f}")
    #     print(f"Sensitivity: {Sensitivity:.4f}, Specificity: {Specificity:.4f}")
        
    # plotting(y_ts, y_ts_pred_argmax, y_ts_pred, model_name="final", save_path=fig_path)
    

if __name__ == "__main__":
    main()
