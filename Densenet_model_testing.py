import os
import re
import ast
import sys
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import (
    EnsureChannelFirst, Compose, Resize)

def parse_args():
    """
        Parses inputs from the commandline.
        :return: inputs as a Namespace object
    """
    parser = argparse.ArgumentParser(description="Write a csv file for checking the shapes and spacing of the images")
    parser.add_argument("-m", "--model_folder", help="The name for the model folder. This arguments is required!",
                        required=True)
    parser.add_argument("-t", "--type", help="D (densenet) or B (baseline). This arguments is required!",
                        required=True)
    parser.add_argument("-log", "--log_file", help="The name of the log file. If none then find .log file in the model_folder", 
                        required=False)

    return parser.parse_args()

####################
## Extract log data
def extract_data(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    text_x, text_seg, text_y = None, None, None

    for i in range(len(lines)):
        if 'TEST' in lines[i]:
            text_x = lines[i + 1].strip()
            text_seg = lines[i + 2].strip()
            text_y = lines[i + 3].strip()
            break
    return text_x, text_seg, text_y

def extract_numbers_from_string(s):
    numbers = re.findall(r'\d+', s)
    return [int(num) for num in numbers]

def find_and_extract_numbers(log_file_path):
    pattern = re.compile("Best (final )?metric for fold (\d+)")
    found_numbers = []

    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # 提取數字部分
                number = match.group(2)
                found_numbers.append(int(number))
    best_metric, final_metric = found_numbers[0], found_numbers[1]

    return best_metric, final_metric

####################
## Empty variables
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
    "metric": []
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

    y_truth and y_prediction should both be lists with the multiclass label of each
    object, e.g.

    y_truth = [0, 0,	0,	0,	0,	0,	2,	2,	1,	1,	2]    ### Groundtruth
    y_prediction = [0, 0,	0,	0,	0,	0,	1,	2,	1,	2,	2]    ### Predicted labels
    y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ] # Normalized score per patient for all labels (three in this example)

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

    return Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, (TP, FP, FN, TN)

####################
## Save data
def append_metrics(dictionary, Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, metric):
    dictionary["Accuracy"].append(Accuracy)
    dictionary["Sensitivity"].append(Sensitivity)
    dictionary["Specificity"].append(Specificity)
    dictionary["Precision"].append(Precision)
    dictionary["NPV"].append(NPV)
    dictionary["F1_score"].append(F1_score)
    dictionary["AUC"].append(AUC)
    dictionary["AverageAccuracy"].append(AverageAccuracy)
    dictionary["metric"].append(metric)
    return dictionary

####################
## Plotting
def plotting(y_truth, y_prediction, y_score, model_name:str, save_path, show=False):
    '''
    Plotting the confusion matrix and ROC plot
    
    Args:
        y_truth and y_prediction should both be lists or tensors with the multiclass label of each object, e.g.
    
        y_truth = [0, 0, 0,	0, 0, 0, 2, 2, 1, 1, 2]         ### Groundtruth
        y_prediction = [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2]    ### Predicted labels
        y_score = [[0.3, 0.3, 0.4], [0.2, 0.6, 0.2], ... ]  ### Normalized score per patient for all labels (three in this example)
        model_name = 'best' or 'final'
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
    
    ####################
    ## Process arguments
    args = parse_args()
    attempt = str(args.model_folder)
    log_file = args.log_file
    m_type = args.type.lower()
    if m_type == "d" or m_type == "densenet":
        m = "densenet"
        channels = 5
    elif m_type == "b" or m_type == "baseline":
        m = "baseline"
        channels = 8
    
    ####################
    ## Set path
    script_dir = os.getcwd()
    model_folder = os.path.join(script_dir, "data", "BLT_radiomics", "models", m, attempt)
    fig_path = os.path.join(model_folder, "fig")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        
    if log_file is not None:
        log_dir = os.path.join(script_dir, "log", log_file)
    elif log_file is None:
        for file in os.listdir(model_folder):
            if file.endswith(".log"):
                log_dir = os.path.join(model_folder, file)
    else:
        raise FileExistsError("The log file is not exists.")
    
    ####################
    ## Fix the testing set values
    s_test_x, s_test_seg, s_test_y = extract_data(log_dir)
    test_x = ast.literal_eval(s_test_x)
    test_seg = ast.literal_eval(s_test_seg)
    if 'r098906' not in script_dir:
        test_x = [path.replace('/data/scratch/r098906/BLT_radiomics/', './data/BLT_radiomics/image_files/') for path in test_x]
        test_seg = [path.replace('/data/scratch/r098906/BLT_radiomics/', './data/BLT_radiomics/image_files/') for path in test_seg]
    num_y = extract_numbers_from_string(s_test_y)
    test_y = torch.tensor(num_y)
    num_class = len(test_y.unique())
    best_metric, final_metric = find_and_extract_numbers(log_dir)
    
    ####################
    ## Load data
    val_transforms = Compose([EnsureChannelFirst(), Resize((78,78,31))])
    test_ds = ImageDataset(image_files=test_x, seg_files=test_seg, labels=test_y, 
                        transform=val_transforms, seg_transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=25, num_workers=2, pin_memory=torch.cuda.is_available())
    
    
    ####################
    ##     Testing    ##
    ####################
    ## Model
    best_metric_model = os.path.join(model_folder, f"model_bestEMA_fold_{best_metric}.pth")
    final_metric_model = os.path.join(model_folder, f"model_finalEMA_fold_{final_metric}.pth")
    if not os.path.exists(best_metric_model) or not os.path.exists(final_metric_model):
        best_metric_model = os.path.join(model_folder, f"model_bestEMA_fold_{best_metric-1}.pth")
        final_metric_model = os.path.join(model_folder, f"model_finalEMA_fold_{final_metric-1}.pth")
    
    ####################
    ## Best metric model
    print(f"Test model:\tmodel_bestEMA_fold_{best_metric}.pth")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=channels, out_channels=num_class, norm='instance').to(device)
    model.load_state_dict(torch.load(best_metric_model, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    y_ts, y_ts_pred, y_ts_pred_argmax = create_empty_predict_lists(device)
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_segs, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2].to(device))
            if channels == 5:
                masked_test_images = torch.cat((test_images,test_segs[:,0:1,:,:]), dim=1)
            elif channels == 8:
                masked_test_images = torch.cat((test_images,test_segs), dim=1)
            
            ts_pred = model(masked_test_images)
            ts_pred_argmax = ts_pred.argmax(dim=1)
            
            # append the predicted values and calculate the metrics
            y_ts = torch.cat([y_ts, test_labels], dim=0)
            y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
            y_ts_pred_argmax = torch.cat([y_ts_pred_argmax, ts_pred_argmax], dim=0)
        
        Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, metric = performance_multiclass(y_ts, y_ts_pred_argmax, y_ts_pred)
        print(f"\nBest metric model evaluate on testing set; AUC: {AUC:4f}")
        print(f"Sensitivity: {Sensitivity:.4f}, Specificity: {Specificity:.4f}")
    
    plotting(y_ts, y_ts_pred_argmax, y_ts_pred, model_name="best", save_path=fig_path)
    
    ####################
    ## Final best model 
    print(f"Test model:\tmodel_finalEMA_fold_{final_metric}.pth")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=channels, out_channels=num_class, norm='instance').to(device)
    model.load_state_dict(torch.load(final_metric_model, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    y_ts, y_ts_pred, y_ts_pred_argmax = create_empty_predict_lists(device)
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_segs, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
                test_data[2].to(device))
            if channels == 5:
                masked_test_images = torch.cat((test_images,test_segs[:,0:1,:,:]), dim=1)
            elif channels == 8:
                masked_test_images = torch.cat((test_images,test_segs), dim=1)
            
            ts_pred = model(masked_test_images)
            ts_pred_argmax = ts_pred.argmax(dim=1)
            
            # append the predicted values and calculate the metrics
            y_ts = torch.cat([y_ts, test_labels], dim=0)
            y_ts_pred = torch.cat([y_ts_pred, ts_pred], dim=0)
            y_ts_pred_argmax = torch.cat([y_ts_pred_argmax, ts_pred_argmax], dim=0)
        
        Accuracy, Sensitivity, Specificity, Precision, NPV, F1_score, AUC, AverageAccuracy, metric = performance_multiclass(y_ts, y_ts_pred_argmax, y_ts_pred)
        print(f"\nFinal model evaluate on testing set; AUC: {AUC:4f}")
        print(f"Sensitivity: {Sensitivity:.4f}, Specificity: {Specificity:.4f}")
        
    plotting(y_ts, y_ts_pred_argmax, y_ts_pred, model_name="final", save_path=fig_path)



if __name__ == "__main__":
    main()