import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def get_mean_std(list_w_dicts, metric):
    metric_values = [fold[metric] for fold in list_w_dicts]
    metric_values = list(zip(*metric_values))

    means = [np.mean(epoch) for epoch in metric_values]
    stds = [np.std(epoch) for epoch in metric_values]
    return means, stds

def plot_cv(train_history_cv, val_history_cv, save_path):
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
    
def plot_cm_roc(y_truth, y_prediction, y_score, model_name:str, save_path):
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
    
    class_names = ["Benign","Malignant"]
    
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
    
    # Create ROC plot
    y_truth_binarized = label_binarize(y_truth, classes=[0, 1])
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