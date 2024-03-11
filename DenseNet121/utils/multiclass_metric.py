import numpy as np
import torch
from sklearn.metrics import confusion_matrix

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

    # Calculation of F1_Score
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