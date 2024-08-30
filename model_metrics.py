import torch
from torch import tensor
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics_for_all_classes(predictions,target):
    precision_all = precision_score(predictions.cpu(),target.cpu(),average='macro')
    recall_all = recall_score(predictions.cpu(),target.cpu(),average='macro')
    f1_score_all = f1_score(predictions.cpu(),target.cpu(),average='macro')

    return precision_all, recall_all, f1_score_all


def calculate_metrics(output, target):
    """
    Calculate true positives, true negatives, false positives, and false negatives for a multi-class classification.
    """
    _, pred = output.max(1) 
    pred = pred.cpu() 

    true_positives = torch.zeros(output.size(1))
    true_negatives = torch.zeros(output.size(1))
    false_positives = torch.zeros(output.size(1))
    false_negatives = torch.zeros(output.size(1))

    for i in range(len(target)):
        pred_class = pred[i]
        true_class = target[i]
        for class_label in range(output.size(1)):
            if pred_class == class_label and true_class == class_label:
                true_positives[class_label] += 1
            elif pred_class == class_label and true_class != class_label:
                false_positives[class_label] += 1
            elif pred_class != class_label and true_class == class_label:
                false_negatives[class_label] += 1
            else:
                true_negatives[class_label] += 1

    return true_positives, true_negatives, false_positives, false_negatives
    
def accuracy_per_class(output, target):
    # Calculate TP, TN, FP, FN
    tp, tn, fp, fn = calculate_metrics(output, target)

    # Calculate accuracy for each class
    accuracy_per_class = {}
    for class_label in range(output.size(1)):
        tp_class = tp[class_label].item()
        tn_class = tn[class_label].item()
        fp_class = fp[class_label].item()
        fn_class = fn[class_label].item()
        if tp_class + tn_class + fp_class + fn_class == 0:
            accuracy = 0.0
        else:
            accuracy = (tp_class + tn_class) / (tp_class + tn_class + fp_class + fn_class)
        accuracy_per_class[class_label] = accuracy
    
    # Returns a dictionary where keys are class labels and values are the accuracy scores for each class.
    return accuracy_per_class

def specificity_all(output, target):
    # Calculate TN and FP for all classes
    _, tn, fp, _ = calculate_metrics(output, target)

    # Calculate overall specificity
    total_tn = tn.sum().item()
    total_fp = fp.sum().item()
    specificity = total_tn / max((total_tn + total_fp), 1e-10)

    return specificity

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        
        # Returns accuracy in percentage for each value of k
        return res


def precision(output, target):
    # Calculate TP, TN, FP, FN
    tp, _, fp, _ = calculate_metrics(output, target)

    # Calculate precision for all classes
    precision_per_class = {}
    for class_label in range(output.size(1)):
        tp_class = tp[class_label].item()
        fp_class = fp[class_label].item()
        if tp_class + fp_class == 0:
            precision = 0.0
        else:
            precision = tp_class / (tp_class + fp_class)
        precision_per_class[class_label] = precision
    
    # Returns a dictionary where keys are class labels and values are the precision scores for each class.
    return precision_per_class
    
  
def recall(output, target):
    # Calculate TP, TN, FP, FN
    tp, _, _, fn = calculate_metrics(output, target)

    # Calculate recall for all classes
    recall_per_class = {}
    for class_label in range(output.size(1)):
        tp_class = tp[class_label].item()
        fn_class = fn[class_label].item()
        if tp_class + fn_class == 0:
            recall = 0.0
        else:
            recall = tp_class / (tp_class + fn_class)
        recall_per_class[class_label] = recall
    
    # Returns a dictionary where keys are class labels and values are the recall scores for each class.
    return recall_per_class


def f1_score_per_class(output, target):
    # Calculate precision and recall for all classes
    precision_per_class = precision(output, target)
    recall_per_class = recall(output, target)

    # Calculate F1 score for all classes
    f1_score_per_class = {}
    for class_label in range(output.size(1)):
        precision_class = precision_per_class[class_label]
        recall_class = recall_per_class[class_label]
        if precision_class + recall_class == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision_class * recall_class) / (precision_class + recall_class)
        f1_score_per_class[class_label] = f1_score
    
    # Returns a dictionary where keys are class labels and values are the f1 scores for each class.
    return f1_score_per_class
    

def specificity(output, target):
    # Calculate TN and FP for all classes
    _, tn, fp, _ = calculate_metrics(output, target)

    # Calculate specificity for all classes
    specificity_per_class = {}
    for class_label in range(output.size(1)):
        tn_class = tn[class_label].item()
        fp_class = fp[class_label].item()
        if tn_class + fp_class == 0:
            specificity = 0.0
        else:
            specificity = tn_class / (tn_class + fp_class)
        specificity_per_class[class_label] = specificity
    
    # Returns a dictionary where keys are class labels and values are the specificity scores for each class.
    return specificity_per_class


#confusion metric for each class
def confusion_matrix_per_class(output, target):
    # Calculate TP, TN, FP, FN
    tp, tn, fp, fn = calculate_metrics(output, target)

    # Calculate the confusion matrix
    confusion_matrix_per_class = {}
    for class_label in range(output.size(1)):
        tp_class = tp[class_label].item()
        tn_class = tn[class_label].item()
        fp_class = fp[class_label].item()
        fn_class = fn[class_label].item()
        confusion_matrix_per_class[class_label] = [[tp_class, fp_class], [fn_class, tn_class]]
    
    # Returns a dictionary where keys are class labels and values are confusion matrices for each class. Each confusion matrix is represented as a list: [[TP, FP], [FN, TN]].
    return confusion_matrix_per_class


#confusion metric for all classes 
def confusion_matrix(output, target):
    num_classes = output.size(1)

    # Initialize the confusion matrix
    confusion_matrix_overall = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    _, predicted_classes = output.max(1)

    for i in range(target.size(0)):
        predicted_class = predicted_classes[i]
        true_class = target[i]
        confusion_matrix_overall[true_class][predicted_class] += 1
    
    # Returns the overall confusion matrix as a tensor, where rows correspond to true classes and columns correspond to predicted classes.
    return confusion_matrix_overall
