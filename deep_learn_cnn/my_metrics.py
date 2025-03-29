from sklearn.metrics import confusion_matrix
import numpy as np

def binary_accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

def binary_precision(y_pred, y_true):
    true_positives  = ((y_pred == 1) & (y_true == 1)).sum()
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    return true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0

def binary_recall(y_pred, y_true):
    true_positives  = ((y_pred == 1) & (y_true == 1)).sum()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
    return true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

def binary_f1(y_pred, y_true):
    precision = binary_precision(y_pred, y_true)
    recall    = binary_recall(y_pred, y_true)
    return 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0

def binary_conf_matrix(y_pred, y_true):
    return confusion_matrix(y_true, y_pred)

def multiclass_accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

def multiclass_precision(y_pred, y_true):
    classes = np.unique(y_true)
    precisions = []
    for cls in classes:
        true_positives  = ((y_pred == cls) & (y_true == cls)).sum()
        false_positives = ((y_pred == cls) & (y_true != cls)).sum()
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        precisions.append(precision)
    return precisions

def multiclass_recall(y_pred, y_true):
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        true_positives  = ((y_pred == cls) & (y_true == cls)).sum()
        false_negatives = ((y_pred != cls) & (y_true == cls)).sum()
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        recalls.append(recall)
    return recalls

def multiclass_f1(y_pred, y_true):
    precisions = multiclass_precision(y_pred, y_true)
    recalls = multiclass_recall(y_pred, y_true)
    return [2 * (p * r) / (p + r) if p + r > 0 else 0 for p, r in zip(precisions, recalls)]

def multiclass_conf_matrix(y_pred, y_true):
    return confusion_matrix(y_true, y_pred)

def print_conf_matrix(conf_matrix):
    print("Confusion Matrix:")
    print(conf_matrix)
