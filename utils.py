"""
Additional tools. 
"""
import numpy as np
from sklearn.metrics import accuracy_score

def multilabel_weighted_accuracy(y_true, y_pred):
    classes, weights = np.unique(y_true, return_counts=True)
    weights = weights / np.sum(weights)
    partial_scores = np.array([accuracy_score(y_true == label, 
                                              y_pred == label) 
                               for label in classes])
    return np.sum(partial_scores * weights)