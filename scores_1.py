"""
Calculate metrics for exp_1. 
"""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from utils import multilabel_weighted_accuracy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

# Classifiers and estimators
classifiers = { "GNB": GaussianNB(), 
                "k-NN": KNeighborsClassifier(), 
                "CART": DecisionTreeClassifier(random_state=42), 
                "forest": RandomForestClassifier(random_state=42), 
                "MLP": MLPClassifier(random_state=42) }

mltlab_est = { "MultioutputClassifier": MultiOutputClassifier, 
               "ClassifierChain": ClassifierChain }


# multilabel estimators x classifiers x folds x metrics
scores = np.zeros((len(mltlab_est), len(classifiers), 10, 5))

for est_idx, est_name in enumerate(mltlab_est):
    for clf_idx, clf_name in enumerate(classifiers):
        for fold in range(10):
            y_true = np.load(f"preds/exp_1/test_{fold}_{est_name}_{clf_name}.npy")
            y_pred = np.load(f"preds/exp_1/preds_{fold}_{est_name}_{clf_name}.npy")
            
            scores[est_idx, clf_idx, fold, 0] = f1_score(y_true, y_pred, average="micro")
            scores[est_idx, clf_idx, fold, 1] = f1_score(y_true, y_pred, average="macro")
            scores[est_idx, clf_idx, fold, 2] = f1_score(y_true, y_pred, average="weighted")
            scores[est_idx, clf_idx, fold, 3] = f1_score(y_true, y_pred, average="samples")
            scores[est_idx, clf_idx, fold, 4] = multilabel_weighted_accuracy(y_true, y_pred)
            

    np.save(f"scores/exp_1", scores)
