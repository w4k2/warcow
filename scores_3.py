"""
Calculate metrics for exp_3. 
"""
import numpy as np
from sklearn.metrics import f1_score
from utils import multilabel_weighted_accuracy
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

# noft, ft
data = "nofinetuning"

# Classifiers and estimators
classifiers = { "GNB": GaussianNB(), 
                }

mltlab_est = { "MultioutputClassifier": MultiOutputClassifier, 
               "ClassifierChain": ClassifierChain }


# multilabel estimators x classifiers x folds x metrics
scores = np.zeros((len(mltlab_est), len(classifiers), 10, 5))

for est_idx, est_name in enumerate(mltlab_est):
    for clf_idx, clf_name in enumerate(classifiers):
        for fold in range(10):
            y_true = np.load(f"preds/exp_3/test_{data}_{fold}_{est_name}_{clf_name}.npy")
            y_pred = np.load(f"preds/exp_3/preds__{data}_{fold}_{est_name}_{clf_name}.npy")
            
            scores[est_idx, clf_idx, fold, 0] = f1_score(y_true, y_pred, average="micro")
            scores[est_idx, clf_idx, fold, 1] = f1_score(y_true, y_pred, average="macro")
            scores[est_idx, clf_idx, fold, 2] = f1_score(y_true, y_pred, average="weighted")
            scores[est_idx, clf_idx, fold, 3] = f1_score(y_true, y_pred, average="samples")
            scores[est_idx, clf_idx, fold, 4] = multilabel_weighted_accuracy(y_true, y_pred)
            
    np.save(f"scores/exp_3_%s" % data, scores)
