"""
Classifiying extracted IMG embeddings using GNB paierd with MultioutputClassifier and ClassifierChain
"""

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore') 

# ft, noft
data = "noft"

X = np.load("data/multimodal_img_embeddings_%s.npy" % data)
y = np.load("data/imgs_y.npy")

# Cross-validation
ydot = 2**np.arange(y.shape[1])[::-1][:, None]
z = y @ ydot
z_labels, z_counts = np.unique(z, return_counts=True)
oz_labels = np.copy(z_labels)
z_labels[z_counts <= 2] = -1
z[np.in1d(z, oz_labels[z_labels==-1])] = -1

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)

classifiers = { 
                "GNB": GaussianNB(), 
                }

mltlab_est = { 
                "MultioutputClassifier": MultiOutputClassifier, 
                "ClassifierChain": ClassifierChain 
               }

for fold, (train, test) in enumerate(rskf.split(z, z)):
    pca = PCA(n_components=.95, random_state=42)
    X_train = pca.fit_transform(X[train])
    X_test = pca.transform(X[test])
    print(f"FOLD {fold}")
    
    for est_name in tqdm(mltlab_est):
        for clf_name in tqdm(classifiers):
            est = mltlab_est[est_name](classifiers[clf_name])
            est.fit(X_train, y[train])
            y_pred = est.predict(X_test)

            np.save(f"preds/exp_4/test_{data}_{fold}_{est_name}_{clf_name}", y[test])
            np.save(f"preds/exp_4/preds__{data}_{fold}_{est_name}_{clf_name}", y_pred)