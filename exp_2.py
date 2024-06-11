"""
Get prediction for posts which contain images with the best classifier from exp_1. 
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import ClassifierChain
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore') 

X = np.load("data/texts_embeddings_imgs.npy")
y = np.load("data/imgs_y.npy")

# Cross-validation
ydot = 2**np.arange(y.shape[1])[::-1][:, None]
z = y @ ydot
z_labels, z_counts = np.unique(z, return_counts=True)
oz_labels = np.copy(z_labels)
z_labels[z_counts <= 2] = -1
z[np.in1d(z, oz_labels[z_labels==-1])] = -1

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)

for fold, (train, test) in enumerate(rskf.split(z, z)):
    pca = PCA(n_components=.95, random_state=42)
    X_train = pca.fit_transform(X[train])
    X_test = pca.transform(X[test])
    print(f"FOLD {fold}")
    
    clf = ClassifierChain(MLPClassifier(random_state=42))
    clf.fit(X_train, y[train])
    y_pred = clf.predict_proba(X_test)

    np.save(f"preds/exp_2/test_{fold}", y[test])
    np.save(f"preds/exp_2/preds_{fold}", y_pred)
