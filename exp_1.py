"""
Compare classification accuracy on posts texts with basic classifiers. 
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore') 

X = np.load("data/texts_embeddings.npy")
y = np.load("data/st_labels_binarized.npy")

# Cross-validation
ydot = 2**np.arange(y.shape[1])[::-1][:, None]
z = y @ ydot
z_labels, z_counts = np.unique(z, return_counts=True)
oz_labels = np.copy(z_labels)
z_labels[z_counts <= 2] = -1
z[np.in1d(z, oz_labels[z_labels==-1])] = -1

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)

# Classifiers and estimators
classifiers = { "GNB": GaussianNB(), 
                "k-NN": KNeighborsClassifier(), 
                "CART": DecisionTreeClassifier(random_state=42), 
                "forest": RandomForestClassifier(random_state=42), 
                "MLP": MLPClassifier(random_state=42) }

mltlab_est = { "MultioutputClassifier": MultiOutputClassifier, 
               "ClassifierChain": ClassifierChain }

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

            np.save(f"preds/exp_1/test_{fold}_{est_name}_{clf_name}", y[test])
            np.save(f"preds/exp_1/preds_{fold}_{est_name}_{clf_name}", y_pred)