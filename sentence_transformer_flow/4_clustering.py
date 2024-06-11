"""
Perform hashtags clustering.
"""
from sklearn.cluster import KMeans
import numpy as np

n_clusters = 50 

data = np.load("data/st_concatenated.npy")

kmeans = KMeans(n_clusters = n_clusters, random_state=42,
                max_iter=300, verbose=1, init='k-means++',
                n_init='auto')

kmeans.fit(data)
preds = kmeans.predict(data)
clusters, counts = np.unique(preds, return_counts=True)

print(counts)

np.save("data/st_cluster_preds", preds)
