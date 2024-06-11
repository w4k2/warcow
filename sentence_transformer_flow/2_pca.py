"""
Compare variance with components number and perform PCA.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

hashtags = np.load('data/st_embeddings_hashtags.npy')

scaler = StandardScaler()
hashtags_scaled = scaler.fit_transform(hashtags)

# Analyze variance
pca = PCA(random_state=42)
hashtags_pca = pca.fit_transform(hashtags_scaled)

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("figures/pca/variance_full.png")
plt.savefig("figures/pca/variance_full.eps")

# PCA features
pca = PCA(n_components=.8, random_state=42)
hashtags_pca = pca.fit_transform(hashtags_scaled)

print(hashtags_pca.shape)

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("figures/pca/variance_pca.png")
plt.savefig("figures/pca/variance_pca.eps")

np.save("data/st_hashtags_pca.npy", hashtags_pca)