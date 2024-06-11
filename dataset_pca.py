"""
PCA with all components on published dataset
"""

import numpy as np
from sklearn.decomposition import PCA


pca = PCA(random_state=41)

# Full TXT embeddings
print("Full TXT embeddings")
full_txt = np.load("data/texts_embeddings.npy")
full_txt = pca.fit_transform(full_txt)
np.save("data/full_txt_embeddings_pca", full_txt)

# Multimodal TXT embeddings
print("Multimodal TXT embeddings")
noft_txt = np.load("data/multimodal_txt_noft.npy")
noft_txt = pca.fit_transform(noft_txt)
np.save("data/multimodal_txt_noft_pca", noft_txt)

# Multimodal TXT emneddings for 80% of IMG
print("Multimodal TXT emneddings for 80% of IMG")
ft_txt = np.load("data/multimodal_txt_ft.npy")
ft_txt = pca.fit_transform(ft_txt)
np.save("data/multimodal_txt_ft_pca", ft_txt)

# Full IMG embeddings
print("Full IMG embeddings")
noft_img = np.load("data/multimodal_img_noft.npy")
noft_img = pca.fit_transform(noft_img)
np.save("data/multimodal_img_noft_pca", noft_img)

# 80% IMG embeddings after finetuning
print("80% IMG embeddings after finetuning")
ft_img = np.load("data/multimodal_img_ft.npy")
ft_img = pca.fit_transform(ft_img)
np.save("data/multimodal_img_ft_pca", ft_img)