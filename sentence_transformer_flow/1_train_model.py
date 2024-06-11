'''
Use Sentence Transformer to create embeddings and cluster hashtags with dates. 
'''
from sentence_transformers import SentenceTransformer
import numpy as np

print('# Loading hashtags')
hashtags = np.load('data/hashtags.npy')

print('# Initializing model')
model = SentenceTransformer('clips/mfaq')

print('# Encoding embeddings')
embeddings = model.encode(hashtags[:, 0])

print('# Normalization')
ebd_std = np.std(embeddings, axis=0)
ebd_mean = np.mean(embeddings, axis=0)

embeddings = embeddings - ebd_mean[None, :]
embeddings = embeddings / ebd_std[None, :]

print('# Saving')
np.save("data/st_embeddings_hashtags", embeddings)
np.save("data/st_dates", hashtags[:, 1])
