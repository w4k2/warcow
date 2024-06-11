"""
Transform texts to embeddings.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

device = "cuda"
device = 'mps'

texts = np.load("data/extracted_texts.npy")

model = SentenceTransformer('clips/mfaq', device = device)
embeddings = model.encode(texts, device = device, show_progress_bar = True)

print(embeddings.shape)

np.save("data/texts_embeddings", embeddings)