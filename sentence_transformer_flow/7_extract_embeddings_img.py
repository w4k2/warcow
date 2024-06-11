"""
Extract embedding of the posts which contain images. 
"""
from sentence_transformers import SentenceTransformer
import numpy as np

device = "mps"

texts = np.load("data/multimodal_texts.npy")[:, 1]

model = SentenceTransformer('clips/mfaq', device = device)
embeddings = model.encode(texts, device = device, show_progress_bar = True)

print(embeddings.shape)

np.save("data/texts_embeddings_imgs", embeddings)
