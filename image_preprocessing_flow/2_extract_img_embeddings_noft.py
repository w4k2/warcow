"""
Extracting embeddings from IMG using pretrained ResNet18 without finetunig
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torchvision.models.feature_extraction import create_feature_extractor


num_epochs = 20


X = np.load("data/imgs_preprocessed.npy")
X = torch.from_numpy(np.moveaxis(X, 3, 1)).float()

y = np.load("data/imgs_y.npy")

"""
Train-Test-Split
"""
ydot = 2**np.arange(y.shape[1])[::-1][:, None]
z = y @ ydot
z_labels, z_counts = np.unique(z, return_counts=True)
oz_labels = np.copy(z_labels)
z_labels[z_counts <= 2] = -1
z[np.in1d(z, oz_labels[z_labels==-1])] = -1

rskf = RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=42)
extract, train = list(rskf.split(z, z))[0]

X_extract, y_extract = X, y
    
"""
Model
"""
num_classes = 50
batch_size = 8
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
device = torch.device("mps")
model = model.to(device)

"""
Extraction
"""
model.eval()
all_extracted = []
current_sample = 0
batch_size = 500
print("EXTRACION!")
while current_sample < X_extract.shape[0]:
    print("Batch %i:%i" % (current_sample, current_sample+batch_size))
    X_extract_batch = X_extract[current_sample:current_sample+batch_size]
    return_nodes = {
        'flatten': 'extracted_flatten',
    }
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    X_img_batch_extracted = extractor(X_extract_batch.to(device))["extracted_flatten"].cpu().detach().numpy()
    
    all_extracted.append(X_img_batch_extracted)
    current_sample += X_extract_batch.shape[0]
    
all_extracted = np.vstack(tuple(all_extracted))
print("IMG extracted!")
print(all_extracted.shape)
np.save("data/multimodal_img_embeddings_noft", all_extracted)
