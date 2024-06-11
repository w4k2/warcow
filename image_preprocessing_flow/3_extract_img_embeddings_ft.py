"""
Extracting embeddings from 80% of IMG using pretrained ResNet18 with finetuning on remaining 20%
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor


num_epochs = 20


X = np.load("data/imgs_preprocessed.npy")
X = torch.from_numpy(np.moveaxis(X, 3, 1)).float()

X_txt = np.load("data/multimodal_txt_noft.npy")

y = np.load("data/multimodal_y_noft.npy")

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

X_train, y_train = X[train], torch.from_numpy(y[train]).long()
X_extract, y_extract = X[extract], y[extract]

np.save("data/multimodal_y_ft", y_extract)
    
X_txt_finetuning = X_txt[extract]
np.save("data/multimodal_txt_ft", X_txt_finetuning)

"""
Model
"""
num_classes = 50
batch_size = 8
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("mps")
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.MultiLabelSoftMarginLoss()
val_criterion = nn.MultiLabelSoftMarginLoss()
    
"""
Train
"""
    
train_dataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in tqdm(range(num_epochs), total=num_epochs):
    for i, batch in enumerate(train_data_loader):
        inputs, labels = batch

        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        
        loss.backward()
        optimizer.step()

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
np.save("data/multimodal_img_embeddings_ft", all_extracted)
