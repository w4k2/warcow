"""
Classify preprocessed images using ResNet18, 5x2 CV
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


num_epochs = 40

X = np.load("data/imgs_preprocessed.npy")
X = torch.from_numpy(np.moveaxis(X, 3, 1)).float()
y = np.load("data/imgs_y.npy")


# Cross-validation
ydot = 2**np.arange(y.shape[1])[::-1][:, None]
z = y @ ydot
z_labels, z_counts = np.unique(z, return_counts=True)
oz_labels = np.copy(z_labels)
z_labels[z_counts <= 2] = -1
z[np.in1d(z, oz_labels[z_labels==-1])] = -1

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)

# TRAIN/TEST x FOLDS x EPOCHS
epochs_loss = np.zeros((2, 10, num_epochs))

for fold, (train, test) in enumerate(rskf.split(z, z)):
    print("FOLD: %i" % fold)
    X_train, y_train = X[train], torch.from_numpy(y[train]).long()
    X_test, y_test = X[test], y[test]
    
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
    """
    
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_test, torch.from_numpy(y[test]).long())
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        batches_loss = []
        for i, batch in enumerate(data_loader):
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs.to(device), labels.to(device))
            
            batches_loss.append(loss.cpu().detach().numpy())
            
            loss.backward()
            optimizer.step()
            
        epoch_loss = np.mean(np.array(batches_loss).ravel())
        # TRAIN LOSS
        epochs_loss[0, fold, epoch] = epoch_loss
        
        # VAL LOSS
        val_batches_loss = []
        for i, batch in enumerate(val_data_loader):
            val_inputs, val_labels = batch
            val_outputs = model(val_inputs.to(device))
            val_loss = val_criterion(val_outputs.to(device), val_labels.to(device))
            val_batches_loss.append(val_loss.cpu().detach().numpy())
            
        val_epoch_loss = np.mean(np.array(val_batches_loss).ravel())
        # TRAIN LOSS
        epochs_loss[1, fold, epoch] = val_epoch_loss
    
    model.eval()
    
    all_predicted = []
    all_probas = []
    current_sample = 0
    batch_size = 500
    
    while current_sample < X_test.shape[0]:
        X_test_pred = X_test[current_sample:current_sample+batch_size]
        
        logits = model(X_test_pred.to(device))
        probas = torch.sigmoid(logits)
        all_probas.append(probas.cpu().detach().numpy())
        preds = (probas.cpu().detach().numpy() > .5).astype(int)
        all_predicted.append(preds)
        
        current_sample += X_test_pred.shape[0]
    
    all_predicted = np.concatenate((all_predicted), axis=0)
    all_probas = np.concatenate((all_probas), axis=0)

    np.save("preds_img/fold_%i_preds_r18" % fold, all_predicted)
    np.save("preds_img/fold_%i_probas_r18" % fold, all_probas)
    np.save("preds_img/epochs_loss_r18", epochs_loss)
    np.save("preds_img/fold_%i_test_r18" % fold, y_test)
    
    X_train, y_train = None, None
    X_test, y_test = None, None
    dataset = None
    data_loader = None
    model = None
    all_predicted = None
    all_probas = None
    torch.mps.empty_cache()