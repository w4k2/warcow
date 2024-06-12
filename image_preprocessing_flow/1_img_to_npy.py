"""
Apply tranforms to images and save them to npy file. Save corresponding labels.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import from_numpy
from tqdm import tqdm
from torchvision.transforms import v2
import torch


path = "data/path_to_img_data"
files = os.listdir(path)
files.sort()
files = files[:-2]

id_table = np.load("../data/st_posts_data.npy")
post_data = np.load("../path_to_post_data.npy")
y = np.load("../data/st_labels_binarized.npy")
txt = np.load("../data/extracted_texts.npy")

img_y = []
imgs_resized = []
multimodal_texts = []

# For each downloaded image
for file in tqdm(files, total=len(files)):
    # Get img id
    media_id = file.split("__")[0]
    # Look for corresponding post id
    indxs = np.argwhere(id_table[:, 1] == media_id)
    
    for indx in indxs:
        indx = indx[0]
    
        post_id = id_table[indx, 0]
        # Look for index of a given post in post data
        try:
            data_indx = np.argwhere(post_data[:, 0] == post_id)[0][0]
            
            # Load img
            img = plt.imread(path+file)
            if len(img.shape) == 2:
                img = np.stack((img, img, img), axis=2)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            
            transforms = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = np.swapaxes(transforms(from_numpy(np.swapaxes(np.array(img), 0, 2))).numpy(), 0, 2)
            
            imgs_resized.append(img)
            
            # Get y for this image
            img_y.append(y[data_indx])
            
            # Get multimodal texts [post id, txt]
            multimodal_texts.append([post_id, txt[data_indx]])
            
        except:
            pass
    
imgs_resized = np.array(imgs_resized)
img_y = np.array(img_y)
multimodal_texts = np.array(multimodal_texts)
print(imgs_resized.shape)

np.save("../data/imgs_preprocessed.npy", imgs_resized)
np.save("../data/imgs_y.npy", img_y)
np.save("../data/multimodal_texts.npy", multimodal_texts)