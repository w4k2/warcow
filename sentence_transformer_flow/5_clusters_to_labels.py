"""
Convert hashtags to labels according to clusters.
"""
import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import time

# Read data
row_labels_file = "data/st_labels_row.csv"
labels = []

with open(row_labels_file, encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        labels.append(row)

hashtags = np.load("data/hashtags.npy")
cluster_preds = np.load("data/st_cluster_preds.npy")
posts_data = np.load("data/st_posts_data.npy")

# Find the same hashtags by date
st_y = []

def timestamp_from_str(data, fmt="%Y-%m-%dT%H:%M:%S.000Z"):
    output = [time.mktime(time.strptime(v,fmt)) for v in data]
    return np.array(output)

left = hashtags[:,1]
right = posts_data[:,1]

left = timestamp_from_str(left)
right = timestamp_from_str(right)

for labels_idx, label_set in enumerate(tqdm(labels)):
    selected_dates = np.where((left == right[labels_idx]))
    st_y.append(list(cluster_preds[selected_dates]))

# One-hot encoding for labels - "true y" creation and saving     
binarizer = MultiLabelBinarizer()
transformed_labels = binarizer.fit_transform(st_y)

np.save("data/txt_y", transformed_labels)