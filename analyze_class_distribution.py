"""
Plot class distribution for hashtagged TXT and IMG
"""
import numpy as np
import matplotlib.pyplot as plt


# full labels
y = np.load("data/txt_y.npy")
# multimodal labels
y_mm = np.load("data/imgs_y.npy")

y_sum = np.sum(y, axis=0)
y_mm_sum = np.sum(y_mm, axis=0)

whole_y = np.array([y_sum, y_mm_sum])

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
colors = ["black", "gray"]
colors = ["tomato", "dodgerblue"]
labels = ["TXT", "IMG"]

x = np.arange(whole_y.shape[1])
width = 0.45
multiplier = 0

for data_type_id in range(whole_y.shape[0]):
    offset = width * multiplier+0.2
    bars = ax.bar(x + offset, whole_y[data_type_id], width, color=colors[data_type_id], log=False, label=labels[data_type_id])
    
    multiplier += 1

ax.set_xticks(x + width, x)
ax.legend(frameon=False)
ax.grid(ls=":", c=(0.7, 0.7, 0.7), axis='y')
ax.set_ylabel("Number od samples")
ax.set_xlabel("Label")
ax.set_ylim(0, 80000)
ax.set_xlim(-.5, 50)
ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("figures/class_distribution.png")
plt.savefig("figures/class_distribution.eps")