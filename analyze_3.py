"""
Visualize results of Experiment 3
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils import multilabel_weighted_accuracy
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

n_folds = 10
losses = np.load("preds_img/epochs_loss_r18.npy")

metrics = ["$F_1$ micro", "$F_1$ macro", "MWA"]
colors = ["tomato", "dodgerblue", "limegreen"]

fig, ax = plt.subplots(1, 2, figsize=(20, 8))

scores = np.zeros((3, 10))

# for fold, (train, test) in enumerate(rskf.split(z, z)):
for fold_id in range(n_folds):
    
    img_probas = np.load("preds_img/fold_%i_probas_r18.npy" % fold_id)
    img_y = np.load("preds_img/fold_%i_test_r18.npy" % fold_id)
    
    preds = (img_probas > .5).astype(int)
    
    f1_micro = f1_score(img_y, preds, average='micro')
    f1_macro = f1_score(img_y, preds, average='macro')
    mwa = multilabel_weighted_accuracy(img_y, preds)
    
    scores[0, fold_id] = f1_micro
    scores[1, fold_id] = f1_macro
    scores[2, fold_id] = mwa

mean_scores= np.mean(scores, axis=1)
std_scores= np.std(scores, axis=1)
rects = ax[0].bar(metrics, np.round(mean_scores, 3), color=colors)
ax[0].bar_label(rects, padding=25)
ax[0].errorbar(metrics, mean_scores, std_scores, fmt='.', color='Black', elinewidth=10,capthick=30,errorevery=1, alpha=None, ms=4, capsize=2)
ax[0].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[0].set_ylabel("score")
ax[0].set_ylim(0, 0.3)
ax[0].set_title("Metric values for IMG modality")
ax[0].spines[['right', 'top']].set_visible(False)

losses_tr = np.mean(losses[0], axis=0)
losses_std_tr = np.std(losses[0], axis=0)
losses_val = np.mean(losses[1], axis=0)
losses_std_val = np.std(losses[1], axis=0)
ax[1].plot([i+1 for i in range(40)], losses_tr, lw=1, c="tomato", label="training")
ax[1].plot([i+1 for i in range(40)], losses_val, lw=1, c="dodgerblue", label="test")

ax[1].fill_between(
           [i+1 for i in range(40)],
            losses_tr +losses_std_tr,
            losses_tr -losses_std_tr,
            alpha=0.2,
            color="tomato"
        )

ax[1].fill_between(
           [i+1 for i in range(40)],
            losses_val +losses_std_val,
            losses_val -losses_std_val,
            alpha=0.2,
            color="dodgerblue"
        )

ax[1].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("MultiLabel Soft Margin Loss")
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylim(0.12, .22)
ax[1].set_xlim(1, 40)
ax[1].set_title("Training and validation loss for IMG modality")
ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig("figures/img.png", dpi=200)
plt.savefig("figures/img.eps")

