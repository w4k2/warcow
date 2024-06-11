"""
Plot results for experiments on multimodal data
"""
import numpy as np
from sklearn.metrics import f1_score
from utils import multilabel_weighted_accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


n_folds = 10
# IMG|TXT|LF x METRICS x FOLDS
scores = np.zeros((3, 3, 10))

metrics = ["$F_1$ micro", "$F_1$ macro", "MWA"]
colors = ["tomato", "dodgerblue", "limegreen"]
modalities = ["IMG", "LF", "TXT"]

# IMG losses
losses = np.load("preds_img/epochs_loss_r18.npy")

for fold_id in tqdm(range(n_folds)):
    txt_probas = np.load("preds/exp_2/preds_%i.npy" % fold_id)
    txt_y = np.load("preds/exp_2/test_%i.npy" % fold_id)
    
    img_probas = np.load("preds_img/fold_%i_probas_r18.npy" % fold_id)
    img_y = np.load("preds_img/fold_%i_test_r18.npy" % fold_id)
    
    if not np.array_equal(txt_y, img_y):
        print("Wrong labels!")
    
    img_preds = (img_probas > .5).astype(int)
    txt_preds = (txt_probas > .5).astype(int)
    lf_preds = ((img_probas+txt_probas)/2 > .5).astype(int)
    
    all_preds = [img_preds, lf_preds, txt_preds]
    for preds_id, preds in enumerate(all_preds):
        f1_micro = f1_score(img_y, preds, average='micro')
        f1_macro = f1_score(img_y, preds, average='macro')
        mwa = multilabel_weighted_accuracy(img_y, preds)
        
        
        scores[preds_id, 0, fold_id] = f1_micro
        scores[preds_id, 1, fold_id] = f1_macro
        scores[preds_id, 2, fold_id] = mwa

fig, ax = plt.subplots(1, 1, figsize=(15, 6))

mean_scores= np.mean(scores, axis=2)
std_scores= np.std(scores, axis=2)

x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for score_id in range(3):
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(mean_scores, 3)[score_id], width, label=modalities[score_id], color=colors[score_id])
    ax.bar_label(rects, padding=25)
    
    ax.errorbar(x + offset, np.round(mean_scores, 3)[score_id], std_scores[score_id], fmt='.', color='Black', elinewidth=10,capthick=30,errorevery=1, alpha=None, ms=4, capsize = 2)
    
    multiplier += 1

ax.set_xticks(x + width, metrics)
ax.legend(frameon=False)
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.set_ylabel("score")
ax.set_ylim(0, 0.5)
ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("figures/multimodal.png", dpi=200)
plt.savefig("figures/multimodal.eps", dpi=200)
