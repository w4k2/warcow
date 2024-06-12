"""
Visulize accuracy results of Experiment 4.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})


classifiers = ["GNB",
               ]
metrics = ["f1 micro", "f1 macro", "f1 weighted", "f1 samples", "MWA"]
est = [ "MultioutputClassifier", "ClassifierChain" ]
    
"""
Embeddings plot
"""
# multilabel estimators x classifiers x folds x metrics
scores_whole = np.load("scores/exp_4.npy")
scores_nofinetuning = np.load("scores/exp_4_noft.npy")
scores_finetuning = np.load("scores/exp_4_ft.npy")

# type x multilabel estimators x classifiers x folds x metrics
scores = np.array([scores_whole, scores_nofinetuning, scores_finetuning])
# type x multilabel estimators x classifiers x metrics
mean_scores = np.mean(scores, axis=3)
std_scores = np.std(scores, axis=3)
# type x metrics
mean_scores = mean_scores[:, 0, 0, [0, 1, 4]]
std_scores = std_scores[:, 0, 0, [0, 1, 4]]

metrics = ["$F_1$ micro", "$F_1$ macro", "MWA"]
colors = ["tomato", "dodgerblue", "limegreen"]
type = [
    "All available images, pre-trained ResNet-18 without finetuning", 
    "80% images (stratified sampling), pre-trained ResNet-18 without finetuning", 
    "80% images (stratified sampling), pre-trained ResNet-18 with finetuning on remaining 20%"]

fig, ax = plt.subplots(1, 1, figsize=(15, 7))

x = np.arange(len(metrics))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

for score_id in range(3):
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(mean_scores, 4)[score_id], width, label=type[score_id], color=colors[score_id])
    ax.bar_label(rects, padding=25)
    
    ax.errorbar(x + offset, np.round(mean_scores, 3)[score_id], std_scores[score_id], fmt='.', color='Black', elinewidth=10,capthick=30,errorevery=1, alpha=None, ms=4, capsize = 2)
    
    multiplier += 1

ax.set_xticks(x + width, metrics)
ax.legend(frameon=False)
ax.grid(ls=":", c=(0.7, 0.7, 0.7))
ax.set_ylabel("score")
ax.set_ylim(0, 0.3)
ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("figures/embeddings.png", dpi=200)
plt.savefig("figures/embeddings.eps", dpi=200)

