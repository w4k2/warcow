"""
Visulize accuracy on classifiers with f1 and MWA metrics.
"""
import numpy as np
import matplotlib.pyplot as plt

classifiers = ["GNB","k-NN","CART","RF","MLP"]
metrics = ["f1 micro", "f1 macro", "f1 weighted", "f1 samples", "MWA"]
est = [ "MultioutputClassifier", "ClassifierChain" ]

# multilabel estimators x classifiers x folds x metrics
scores = np.load("scores/exp_1.npy")

scores_mean = np.mean(scores, axis=2)
print(scores_mean.shape)


for midx, mname in enumerate(metrics):
    fig, ax = plt.subplots(1,1,figsize=(8,8*0.618))

    for i in range(2):
        ax.bar(np.arange(5)-.125+.25*i%2, 
                scores_mean[i, :, midx],
                width=.2,
                label=est[i],
                color='red' if i == 0 else 'black')

    ax.set_xticks(np.arange(5), classifiers)
    ax.set_ylim([0, 0.35])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(mname)
    ax.legend(loc=2, frameon=False)

    ax.grid(ls=":")

    plt.savefig("foo.png")
    plt.savefig("figures/exp_1/%s.png" % mname)
    plt.savefig("figures/exp_1/%s.eps" % mname)
    plt.close()