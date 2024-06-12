# WarCov dataset

Dataset from popular social media platform containg posts associated with the subject of COVID-19 pandemic and war in the Ukraine in Polish language. Apart from the texts it also contains images attached to posts and proceduraly generated labels. 

## Download and usage

Dataset is available to download on XXX. 

### Additional representations

Upon request on the project's GitHub page, we declare that we will extract features from the original dataset using any indicated method if possible to run with the computational resources of our University. The obtained embeddings will be added to a publicly available data repository in such a case. We also encourage researchers to fork the solution for their projects.

### Tasks

Dataset is dedicated to use in classification task in three main domains:
* text classification (Natural Language Processing), 
* image classification, 
* multimodal classification with both representations. 

More information can be found in the preprint article (submitted to the Thirty-eighth Annual Conference on Neural Information Processing Systems - NeurIPS 2024). Available on [arXiv]().

## Repository content

Repository contains preprocessing flow and complete classification research. In main directory experiments and scripts for analysis purposes are stored. 

* `utils.py` - additional tools used in experiments
* `analyze_class_distribution.py` - visualizations of the labels distribution for hashtagged texts and images
* `analyze_multimodal.py` - calculating metric scores for Late Fusion approach to multimodal classification and visualization of the results
* `dataset_pca.py` – Performs PCA with maximum number of components in order to preprare the dataset for publication.

* `exp_1.py` - classification experiment on all texts in the dataset with usage of two multilabel and five basic classifiers
* `scores_1.py` - based on predictions from `exp_1.py` metrics scores are calculated
* `analyze_1.py` - visualizations of the first experiments results

* `exp_2.py` - classification experiment on texts embeddings for posts which contained images with the classifier which performed the best in first experiment

* `exp_3.py` - classification experiment on preprocessed images using ResNet18
* `analyze_3.py` - visualizations of the third experiments results

* `exp_4.py` - classifiying extracted IMG embeddings using GNB paierd with MultioutputClassifier and ClassifierChain
* `scores_4.py` - metrics calculation based on predictions from `exp_4.py`
* `analyze_4.py` - visualizations of the fourth experiments results

### Sentence transformer flow

Directory `sentence_transformer_flow` contains preprocessing scripts for posts texts and label creation. 

* `1_train_model.py` - loads file `hashtags.npy` which stores NumPy array with all hashtags with their datastamps in the form `['COVID', '2022-01-01T00:58:43.000Z']`; then it uses Sentence Transformer model to create embeddings of hashtags only (without dates); at the end embeddings are normalized and saved as `data/st_embeddings_hashtags.npy`; it additionaly creates file `data/st_dates.npy` for storing dates only.

* `2_pca.py` - uses `data/st_embeddings_hashtags.npy` for analyzing of the explained variance depending on features number and creates representation limited to 80% of components saved as `data/st_hashtags_pca.npy`. 

* `3_concatenate.py` - uses `data/st_dates.npy` to get month, day, hour of post publication and minute; then adds them to `data/st_hashtags_pca.npy`; saves new representation as `data/st_concatenated.npy` .

* `4_clustering.py` - performs KMeans clustering on representation from `data/st_concatenated.npy` to accumulate hashtags into 50 classification labels; predictions are saved as `data/st_cluster_preds.npy`.

* `5_clusters_to_labels` - file `data/st_labels_row.csv` containing all hashtags (one row corresponds to one post) is used to map posts with new labels obtained from clustering; then labels are encoded to the binarized form with MultiLabelBinarizer and saved as `data/st_labels_binarized.npy`.

* `6_posts_embeddings.py` - loads `data/extracted_texts.npy` containing NumPy array with all posts (one element is one text) and creates embeddings with Sentence Transformer model; they are saved as `data/texts_embeddings.npy`.

* `7_extract_embeddings_img.py` - loads `data/multimodal_texts.npy` containing texts of all posts which had also images and creates embeddings with Sentence Transformer model; they are saved as `data/texts_embeddings_imgs.npy`for multimodal representation purposes.

### Image preprocessing flow
Directory `image_preprocessing_flow` contains preprocessing scripts for image data.

* `1_img_to_npy.py` - loads raw image data, pairing each image with its corresponding text and labels. Images are preprocessed using modified ResNet-18 transforms.

* `2_extract_img_embeddings_noft.py` - extracts embeddings from preprocessed image data using ResNet-18 pretrained on ImageNet.

* `3_extract_img_embeddings_ft.py` - extracts embeddings from 80% of preprocessed image data using ResNet-18 pretrained on ImageNet and additionally finetuned using remaining 20% of images. Data was splitted using stratified sampling.

## License information and citing

Dataset on license `Attribution-NonCommercial-ShareAlike 4.0 International`. Additional information in `LICENSE.txt` file. 

If you use dataset in a publication, we would appreciate citations to the following paper: 

```
bibtex
```