DSS5104 team19 assignment 2

#1.overall description
This project builds a complete transfer learning pipeline for PathMNIST image classification. It begins with data loading, preprocessing, and visual inspection to verify the classes, input format, and overall data characteristics. In the model selection stage, three backbone architectures—ResNet-50, EfficientNet-B0, and Swin-T—are compared under settings including feature extraction, full fine-tuning, and data augmentation. Based on these results, the best-performing model was further refined through additional exploration of data augmentation, in order to examine more how this affects generalization performance. The final selected architecture is then used for a data efficiency experiment, comparing pretrained initialization and training from scratch at 100%, 50%, 25%, 10%, and 5% of the training data to examine the value of transfer learning under limited-data conditions. Finally, the best model is analyzed through confusion matrix analysis, per-class metrics, and representative misclassified examples, forming a complete workflow from model comparison to detailed evaluation.

#2.basic setup:
dataset: PathMNIST  
pretrained architectures: ResNet-50, EfficientNet-B0, Swin Transformer(Tiny)  
strategies: feature extraction, full fine-tuning, data augmentation  

#3.Main Scripts and explanations

##3.1 ****model selection+strategy comparison****:

###3.1.1 ****'RESNET50.ipynb'****
RESNET50.ipynb is the initial baseline experiment file of this project and serves as the starting point of the overall workflow. It covers PathMNIST data loading, sample visualization, preprocessing, and DataLoader construction, while also verifying the dataset classes and input format. Based on this setup, the notebook implements feature extraction, data augmentation, and full fine-tuning using an ImageNet-pretrained ResNet-50. It saves three model checkpoint files: resnet50_fe_no_aug.pth, resnet50_fe_aug_mlphead.pth, and resnet50_full_finetuning.pth, which provide the foundation for later model comparison, strategy refinement, and data efficiency experiments.  
(output pth: resnet50_fe_no_aug.pth; resnet50_fe_aug_mlphead.pth; resnet50_full_finetuning.pth)

###3.1.2 ****B0.ipynb****
B0.ipynb extends the experimental pipeline developed for ResNet-50 to EfficientNet-B0, serving as the second CNN architecture for comparison. It includes the same PathMNIST preprocessing and DataLoader setup, and implements all the 3 strategies mentioned above.  
(output pth: B0_fe_no_aug.pth; B0_fe_aug_mlphead.pth; B0_full_finetuning.pth)

###3.1.3 ****swin_head.py/swin_head_aug.py/swin_full.py****
swin_head.py, swin_head_aug.py, and swin_full.py correspond to the three Swin Transformer (Tiny) settings used in this project: feature extraction, feature extraction with data augmentation, and full fine-tuning. They extend the overall framework from the earlier CNNs to a Transformer, while also adopting AMP mixed precision to improve training efficiency. The results are shown in the 'results for Swin' folder.  
(output pth: swin_feature_extraction_amp.pth; swin_fe_aug_amp.pth; swin_full_finetuning_amp.pth)

###3.1.4 ****inference latency.py****
It loads 3 trained .pth files (full fine-tuning), rebuilds the corresponding ResNet-50, EfficientNet-B0, and Swin-T models, and tests their inference speed on the PathMNIST validation set, reporting latency and throughput. The result screenshot is shown in the folder as .png.  
(required pth: resnet50_full_finetuning.pth; B0_full_finetuning.pth; swin_full_finetuning_amp.pth)

##3.2 ****further refinement****:

###3.2.1 ****resnet_amp.py/resnet_aug_amp.py****
resnet_amp.py and resnet_amp_aug.py are refinement scripts built on the selected ResNet-50. Since earlier model selection showed that augmentation did not perform well, these scripts extend training from 4 to 8 epochs and reduce augmentation strength to re-examine its effect under a longer and milder setting, with some basic hyperparameters adjusted accordingly. They correspond to full fine-tuning without augmentation and full fine-tuning with weaker augmentation, and also add AMP and early stopping to the original ResNet-50 workflow. The results are shown in the 'curves+results' folder.  
(output pth: resnet50_amp.pth; resnet50_aug_amp.pth)

##3.3 ****data efficiency+error analysis****

###3.3.1 ****resnet_data_efficiency_utils.py****
This is the helper-function file for the data efficiency experiment. It handles dataset loading, stratified subsampling, DataLoader construction, ResNet-50 model building, training and evaluation, confusion matrix plotting, and result saving/summarization, and is not meant to be run alone.

###3.3.2 ****run_resnet_data_efficiency.py****
This is the main script for the ResNet-50 data efficiency experiment. It runs all settings across 100%, 50%, 25%, 10%, and 5% of the training data, with pretrained / scratch and multiple random seeds, and then summarizes the results and plots the final data efficiency curve.

****output****
```text
experiment result/
├── all_results_raw.csv
├── summary_results.csv
├── data_efficiency_accuracy.png
└── resnet50_full_{pretrained|scratch}_frac{fraction}_seed{seed}/
    ├── best_model.pth
    ├── history.csv
    ├── per_class_metrics.csv
    ├── confusion_matrix.npy
    ├── confusion_matrix.png
    └── summary.json
```
The script generates a folder named experiment result/. Inside it, there are overall summary files and 30 configuration-specific subfolders. Each subfolder follows the naming format resnet50_full_{pretrained|scratch}_frac{fraction}_seed{seed} and contains the saved model and evaluation outputs for that run. The overall results and best model are shown in the 'important experiment results' folder. 

3.3.3 ****error analysis.py****
This loads the selected best model checkpoint from the data efficiency experiment, reruns inference on the full PathMNIST test set, and selects 20 representative misclassified examples according to predefined error-type pairs, while saving the corresponding images and prediction tables. The examples and the explanation for them are shown in the 'targeted_error_analysis' folder.
(Required path: experiment result/resnet50_full_pretrained_frac1.0_seed42/best_model.pth)



