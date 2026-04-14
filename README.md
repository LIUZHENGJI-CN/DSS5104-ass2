# DSS5104-ass2
DSS5104 team19 assignment 2

#1.overall description
This project builds a complete transfer learning pipeline for PathMNIST image classification. It begins with data loading, preprocessing, and visual inspection to verify the classes, input format, and overall data characteristics. In the model selection stage, three backbone architectures—ResNet-50, EfficientNet-B0, and Swin-T—are compared under settings including feature extraction, full fine-tuning, and data augmentation. Based on these results, the best-performing model was further refined through additional exploration of data augmentation, in order to examine more how this affects generalization performance. The final selected architecture is then used for a data efficiency experiment, comparing pretrained initialization and training from scratch at 100%, 50%, 25%, 10%, and 5% of the training data to examine the value of transfer learning under limited-data conditions. Finally, the best model is analyzed through confusion matrix analysis, per-class metrics, and representative misclassified examples, forming a complete workflow from model comparison to detailed evaluation.

#2.basic setup:
dataset: PathMNIST
pretrained architectures: ResNet-50, EfficientNet-B0, Swin Transformer(Tiny)
strategies: feature extraction, full fine-tuning, data augmentation

#3.code explanations
#3.1 model selection+strategy
