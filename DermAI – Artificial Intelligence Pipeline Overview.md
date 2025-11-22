# DermAI – Artificial Intelligence Pipeline Overview

This document summarizes the AI components developed for the DermAI system.

1. Image Validation Module

A full pre-processing and verification pipeline to ensure:

Correct image format

Proper skin lesion localization

Absence of artifacts (text, borders, stickers)

Adequate resolution and color profile

This module prevents invalid images from being processed by the classifier.

2. Model Training Pipeline

Implemented using stratified 3-fold cross-validation to ensure:

Robust generalization

Fair evaluation on non-seen data

Stable performance across splits

Includes:

ResNet50 fine-tuning

Class imbalance handling

Learning rate scheduling

Extensive augmentation

3. Ensemble Learning

Three fold models are combined using probability averaging.
The ensemble provides:

Improved stability

Higher malignant recall

Reduced variance across folds

4. Grad-CAM Explainability

Grad-CAM heatmaps were generated to highlight regions of the lesion influencing the prediction, improving interpretability for medical use.

5. Comparative Study

Multiple deep-learning architectures were trained:

ResNet50

EfficientNet

DenseNet

MobileNet

VGG

InceptionV3

Custom CNN

The comparison supports the selection of ResNet50 as the optimal architecture for DermAI.

6. Final Training (Production Model)

After identifying the best-performing fold and optimal threshold, a final model will be trained using:

Full dataset

Best hyperparameters

Selected threshold for inference

This final model becomes the production model for DermAI.
