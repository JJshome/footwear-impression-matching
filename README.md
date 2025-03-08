# Footwear Impression Matching System

An advanced deep learning-based system for matching crime scene footwear impressions to reference databases. This system is designed for forensic analysis applications.

## Overview

This repository implements a state-of-the-art footwear impression matching system based on the FID-300 dataset. It includes comprehensive data processing, augmentation techniques specifically designed for footwear impressions, and an advanced neural network architecture with attention mechanisms.

## Features

- **Data Processing**: Tools for preprocessing footwear impression images
- **Extensive Augmentation**: Specialized techniques for footwear impressions with white background handling
- **Advanced Architecture**: Siamese network with attention mechanisms (CBAM, SE)
- **Multi-objective Learning**: Combined loss functions (Focal, Contrastive, Triplet)
- **Evaluation Metrics**: Precision, recall, AP, AUC metrics for forensic applications

## Dataset

The system is built for the FID-300 (Footwear Impression Database) which contains:
- Reference impressions: Laboratory-created footwear impressions
- Crime scene impressions: Footwear impressions collected from crime scenes
- Label mapping: Correspondence between crime scene and reference impressions

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the FID-300 dataset from the source
2. Organize the data in the following structure:
```
data/
├── references/      # Reference impressions
├── tracks_cropped/  # Cropped crime scene impressions
└── label_table.csv  # Mapping between crime scene and reference impressions
```
3. Run the dataset preprocessing script:
```bash
python data/process_dataset.py
```

## Training

```bash
python train.py --config configs/default.yaml
```

## Evaluation

```bash
python test.py --model_path path/to/model --data_path path/to/test_data
```

## Model Architecture

The system uses an enhanced Siamese network with:
- ResNet/ResNeXt backbone
- Channel and Spatial Attention (CBAM)
- Squeeze-and-Excitation blocks
- Multi-Channel Normalized Cross-Correlation
- Domain-specific projection layers
- Enhanced contrastive learning

## Results

Performance metrics on FID-300 dataset:
- Mean Average Precision (mAP): [to be added]
- Rank-1 Retrieval Rate: [to be added]
- AUC: [to be added]

## Citation

If you use this code in your research, please cite:
```
[Citation to be added]
```

## Acknowledgments

This work builds upon the FID-300 dataset:
- Kortylewski, A., Albrecht, T., & Vetter, T. (2014). Unsupervised Footwear Impression Analysis and Retrieval from Crime Scene Data. ACCV 2014, Workshop on Robust Local Descriptors.
