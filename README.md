# Footwear Impression Matching System

![Footwear Banner](https://img.shields.io/badge/Forensic-Footwear%20Analysis-blue)
![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced deep learning-based system for matching crime scene footwear impressions to reference databases, designed for forensic analysis applications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Interactive Demo](#interactive-demo)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)

## Overview

Footwear impressions are among the most frequently secured types of evidence at crime scenes. The process of matching crime scene footwear impressions to reference databases is a critical task in forensic analysis. This repository implements a state-of-the-art deep learning system for automating this matching process.

The system handles the challenging aspects of footwear impression matching:
- Incomplete impressions due to partial contact
- Noise and distortions in crime scene impressions
- Variations in pressure and movement
- Different surface textures and contaminants

Our approach combines advanced convolutional neural networks, attention mechanisms, and specialized correlation techniques to achieve robust matching performance.

## Features

### Data Processing & Augmentation
- Specialized preprocessing for footwear impressions
- Background removal with white background preservation
- Extensive augmentation techniques specifically designed for footwear impressions:
  - Elastic deformations to simulate physical distortions
  - Partial impression simulation (cutouts)
  - Smudge and artifact simulation
  - Perspective transformations
  - Various noise patterns

### Advanced Architecture
- Siamese network with attention mechanisms (CBAM, SE)
- Multi-channel normalized cross-correlation (MCNCC)
- Domain-specific feature extraction for track and reference impressions
- Self-attention for pattern enhancement
- EMA (Exponential Moving Average) model for improved generalization

### Training & Evaluation
- Multi-objective learning with combined loss functions
- Support for triplet and contrastive learning
- Comprehensive evaluation metrics for forensic applications
- Visualization tools for analysis and interpretation
- Configuration-based experimentation

## System Architecture

The system consists of several components:

1. **Data Pipeline**
   - Preprocessing modules for track and reference impressions
   - Augmentation techniques tailored for forensic footwear analysis
   - Specialized dataloaders for pair and triplet sampling

2. **Neural Network Architecture**
   - Backbone: ResNet/ResNeXt for feature extraction
   - Attention modules: CBAM and Squeeze-and-Excitation
   - Domain-specific projections for track and reference domains
   - Multi-Channel Normalized Cross-Correlation for robust pattern matching

3. **Loss Functions**
   - Enhanced contrastive loss
   - Focal loss for handling class imbalance
   - Triplet loss for metric learning
   - Regularization techniques

4. **Training & Inference**
   - EMA model maintenance
   - Mixed precision training
   - Learning rate scheduling
   - Evaluation metrics computation
   - Visualization utilities

Here's a high-level diagram of the architecture:

```
Input Images → Preprocessing → Data Augmentation → Feature Extraction → 
Attention Mechanisms → Domain Projections → Pattern Correlation → 
Classification → Matching Result
```

## Dataset

The system is designed to work with the FID-300 (Footwear Impression Database) which contains:

- 300 crime scene (probe) footwear impressions
- 1175 reference impressions
- Label mapping between probe and reference impressions

The dataset organization structure should be:

```
data/
├── references/      # Reference impressions
├── tracks_cropped/  # Cropped crime scene impressions
└── label_table.csv  # Mapping between crime scene and reference impressions
```

### Citation for FID-300

If you use the FID-300 dataset, please cite:

```
Kortylewski, A., Albrecht, T., & Vetter, T. (2014). Unsupervised Footwear Impression Analysis and Retrieval from Crime Scene Data. ACCV 2014, Workshop on Robust Local Descriptors.
```

## Installation

### Requirements

- Python 3.7+ 
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JJshome/footwear-impression-matching.git
cd footwear-impression-matching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Obtain the FID-300 dataset and organize it as described in the [Dataset](#dataset) section.

2. Process the dataset:
```bash
python data/process_dataset.py --data_dir /path/to/FID-300 --output_dir data/processed --split_ratio 0.8
```

This will:
- Process all reference and track images (removing background, normalizing)
- Create positive and negative pairs for training
- Split into training and validation sets
- Save the processed data to the specified output directory

### Training

The system uses YAML configuration files for experiment setup. Two configurations are provided:

- `configs/default.yaml`: Full configuration with all features enabled
- `configs/fast.yaml`: Lighter configuration for quick experimentation

To train the model:

```bash
# Using default configuration
python train.py --config configs/default.yaml

# Using fast configuration
python train.py --config configs/fast.yaml

# Custom output directory
python train.py --config configs/default.yaml --output_dir results/experiment1
```

The training script will:
- Load the dataset
- Build the model based on the configuration
- Train for the specified number of epochs
- Perform validation at regular intervals
- Save checkpoints and visualizations
- Track and plot metrics

### Evaluation

To evaluate a trained model on a test set:

```bash
python inference.py --model path/to/model.pth --test_csv data/processed/val_pairs.csv
```

This will generate:
- Precision-recall curve
- ROC curve
- Detailed metrics (AP, accuracy, AUC)
- Analysis of failure cases

### Inference

For matching a single track impression against a reference database:

```bash
python inference.py --model path/to/model.pth --track_img path/to/track.jpg --ref_dir path/to/references
```

This will:
- Match the track image against all reference images
- Return and visualize the top matches
- Save the results to the output directory

### Interactive Demo

An interactive demo is provided for easier testing and demonstration:

```bash
python demo.py --model path/to/model.pth --ref_dir path/to/references
```

The demo provides a GUI interface where you can:
- Load a track image
- Match it against the reference database
- Visualize the results with similarity scores

## Results

Our model achieves the following performance on the FID-300 dataset:

| Metric | Value |
|--------|-------|
| Accuracy | ~92% |
| Average Precision | ~0.95 |
| ROC AUC | ~0.97 |
| Rank-1 Retrieval | ~90% |

The system shows robust performance even with partial impressions and various distortions commonly found in crime scene evidence.

### Visualization Tools

The repository includes visualization tools for analyzing the results:

- Precision-recall curves
- ROC curves
- Embedding visualization
- Failure case analysis

Check the `notebooks/visualization.ipynb` for examples of these visualizations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

1. Kortylewski, A., Albrecht, T., & Vetter, T. (2014). Unsupervised Footwear Impression Analysis and Retrieval from Crime Scene Data. ACCV 2014, Workshop on Robust Local Descriptors.

2. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV).

3. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition.

4. Kong, T., Sun, F., Yao, A., Liu, H., Lu, M., & Chen, Y. (2017). Ron: Reverse connection with objectness prior networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition.

5. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

If you find this work useful, please consider citing:

```
[Citation information to be added]
```
