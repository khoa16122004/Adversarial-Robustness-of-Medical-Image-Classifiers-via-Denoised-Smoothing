# Adversarial Robustness of Medical Image Classifiers via Denoised Smoothing

This repository implements robust medical image classification using denoised smoothing techniques, combining approaches from certified robustness and black-box defense methods.

## Overview

Medical image classification systems are vulnerable to adversarial attacks that can compromise their reliability in clinical settings. This project implements defense mechanisms using denoised smoothing combined with both first-order (FO) and zero-order (ZO) optimization approaches to create robust medical image classifiers.

## Reference Implementations

This work builds upon:
1. [Microsoft's Denoised Smoothing](https://github.com/microsoft/denoised-smoothing) - For certified robustness via denoising
2. [Black-box Defense](https://github.com/damon-demon/Black-Box-Defense) - For optimization techniques

## Datasets

The project utilizes two medical imaging datasets:

1. [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
   - Purpose: Brain tumor classification from MRI scans
   - Classes: Multiple tumor types and normal tissue

2. [Sipakmed Dataset](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
   - Purpose: Cervical cancer cell classification
   - Scope: Largest available cervical cytology dataset

## Repository Structure

```
Dataset/
├── Brain_Tumor/                # Brain tumor MRI dataset
│   ├── AT_DDN/                # Adversarial examples using DDN attack
│   ├── AT_FGSM/               # Adversarial examples using FGSM attack
│   ├── AT_PGD/                # Adversarial examples using PGD attack
│   ├── Testing/               # Clean test images
│   └── Training/              # Clean training images
│
└── SIPADMEK/                  # Cervical cancer dataset
    ├── AT_DDN/                # Adversarial examples using DDN attack
    ├── AT_FGSM/               # Adversarial examples using FGSM attack
    ├── AT_PGD/                # Adversarial examples using PGD attack
    └── process/               # Processed data directory            
│
├── main_ds.py                     # Main training script
├── ds_main.py                     # Denoiser training script
└── AE_DS_verify.py               # Certification evaluation script
```

## Key Components

### 1. Main Training Script (```main_ds.py```)
- Implements denoised smoothing training
- Features:
  - Stability loss for robust denoising
  - Cross-entropy loss for classification
  - First-order (FO) optimization
  - Zero-order (ZO) optimization
  - Integration with black-box defense techniques

### 2. Denoiser Training (```ds_main.py```)
- Focused on training the denoising network
- Features:
  - Pure reconstruction objective
  - Image quality preservation
  - Noise resilience training

### 3. Certification Evaluation (```AE_DS_verify.py```)
- Evaluates model robustness
- Features:
  - Certification accuracy measurement
  - Multiple noise standard deviation testing
  - Comprehensive robustness metrics


## Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- scipy
- pandas
- scikit-learn
- tqdm







