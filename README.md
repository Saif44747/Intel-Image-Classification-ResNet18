
# Transfer Learning with ResNet18 — Intel Image Classification

**Project**: Transfer Learning with ResNet18 for Scene Classification on the Intel Image Dataset  
**Repository**: (put GitHub URL here)  
**Authors**: Your Name, Group Members

---

## Overview

This repository contains code to reproduce a transfer learning experiment: fine-tuning a pretrained ResNet18 model on the **Intel Image Classification** dataset (6 classes: `buildings, forest, glacier, mountain, sea, street`). The repo includes scripts for:

- training (`train.py`)
- evaluating on a held-out test set (`eval.py`)
- running inference on new images (`infer.py`)
- utilities to generate plots used in the paper (`utils/plot_results.py`)

Expected outputs (saved under `results/`):
- `results/best_model.pth` — best checkpoint (by validation accuracy)
- `results/metrics.json` — test metrics (accuracy, precision, recall, f1)
- `results/confusion_matrix.png` — confusion matrix heatmap
- `results/accuracy_curve.png` / `results/loss_curve.png` — training curves

---

## Quick requirements

Tested with:

- Python 3.8+  
- PyTorch (1.10+) and torchvision (compatible with your CUDA version)  
- Albumentations, OpenCV, scikit-learn, matplotlib, tqdm, pillow

Example minimal `requirements.txt`:


torch
torchvision
albumentations
opencv-python
scikit-learn
matplotlib
tqdm
pillow
tensorboard

Install:
```bash
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt



Directory structure (recommended
project/
  ├─ data/                       # dataset (not included in repo)
  │   ├─ seg_train/              # train folders: seg_train/<class>/*.jpg
  │   ├─ seg_test/               # test folders: seg_test/<class>/*.jpg
  │   └─ seg_pred/               # prediction images (optional)
  ├─ src/
  │   ├─ dataset.py
  │   ├─ model.py
  │   ├─ train.py
  │   ├─ eval.py
  │   └─ infer.py
  ├─ results/                    # saved models, plots, metrics
  ├─ notebooks/                  # optional demo notebooks
  ├─ images/                     # images for report (sample_images.png, etc.)
  ├─ requirements.txt
  └─ README.md


Download dataset (Kaggle)

kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip -d data/
# you should then have data/seg_train/, data/seg_test/, data/seg_pred/


Prepare dataset for training/validation/test (if needed)

python scripts/split_dataset.py --data_dir ./data/seg_train --val_ratio 0.2 --test_ratio 0.1




output 
Epoch [1/10] | Train Acc: 0.8482 | Val Acc: 0.9113 | Saved new best model!
...
Epoch [6/10] | Train Acc: 0.9267 | Val Acc: 0.9277 | Saved new best model!


Output: per-image predicted class and probability, e.g.:

sample1.jpg -> sea (0.87)
sample2.jpg -> mountain (0.62)






steps to perform om google collab 

Google Colab (if you prefer cloud GPU)

Open a Colab notebook and mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


Install requirements inside Colab:

!pip install -r /content/drive/MyDrive/project/requirements.txt


Download dataset from Kaggle (in Colab):

!pip install kaggle
# upload your kaggle.json to Colab or to drive and move to ~/.kaggle/
!kaggle datasets download -d puneet6060/intel-image-classification
!unzip intel-image-classification.zip -d /content/data/


Run training (same commands as local but prefix with !):

!python /content/drive/MyDrive/project/src/train.py --data_dir /content/data/ ...

Tips & Troubleshooting

CUDA / GPU not found: Ensure GPU drivers + CUDA + correct PyTorch build installed. Test with:

import torch
torch.cuda.is_available()


Out Of Memory (OOM): reduce batch_size, enable num_workers=0, or use mixed precision training (AMP).

Slow data loading: increase --num_workers if CPU RAM allows.

Different torchvision versions: If model = torchvision.models.resnet18(pretrained=True) gives a deprecation warning, use new API if required by your torchvision:

from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)


Reproducibility: set seed for random, numpy and torch:

import random, numpy as np, torch
seed=42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


