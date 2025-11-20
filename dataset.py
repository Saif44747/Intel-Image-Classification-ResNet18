# dataset.py
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, os

class ImgDataset(Dataset):
    def __init__(self, filepaths, labels, transforms=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = cv2.imread(self.filepaths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        label = self.labels[idx]
        return img, label


# ✅ Updated transforms for Albumentations v1.4.0+
train_tf = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # FIXED
    A.HorizontalFlip(),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(),
    A.Normalize(),  # mean/std default (0..1 -> normalized)
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(height=224, width=224),  # FIXED
    A.Normalize(),
    ToTensorV2(),
])
