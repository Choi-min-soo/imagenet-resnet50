# dataset.py
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip,
    Resize, CenterCrop, ColorJitter, RandomErasing, RandAugment
)

def build_transform(cfg: dict, mode="train"):
    crop_size = cfg["crop_size"]
    resize_size = cfg["resize_size"]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        transform = Compose([
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            RandomErasing(p=0.25, scale=(0.02, 0.2))
        ])
    else:
        transform = Compose([
            Resize(resize_size),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    return transform

def get_dataset(cfg: dict, mode="train"):
    data_dir = cfg["dataset"]["data_dir"]
    split = "train" if mode == "train" else "val"
    transform = build_transform(cfg["dataset"], mode=mode)
    return ImageFolder(root=os.path.join(data_dir, split), transform=transform)
