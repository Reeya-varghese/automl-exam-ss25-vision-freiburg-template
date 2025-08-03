import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter, defaultdict
from torchvision.transforms import Lambda
from model import get_transforms
import torch


def int_parameter(level, maxval): return int(level * maxval / 10)
def float_parameter(level, maxval): return float(level) * maxval / 10.

class RandAugmentFixed:
    """Random Augmentation methods used for the datasets and wegihted Sampler for Handling the Imbalanced data."""

    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.augment_list = [

            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.rotate, 0, 30),
            (self.posterize, 0, 4),
            (self.solarize, 0, 256),
            (self.solarize_add, 0, 110),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, 0., 0.3),
            (self.shear_y, 0., 0.3),
            (self.translate_x, 0., 150),
            (self.translate_y, 0., 150)
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

    def auto_contrast(self, img, _): return ImageOps.autocontrast(img)

    def equalize(self, img, _): return ImageOps.equalize(img)

    def rotate(self, img, level):
        degrees = int_parameter(level, 30)
        if random.random() > 0.5: degrees = -degrees
        return img.rotate(degrees, resample=Image.BILINEAR)
    
    def posterize(self, img, level): return ImageOps.posterize(img, 4 - int_parameter(level, 4))
    def solarize(self, img, level): return ImageOps.solarize(img, 256 - int_parameter(level, 256))
    
    def solarize_add(self, img, level):
        level = int_parameter(level, 110)
        if random.random() > 0.5: level = -level
        img_np = np.array(img).astype(np.int32)
        img_np = np.clip(img_np + level, 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))

    def brightness(self, img, level): return ImageEnhance.Brightness(img).enhance(float_parameter(level, 1.8) + 0.1)

    def sharpness(self, img, level): return ImageEnhance.Sharpness(img).enhance(float_parameter(level, 1.8) + 0.1)
    
    def shear_x(self, img, level):
        level = float_parameter(level, 0.3)
        if random.random() > 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)
    
    def shear_y(self, img, level):
        level = float_parameter(level, 0.3)
        if random.random() > 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)
    
    def translate_x(self, img, level):
        level = int_parameter(level, 150)
        if random.random() > 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)
    
    def translate_y(self, img, level):
        level = int_parameter(level, 150)
        if random.random() > 0.5: level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)
  

def compute_class_distribution(dataset):
    return Counter([label for _, label in dataset])


class AugmentedDataset(Dataset):
    def __init__(self, dataset,transform):
        self.dataset = dataset
        self.transform= transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)

        if isinstance(img, torch.Tensor):
            assert img.shape[0] in [1, 3], f"Unexpected channel shape: {img.shape}"

        return img, label


def WeightedSampler(dataset, class_counts):
    targets = [label for _, label in dataset]
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def AugmentDataset(
    dataset, image_size=(224, 224), n=2, m=9, mean=(0.5,), std=(0.5,), grayscale=False, backbone_name="resnet18"
):
    class_counts = compute_class_distribution(dataset)

    
    train_transform = transforms.Compose([
        RandAugmentFixed(n=n, m=m),  
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    dataset_aug = AugmentedDataset(dataset,train_transform)
    sampler = WeightedSampler(dataset_aug, class_counts)

    return dataset_aug, sampler
