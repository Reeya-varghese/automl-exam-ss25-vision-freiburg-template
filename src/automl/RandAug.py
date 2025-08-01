import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter, defaultdict
from torchvision.transforms import Lambda
from model import get_transforms
import torch
# =============================================================================
# 🧩 RAND-AUGMENT (Fixed Version)
# =============================================================================

def int_parameter(level, maxval): return int(level * maxval / 10)
def float_parameter(level, maxval): return float(level) * maxval / 10.

class RandAugmentFixed:
    """Fixed RandAugment implementation with robust PIL operations."""

    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.invert, 0, 1),
            (self.rotate, 0, 30),
            (self.posterize, 0, 4),
            (self.solarize, 0, 256),
            (self.solarize_add, 0, 110),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, 0., 0.3),
            (self.shear_y, 0., 0.3),
            (self.cutout, 0, 60),
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
    def invert(self, img, _): return ImageOps.invert(img)
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
    def color(self, img, level): return ImageEnhance.Color(img).enhance(float_parameter(level, 1.8) + 0.1)
    def contrast(self, img, level): return ImageEnhance.Contrast(img).enhance(float_parameter(level, 1.8) + 0.1)
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
    def cutout(self, img, level):
        size = int_parameter(level, 60)
        if size <= 0: return img
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        x, y = random.randint(0, w), random.randint(0, h)
        x1, y1 = np.clip(x - size // 2, 0, w), np.clip(y - size // 2, 0, h)
        x2, y2 = np.clip(x + size // 2, 0, w), np.clip(y + size // 2, 0, h)
        img_np[y1:y2, x1:x2] = 128
        return Image.fromarray(img_np)


# =============================================================================


def compute_class_distribution(dataset):
    return Counter([label for _, label in dataset])


class AugmentedMinorityDataset(Dataset):
    def __init__(self, dataset, class_counts, minority_threshold, base_aug, randaug):
        self.dataset = dataset
        self.class_counts = class_counts
        self.minority_threshold = minority_threshold
        self.base_aug = base_aug
        self.randaug = randaug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
    
        if self.class_counts[label] < self.minority_threshold:
            img = self.randaug(img)
        else:
            img = self.base_aug(img)

    # Optional debug assertion
        if isinstance(img, torch.Tensor):
            assert img.shape[0] in [1, 3], f"Unexpected channel shape: {img.shape}"

        return img, label


def get_weighted_sampler(dataset, class_counts):
    targets = [label for _, label in dataset]
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def prepare_augmented_balanced_dataset(
    dataset, image_size=(224, 224), minority_threshold=None, n=2, m=9, mean=(0.5,), std=(0.5,), backbone_name="resnet18"
):
    class_counts = compute_class_distribution(dataset)

    if minority_threshold is None:
        sorted_counts = sorted(class_counts.values(), reverse=True)
        minority_threshold = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

    normalize = transforms.Normalize(mean, std)
    # Use shared get_transforms for consistent logic (e.g., grayscale -> RGB for transformers)
    base_aug = get_transforms(mean, std, phase="train", backbone_name=backbone_name)

    randaug = transforms.Compose([
        RandAugmentFixed(n=n, m=m),
        *base_aug.transforms  # reuse resizing, tensor conversion, normalization
    ])


    dataset_aug = AugmentedMinorityDataset(dataset, class_counts, minority_threshold, base_aug, randaug)
    sampler = get_weighted_sampler(dataset_aug, class_counts)

    return dataset_aug, sampler
