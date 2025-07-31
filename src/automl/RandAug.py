import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter, defaultdict


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



# =============================================================================
# 📈 STRATEGY 2: Full Oversampling with RandAug (Increases dataset size)
# =============================================================================

class AugmentedOversampledDataset(Dataset):
    """
    Dataset with actual duplicated and augmented samples to make all class sizes equal.
    """
    def __init__(self, data_tuples):
        self.data = data_tuples

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


def oversample_with_randaugment(dataset, class_counts, max_count, randaug, base_aug):
    """
    Create (img, label) list where each class is expanded to match max_count using augmentation.
    """
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    new_data = []

    for label, indices in class_to_indices.items():
        for idx in indices:
            img, lbl = dataset[idx]
            img = base_aug(img)
            new_data.append((img, lbl))

        for _ in range(max_count - len(indices)):
            idx = random.choice(indices)
            img, lbl = dataset[idx]
            img = randaug(img)
            new_data.append((img, lbl))

    return new_data


def prepare_fully_oversampled_dataset(
    dataset, image_size=(224, 224), grayscale=False, n=2, m=9, mean=(0.5,), std=(0.5,)
):
    """
    Returns a dataset with all classes having equal length (fully oversampled with RandAug).
    """
    class_counts = compute_class_distribution(dataset)
    max_count = max(class_counts.values())

    normalize = transforms.Normalize(mean, std)

    base_aug = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize
    ])

    randaug = transforms.Compose([
        transforms.Resize(image_size),
        RandAugmentFixed(n=n, m=m),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize
    ])

    oversampled_data = oversample_with_randaugment(dataset, class_counts, max_count, randaug, base_aug)
    return AugmentedOversampledDataset(oversampled_data), class_counts
