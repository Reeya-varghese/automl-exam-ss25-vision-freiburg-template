import torch
import torch.nn as nn
from torchvision import models, transforms
import timm


class GrayscaleToRGBAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.adapter(x)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_transforms(mean, std, phase="train", backbone_name="resnet18"):
    tf = [transforms.Resize((224, 224))]

    if phase == "train":
        tf += [transforms.RandomRotation(15), transforms.RandomHorizontalFlip()]

    tf.append(transforms.ToTensor())
    tf.append(transforms.Normalize(mean, std))

    return transforms.Compose(tf)


# ---------------------- Backbone Loaders ----------------------
def load_resnet18(): return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
def load_efficientnet_b0(): return timm.create_model('efficientnet_b0', pretrained=True)
def load_vit(): return timm.create_model('vit_base_patch16_224', pretrained=True)
def load_swin(): return timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
def load_convnext(): return timm.create_model('convnext_tiny', pretrained=True)


def get_backbone_loader(name):
    return {
        "resnet18": load_resnet18,
        "efficientnet_b0": load_efficientnet_b0,
        "vit_base_patch16_224": load_vit,
        "swin_tiny_patch4_window7_224": load_swin,
        "convnext_tiny": load_convnext,
    }[name]


# ---------------------- Model Builder ----------------------
def get_model(backbone_name, num_classes, grayscale=False, custom_head=None):
    device = get_device()
    base = get_backbone_loader(backbone_name)().to(device)

    # Input adapter
    adapter = GrayscaleToRGBAdapter().to(device) if grayscale else nn.Identity()

    # Feature extractor
    if "resnet" in backbone_name:
        features_dim = base.fc.in_features
        base.fc = nn.Identity()
        extractor = nn.Sequential(base, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    elif "efficientnet" in backbone_name:
        features_dim = base.classifier.in_features
        base.classifier = nn.Identity()
        extractor = nn.Sequential(
            nn.Lambda(lambda x: base.forward_features(x)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    elif "vit" in backbone_name:
        features_dim = base.head.in_features
        base.head = nn.Identity()
        extractor = nn.Sequential(
            nn.Lambda(lambda x: base.forward_features(x).mean(dim=1))
        )

    elif "swin" in backbone_name or "convnext" in backbone_name:
        features_dim = base.num_features
        extractor = nn.Sequential(
            nn.Lambda(lambda x: base.forward_features(x)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Head
    if custom_head is None:
        head = nn.Sequential(
            nn.BatchNorm1d(features_dim),
            nn.Linear(features_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )
    else:
        head = custom_head

    # Full model
    model = nn.Sequential(adapter, extractor, head).to(device)
    return model
