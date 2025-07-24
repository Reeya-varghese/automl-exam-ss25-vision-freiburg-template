import torch
import torch.nn as nn
from torchvision import models, transforms

import timm

# --- Utility for device selection ---
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Augmentation and Normalization pipeline ---
def get_train_transforms(mean, std):
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
def load_resnet18(grayscale=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if grayscale:
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(1, conv1.out_channels, conv1.kernel_size,
                                stride=conv1.stride, padding=conv1.padding, bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(conv1.weight.sum(dim=1, keepdim=True))
    return model

def load_efficientnet_b0(grayscale=False):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    if grayscale:
        conv = model.conv_stem
        model.conv_stem = nn.Conv2d(1, conv.out_channels, conv.kernel_size,
                                    stride=conv.stride, padding=conv.padding, bias=False)
        with torch.no_grad():
            model.conv_stem.weight = nn.Parameter(conv.weight.sum(dim=1, keepdim=True))
    return model

def load_vit(grayscale=False):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    if grayscale:
        model.patch_embed.proj = nn.Conv2d(1, model.patch_embed.proj.out_channels,
                                           kernel_size=16, stride=16)
    return model

def get_backbone_loader(backbone_name):
    loaders = {
        "resnet18": load_resnet18,
        "efficientnet_b0": load_efficientnet_b0,
        "vit_base_patch16_224": load_vit
    }
    assert backbone_name in loaders, f"Unsupported backbone: {backbone_name}"
    return loaders[backbone_name]

def get_model(backbone_name, num_classes, grayscale=False, num_layers_to_freeze=0, custom_head=None):
    backbone = get_backbone_loader(backbone_name)(grayscale=grayscale)

    if backbone_name.startswith("resnet"):
        features_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif "efficientnet" in backbone_name:
        features_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    elif "vit" in backbone_name:
        backbone.head = nn.Identity()
        features_dim = 768
        if features_dim is None:
            raise ValueError("Unable to extract ViT feature dimension")
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if custom_head is None:
        head = nn.Sequential(
            nn.Flatten(),
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

    model = nn.Sequential(backbone, head)
    return model

