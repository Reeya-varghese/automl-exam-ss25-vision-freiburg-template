import torch
import torch.nn as nn
from torchvision import models, transforms
import timm

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transforms(mean, std, phase="train", backbone_name="resnet18"):
    """
    Returns image transformation based on training phase.
    Note: This function now provides BASE transforms only. Augmentation is handled
    separately in the AutoML class to ensure proper train/val splitting.

    Args:
        mean (tuple): Dataset mean for normalization.
        std (tuple): Dataset standard deviation for normalization.
        phase (str): 'train' or 'test' - controls whether augmentations to be applied or not.
        backbone_name (str): Name of the backbone model.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    is_vit = "vit" in backbone_name.lower()
    tf = [transforms.Resize((224, 224))]

    if phase == "train":
        tf += [
            transforms.RandomRotation(10),  
            transforms.RandomHorizontalFlip(p=0.5)
        ]

    tf.append(transforms.ToTensor())

    if is_vit:
        tf.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    tf.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(tf)

def get_base_transforms(mean, std, backbone_name="resnet18"):
    """
    Returns base transformation pipeline without any augmentation.
    Useful for validation sets.

    Args:
        mean (tuple): Dataset mean for normalization.
        std (tuple): Dataset standard deviation for normalization.
        backbone_name (str): Name of the backbone model.

    Returns:
        torchvision.transforms.Compose: Base transformation pipeline.
    """
    is_vit = "vit" in backbone_name.lower()
    
    tf = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]

    if is_vit:
        tf.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    tf.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(tf)

def load_resnet18(grayscale=False):
    """
    Loads a ResNet18 model. Modifies the first conv layer if grayscale is True.

    Args:
        grayscale (bool): adjust input layer for grayscale images.

    Returns:
        nn.Module: Modified ResNet18 model.
    """
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if grayscale:
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(1, conv1.out_channels, conv1.kernel_size,
                                stride=conv1.stride, padding=conv1.padding, bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(conv1.weight.sum(dim=1, keepdim=True))
    return model

def load_efficientnet_b0(grayscale=False):
    """
    Loads EfficientNet-B0 model. Modifies the first conv layer if grayscale is True.
    
    Args:
        grayscale (bool): adjust input layer for grayscale images.

    Returns:
        nn.Module: Modified EfficientNet-B0 model.
    """
    
    model = timm.create_model('efficientnet_b0', pretrained=True)
    if grayscale:
        conv = model.conv_stem
        model.conv_stem = nn.Conv2d(1, conv.out_channels, conv.kernel_size,
                                    stride=conv.stride, padding=conv.padding, bias=False)
        with torch.no_grad():
            model.conv_stem.weight = nn.Parameter(conv.weight.sum(dim=1, keepdim=True))
    return model

def load_vit(grayscale=False):
    """
    Loading Vision Transformer (ViT) model.
    
    Args:
        grayscale (bool): Whether input is grayscale.

    Returns:
        nn.Module: ViT model with pretrained weights.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    return model

def get_backbone_loader(backbone_name):
    """
    Returns the model loader function based on the backbone name.

    Args:
        backbone_name (str): Name of the model backbone.

    Returns:
        Callable: Loader function for the model.
    """

    loaders = {
        "resnet18": load_resnet18,
        "efficientnet_b0": load_efficientnet_b0,
        "vit_base_patch16_224": load_vit
    }
    
    assert backbone_name in loaders, f"Unsupported backbone: {backbone_name}"
    return loaders[backbone_name]

def get_model(backbone_name, num_classes, grayscale=False, custom_head=None):
    """
    Constructs a model by attaching a custom head to the selected backbone.

    Args:
        backbone_name (str): Identifier of the backbone model.
        num_classes (int): Number of output classes for classification.
        grayscale (bool): Whether input images are grayscale.
        custom_head (nn.Module): Optional custom head module.

    Returns:
        nn.Sequential: Combined model (backbone + head).
    """

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

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Use custom head if provided, otherwise use default head
    if custom_head is not None:
        head = custom_head
    else:
        head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(features_dim),
            nn.Dropout(0.1), 
            nn.Linear(features_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),  
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2), 
            nn.Linear(1024, num_classes)
        )

    return nn.Sequential(backbone, head)