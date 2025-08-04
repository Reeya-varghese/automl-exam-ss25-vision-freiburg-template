import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from model import get_backbone_loader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ZeroCostCandidateGenerator:
    """
    This class generates and ranks random neural network heads for given 
    backbone models using zero-cost proxies
    Jacobian norm and GradNorm.

    Args:
        real_input (Tensor): Sample input tensor.
        real_target (Tensor): Corresponding target labels for the input.
        num_candidates (int): Number of random head candidates to evaluate.
        top_k (int): Number of top candidates to return.
        num_classes (int): Number of output classes for the classification task.
    """
    
    def __init__(self, real_input, real_target, num_candidates=100, top_k=10, num_classes=7):
        self.real_input = real_input.to(DEVICE)
        self.real_target = real_target.to(DEVICE)
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_classes = num_classes
        self.device = DEVICE
        grayscale = self.real_input.shape[1] == 1
    
        self.BACKBONE_NAMES = ["resnet18", "efficientnet_b0", "vit_base_patch16_224"]

        self.backbones = {
            name: get_backbone_loader(name)(grayscale=grayscale).to(self.device).eval()
            for name in self.BACKBONE_NAMES
        }

    def jacobian_score(self, model, input_tensor):
        """
        Computes the Jacobian norm score.

        Args:
            model (nn.Module): The head model to evaluate.
            input_tensor (Tensor): Feature input to the head.

        Returns:
            float: Jacobian norm score.
        """
        model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = model(input_tensor)
        jacobian = torch.autograd.grad(outputs=output.sum(), inputs=input_tensor, create_graph=True)[0]
        return jacobian.norm().item()

    def gradnorm_score(self, model, input_tensor, target_tensor):
        """
        Computes the GradNorm score.

        Args:
            model (nn.Module): The head model to evaluate.
            input_tensor (Tensor): Feature input to the head.
            target_tensor (Tensor): Ground truth labels.

        Returns:
            float: Sum of gradient norms over all model parameters.
        """
        model.train()
        model.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output, target_tensor)
        loss.backward()
        return sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    def generate_random_head(self, input_dim):
        """
        Generates a random classification head with varying architecture.

        Args:
            input_dim (int): Input feature dimension from the backbone.

        Returns:
            nn.Sequential: A sequential head network.
        """
        hidden_dim = random.choice([
            [1024, 512],
            [2048, 1024, 512],
            [2048, 1024]
        ])
        dropout = random.choice([0.0, 0.1, 0.2])
        use_bn = random.choice([True, False])
        activation = nn.GELU() if random.random() < 0.5 else nn.ReLU()

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for hidden in hidden_dim:
            layers.append(nn.Linear(prev_dim, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden

        layers.append(nn.Linear(prev_dim, self.num_classes))
        return nn.Sequential(*layers)

    def normalize(self, score_list):
        """
        Normalizes a list of scores to the [0, 1] range.

        Args:
            score_list (List[float]): List of raw scores.

        Returns:
            List[float]: Normalized scores.
        """
        min_val, max_val = min(score_list), max(score_list)
        return [(s - min_val) / (max_val - min_val + 1e-8) for s in score_list]

    def extract_features(self, backbone, backbone_name, input_tensor):
        """
        Extracts features from a backbone.

        Args:
            backbone (nn.Module): Backbone network.
            backbone_name (str): Identifier for the backbone architecture.
            input_tensor (Tensor): Input image tensor.

        Returns:
            Tensor: Extracted feature tensor.
        """
        with torch.no_grad():
            if "vit" in backbone_name:
                features = backbone.forward_features(input_tensor)
                return features[:, 0, :] if features.ndim == 3 else features.mean(dim=1)
            elif "efficientnet" in backbone_name:
                x = backbone.forward_features(input_tensor)
                return F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
            elif "resnet" in backbone_name:
                x = backbone.conv1(input_tensor); x = backbone.bn1(x); x = backbone.relu(x)
                x = backbone.maxpool(x); x = backbone.layer1(x); x = backbone.layer2(x)
                x = backbone.layer3(x); x = backbone.layer4(x); x = backbone.avgpool(x)
                return torch.flatten(x, 1)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")

    def get_feature_dim(self, model, backbone_name):
        """
        Args:
            model (nn.Module): The backbone network.
            backbone_name (str): Identifier for the backbone.

        Returns:
            int: Dimension of the extracted features.
        """
        model.eval()
        dummy_input = torch.randn(1, self.real_input.shape[1], 224, 224).to(self.device)
        if "vit" in backbone_name and dummy_input.shape[1] == 1:
            dummy_input = dummy_input.repeat(1, 3, 1, 1)  # Expand grayscale to RGB
        feats = self.extract_features(model, backbone_name, dummy_input)
        return feats.shape[-1]

    def get_top_k_candidates(self):
        """
        Ranks candidate architectures based on zero-cost metrics.

        Returns:
            List[Dict]: Top-k ranked candidates with scores.
        """
        candidates = []

        for i in range(self.num_candidates):
            backbone_name = random.choice(self.BACKBONE_NAMES)

            # Handle grayscale input expansion for ViT
            input_tensor = self.real_input.repeat(1, 3, 1, 1) if "vit" in backbone_name and self.real_input.shape[1] == 1 else self.real_input
            backbone = self.backbones[backbone_name]
            feat_dim = self.get_feature_dim(backbone, backbone_name)

            head = self.generate_random_head(feat_dim).to(self.device)
            feats = self.extract_features(backbone, backbone_name, input_tensor)

            jac = self.jacobian_score(head, feats)
            grad = self.gradnorm_score(head, feats, self.real_target)

            candidates.append({
                "backbone": backbone_name,
                "head": deepcopy(head),
                "jacobian_score": jac,
                "gradnorm_score": grad,
                "id": f"{backbone_name}_{i}"
            })

        # Normalize and combine scores
        jac_norm = self.normalize([c["jacobian_score"] for c in candidates])
        grad_norm = self.normalize([c["gradnorm_score"] for c in candidates])
        for i, c in enumerate(candidates):
            c["combined_score"] = 0.5 * jac_norm[i] + 0.5 * grad_norm[i]

        ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:self.top_k]
