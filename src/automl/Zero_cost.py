import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import timm

from copy import deepcopy
from model import get_backbone_loader
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ZeroCostCandidateGenerator:
    def __init__(self, real_input, real_target, num_candidates=100, top_k=10, num_classes=7):
        self.real_input = real_input.to(DEVICE)
        self.real_target = real_target.to(DEVICE)
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_classes = num_classes
        self.device = DEVICE

        self.BACKBONE_NAMES = ["resnet18", "efficientnet_b0", "vit_base_patch16_224"]
        self.backbones = {
            name: get_backbone_loader(name)(grayscale=True).to(self.device).eval()
            for name in self.BACKBONE_NAMES
        }
        

    # ------------------ Scoring Functions ------------------ #
    def get_jacobian_score(self, model, input_tensor):
        model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = model(input_tensor)
        jacobian = torch.autograd.grad(outputs=output.sum(), inputs=input_tensor, create_graph=True)[0]
        return jacobian.norm().item()

    def get_gradnorm_score(self, model, input_tensor, target_tensor):
        model.train()
        model.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output, target_tensor)
        loss.backward()
        return sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    def generate_random_head(self, input_dim):
        hidden_dim = random.choice([64, 128, 256])
        dropout = random.choice([0.0, 0.3, 0.5])
        use_bn = random.choice([True, False])
        activation = nn.GELU() if random.choice([True, False]) else nn.ReLU()

        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, self.num_classes))
        return nn.Sequential(*layers)

    def normalize(self, score_list):
        min_val, max_val = min(score_list), max(score_list)
        return [(s - min_val) / (max_val - min_val + 1e-8) for s in score_list]

    
    def extract_features(self, backbone, backbone_name, input_tensor):
        with torch.no_grad():
            if "vit" in backbone_name:
                return backbone.forward_features(input_tensor)[:, 0, :]
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
        model.eval()
        dummy_input = torch.randn(1, 1, 224, 224).to(self.device)
        feats = self.extract_features(model, backbone_name, dummy_input)
        return feats.shape[-1]  
    
    def get_top_k_candidates(self):
        candidates = []
    
        for i in range(self.num_candidates):
            backbone_name = random.choice(self.BACKBONE_NAMES)

            backbone = self.backbones[backbone_name]
            feat_dim = self.get_feature_dim(backbone, backbone_name)

            head = self.generate_random_head(feat_dim)

            feats = self.extract_features(backbone, backbone_name, self.real_input)

            jac = self.get_jacobian_score(head, feats)
            grad = self.get_gradnorm_score(head, feats, self.real_target)

            candidates.append({
                "backbone": backbone_name,
                "head": deepcopy(head),
                "jacobian_score": jac,
                "gradnorm_score": grad,
             
                "id": f"{backbone_name}_{i}"  # Add a unique ID
            })


        # Normalize and score
        jac_norm = self.normalize([c["jacobian_score"] for c in candidates])
        grad_norm = self.normalize([c["gradnorm_score"] for c in candidates])
        for i, c in enumerate(candidates):
            c["combined_score"] = 0.5 * jac_norm[i] + 0.5 * grad_norm[i]

        ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:self.top_k]
