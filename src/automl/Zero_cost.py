import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
        grayscale = self.real_input.shape[1] == 1

        self.BACKBONE_NAMES = [
            "resnet18",
            "efficientnet_b0",
            "vit_base_patch16_224",
            "swin_tiny_patch4_window7_224",
            "convnext_tiny"
        ]

        self.backbones = {
            name: get_backbone_loader(name)(grayscale=grayscale).to(self.device).eval()
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
        hidden_layers = random.choice([
            [1024, 512],
            [2048, 1024, 512],
            [2048, 1024]
        ])
        dropout = random.choice([0.0, 0.1, 0.2])
        use_bn = random.choice([True, False])
        activation = nn.GELU() if random.random() < 0.5 else nn.ReLU()

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

    # ✅ Now add final output layer with correct output class size
        layers.append(nn.Linear(prev_dim, self.num_classes))

        return nn.Sequential(*layers)



    def normalize(self, score_list):
        min_val, max_val = min(score_list), max(score_list)
        return [(s - min_val) / (max_val - min_val + 1e-8) for s in score_list]

    
    def extract_features(self, backbone, backbone_name, input_tensor):
        with torch.no_grad():
            if "vit" in backbone_name:
                features = backbone.forward_features(input_tensor)

                if features.ndim == 3:
                    features = features[:, 0, :]  # [CLS] token

                else:
                    features = features.mean(dim=1)  # fallback if [CLS] doesn't exist
                return features
            
            elif "efficientnet" in backbone_name:
                x = backbone.forward_features(input_tensor)
                return F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
            
            elif "resnet" in backbone_name:
                x = backbone.conv1(input_tensor); x = backbone.bn1(x); x = backbone.relu(x)
                x = backbone.maxpool(x); x = backbone.layer1(x); x = backbone.layer2(x)
                x = backbone.layer3(x); x = backbone.layer4(x); x = backbone.avgpool(x)
                return torch.flatten(x, 1)
            
            elif "swin" in backbone_name or "convnext" in backbone_name:
                x = backbone.forward_features(input_tensor)
                return x.mean(dim=1) if x.ndim == 3 else x  # handle [B, C, H, W] or [B, Tokens, D]
            
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")

    def get_feature_dim(self, model, backbone_name):
        input_tensor = self.real_input
        if any(k in backbone_name for k in ["vit", "swin", "convnext"]) and input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)

        feats = self.extract_features(model, backbone_name, input_tensor)
        if feats is None:
            raise ValueError(f"Feature extraction failed for {backbone_name}")
    
        return feats.shape[1]


    
    def get_top_k_candidates(self):
        candidates = []
    
        for i in range(self.num_candidates):
            backbone_name = random.choice(self.BACKBONE_NAMES)

            if any(name in backbone_name for name in ["vit", "swin", "convnext"]) and self.real_input.shape[1] == 1:
                input_tensor = self.real_input.repeat(1, 3, 1, 1)
            else:
                input_tensor = self.real_input


            backbone = self.backbones[backbone_name]
            feat_dim = self.get_feature_dim(backbone, backbone_name)

            head = self.generate_random_head(feat_dim).to(self.device)

            feats = self.extract_features(backbone, backbone_name, input_tensor)

            jac = self.get_jacobian_score(head, feats)
            grad = self.get_gradnorm_score(head, feats, self.real_target)

            candidates.append({
                "backbone": backbone_name,
                "head": deepcopy(head),
                "jacobian_score": jac,
                "gradnorm_score": grad,
                "id": f"{backbone_name}_{i}"
            })

        # Normalize and score
        jac_norm = self.normalize([c["jacobian_score"] for c in candidates])
        grad_norm = self.normalize([c["gradnorm_score"] for c in candidates])
        for i, c in enumerate(candidates):
            c["combined_score"] = 0.5 * jac_norm[i] + 0.5 * grad_norm[i]

        ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        print(f"🎯 Backbone: {backbone_name}, feat_dim: {feat_dim}, feats.shape: {feats.shape}")

        return ranked[:self.top_k]
