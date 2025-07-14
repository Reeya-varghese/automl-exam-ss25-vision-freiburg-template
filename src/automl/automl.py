"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import random
import numpy as np
import logging
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from sklearn.metrics import accuracy_score


from automl.dummy_model import get_resnet_model
from automl.utils import calculate_mean_std


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
        num_layers_to_freeze: int = 5,
        lr: float = 0.001,
        use_augmentation: bool = True,
         backbone: str = "resnet50",
         batch_size: int = 64,
         epochs: int = 10, 
    ) -> None:
        self.seed = seed
        self.num_layers_to_freeze = num_layers_to_freeze
        self.lr = lr
        self.backbone = backbone   
        self.epochs = epochs  
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation   
        self._model: nn.Module | None = None

    def fit(
        self,
        dataset_class: Any,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Transforms
        mean, std = calculate_mean_std(dataset_class)
        tfs = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if self.use_augmentation:
            tfs.insert(1, transforms.RandomHorizontalFlip())
            tfs.insert(1, transforms.RandomRotation(15))
        self._transform = transforms.Compose(tfs)

        
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        

        model = get_resnet_model(
            self.backbone,  # <-- Fix: always pass backbone_name as first argument
            num_classes=dataset_class.num_classes,
            num_layers_to_freeze=self.num_layers_to_freeze,
            grayscale=(dataset_class.channels == 1)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(10):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")
            print(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")
        self._model = model.eval()
        self.device = device
        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self._model(data)
                predicted = torch.argmax(output, 1).cpu().numpy()
                labels.append(target.numpy())
                predictions.append(predicted)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        return predictions, labels
    
    
def run_random_search(
    dataset_class: Any,
    seed: int = 42,
    num_trials: int = 5,
    output_path: str | Path = "predictions.npy"
):
    """Run random search for AutoML hyperparameters and save best predictions."""

    search_space = {
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "num_layers_to_freeze": [10, 20, 5]
    }

    best_acc = 0
    best_preds = None
    best_config = None

    for trial in range(num_trials):
        config = {
            "lr": random.choice(search_space["learning_rate"]),
            "num_layers_to_freeze": random.choice(search_space["num_layers_to_freeze"])
        }

        print(f"[Trial {trial+1}] Config: {config}")

        automl = AutoML(seed=seed, num_layers_to_freeze=config["num_layers_to_freeze"], lr=config["lr"])
        automl.fit(dataset_class)
        preds, labels = automl.predict(dataset_class)

        if not np.isnan(labels).any():
            acc = accuracy_score(labels, preds)
            print(f"Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_preds = preds
                best_config = config
        else:
            # If no labels (final exam test set)
            best_preds = preds
            best_config = config
            print("No labels available. Using first trial's predictions.")
            break

    print(f"✅ Best Config: {best_config}")
    print(f"🏁 Best Accuracy: {best_acc:.4f}" if best_acc > 0 else "Test labels unavailable")

    # Save predictions
    with open(output_path, "wb") as f:
        np.save(f, best_preds)


def optuna_objective(trial, dataset_class, seed=42, epochs=10, batch_size=64):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    num_layers_to_freeze = trial.suggest_categorical('num_layers_to_freeze', [0, 5, 10, 20])
    use_augmentation = trial.suggest_categorical('use_augmentation', [True])  # or [True, False] if you want to search
    backbone = trial.suggest_categorical('backbone', ['resnet18', 'resnet50'])
    automl = AutoML(
        seed=seed,
        num_layers_to_freeze=num_layers_to_freeze,
        lr=lr,
        use_augmentation=use_augmentation,
        backbone=backbone,
        batch_size=batch_size,
        epochs=epochs
    )
    automl.fit(dataset_class)
    preds, labels = automl.predict(dataset_class)
    acc = accuracy_score(labels, preds) if not np.isnan(labels).any() else 0
    trial.report(acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return acc