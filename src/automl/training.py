import logging
from typing import Any, Tuple
import torch
from copy import deepcopy
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score
import random
import optuna
from model import get_model, get_transforms
from utils import calculate_mean_std
from torch.utils.data import Subset, random_split
from dac import DynamicAlgorithmController
from torchvision import transforms

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
            self,
            seed: int,
            num_layers_to_freeze: int = 0,
            lr: float = 0.001,
            use_augmentation: bool = True,
            backbone: str = "resnet18",
            batch_size: int = 64,
            epochs: int = 20,
            custom_head: nn.Module = None,
            optimizer='adam'
    ) -> None:
        self.seed = seed
        self.num_layers_to_freeze = num_layers_to_freeze
        self.optimizer = optimizer
        self.lr = lr
        self.custom_head = custom_head
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed()
        self.dac = None  # Will be initialized conditionally

    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return self._history

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, dataset_class: Any, subsample: int = None, trial: optuna.trial.Trial = None) -> "AutoML":

        unsupported = ["swin", "convexnet"]
        if any(x in self.backbone.lower() for x in unsupported):
            raise ValueError(f"{self.backbone} is disabled in this configuration.")

        mean, std = calculate_mean_std(dataset_class)
        
        # Create base transform WITHOUT augmentation first
        # Import the new function if available, otherwise fallback to test phase
        try:
            from model import get_base_transforms
            base_transform = get_base_transforms(mean, std, backbone_name=self.backbone)
        except ImportError:
            base_transform = get_transforms(mean, std, phase="test", backbone_name=self.backbone)
        
        # Create dataset with base transform only for splitting
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=base_transform  # No augmentation yet
        )

        if subsample is not None:
            indices = np.random.choice(len(dataset), subsample, replace=False)
            dataset = Subset(dataset, indices)

        # Split the dataset first
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len],
                                          generator=torch.Generator().manual_seed(self.seed))

        # Now create augmented transform for training set only
        rand_augment = transforms.RandAugment(num_ops=1, magnitude=5)
        if self.use_augmentation:
            train_transform = transforms.Compose([rand_augment, base_transform])
        else:
            train_transform = base_transform
        
        # Create validation transform (no augmentation)
        val_transform = get_transforms(mean, std, phase="test", backbone_name=self.backbone)
        
        # Apply different transforms to train and validation sets
        # We need to create new dataset instances with the appropriate transforms
        train_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=train_transform  # With augmentation
        )
        
        val_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=val_transform  # Without augmentation
        )
        
        if subsample is not None:
            train_dataset = Subset(train_dataset, indices)
            val_dataset = Subset(val_dataset, indices)
        
        # Apply the same split indices to both datasets
        train_set = Subset(train_dataset, train_set.indices)
        val_set = Subset(val_dataset, val_set.indices)

        # Build sampler based only on train_set
        train_targets = [train_set.dataset.dataset[i][1] if hasattr(train_set.dataset, 'dataset') 
                        else train_set.dataset[i][1] for i in train_set.indices]
        class_counts = np.bincount(train_targets)
        weights = 1. / class_counts[train_targets]
        sampler = WeightedRandomSampler(weights, len(train_targets))
        print(f"[BALANCE] Applied class balancing with WeightedRandomSampler.")
        print(f"[BALANCE] Sample count per class: {np.bincount(train_targets)}", flush=True)
        
        self._val_set = val_set
        train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        model = get_model(
            self.backbone,
            num_classes=dataset_class.num_classes,
            grayscale=(dataset_class.channels == 1),
            custom_head=self.custom_head
        ).to(self.device)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        print(f"[DEBUG] DAC ENABLED for {dataset_class.__name__}")
        self.dac = DynamicAlgorithmController(optimizer, initial_lr=self.lr)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience = 10
        wait = 0
        best_model_state = None

        model.train()
        for epoch in range(self.epochs):
            loss_per_batch = []
            all_preds = []
            all_targets = []
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
                all_preds.extend(torch.argmax(output, 1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            epoch_loss = np.mean(loss_per_batch)
            epoch_acc = accuracy_score(all_targets, all_preds)
            self._history["loss"].append(epoch_loss)
            self._history["acc"].append(epoch_acc)

            val_loss_per_batch = []
            val_preds = []
            val_targets = []
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    pred = torch.argmax(output, 1).cpu().numpy()
                    val_loss_per_batch.append(loss.item())
                    val_preds.extend(pred)
                    val_targets.extend(target.cpu().numpy())

            val_loss = np.mean(val_loss_per_batch)
            val_acc = accuracy_score(val_targets, val_preds)
            self._history["val_loss"].append(val_loss)
            self._history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.dac:
                self.dac.update(epoch_loss)
                optimizer = self.dac.get_optimizer()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch + 1} — Best Val Acc: {best_val_acc:.4f}")
                    break
            model.train()

        if best_model_state:
            model.load_state_dict(best_model_state)

        self._model = model.eval()
        self.device = self.device

        if trial:
            trial.set_user_attr("history", self._history)

        return self

    def predict_on(self, dataset_class: Any, split="test") -> Tuple[np.ndarray, np.ndarray]:
        mean, std = calculate_mean_std(dataset_class)
        test_transform = get_transforms(mean, std, phase="test", backbone_name=self.backbone)
        dataset = dataset_class(root="./data", split=split, download=True, transform=test_transform)
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
        logger.info("Writing predictions to disk")
        return predictions, labels

    def predict(self, dataset_class: Any) -> np.ndarray:
        preds, labels = self.predict_on(dataset_class, split="test")
        return preds, labels

    def evaluate_on_val(self) -> Tuple[np.ndarray, np.ndarray]:
        data_loader = DataLoader(self._val_set, batch_size=100, shuffle=False)
        predictions, labels = [], []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self._model(data)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred)
                labels.append(target.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(labels)