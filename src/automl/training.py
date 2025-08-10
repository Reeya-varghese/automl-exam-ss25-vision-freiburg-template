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


# --- NEW: simple per-epoch emissions tracker ---
class EpochEmissionsTracker:
    """
    Starts/stops a fresh CodeCarbon tracker each epoch and accumulates raw kgCO2.
    We keep the dependency lazy so imports don’t slow startup when not used.
    """
    def __init__(self, project_name_prefix: str = "trial"):
        self.project_name_prefix = project_name_prefix
        self.total_kg = 0.0
        self._epoch_idx = -1
        self._tracker = None

    def start_epoch(self):
        self._epoch_idx += 1
        from codecarbon import EmissionsTracker  # lazy import
        self._tracker = EmissionsTracker(
            project_name=f"{self.project_name_prefix}_epoch_{self._epoch_idx}",
            measure_power_secs=10,
            save_to_file=True,
            log_level="error"
        )
        self._tracker.start()

    def stop_epoch(self) -> float:
        emitted = 0.0
        if self._tracker:
            try:
                emitted = self._tracker.stop() or 0.0
            except Exception:
                emitted = 0.0
        self.total_kg += emitted
        self._tracker = None
        return emitted


class TransformedSubset(torch.utils.data.Dataset):
    """
    A dataset wrapper that applies a given transform only to a specified subset.
    """
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        return self.transform(x), y

    def __len__(self):
        return len(self.indices)


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
        """
        Initialize AutoML instance.
        """
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
        self.dac = None  # Dynamic Algorithm Controller

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

    def fit(
        self,
        dataset_class: Any,
        subsample: int = None,
        trial: optuna.trial.Trial = None,
        per_trial_budget_kg: float | None = None,   # NEW: allowance (adjusted)
        efficiency_weight: float = 1.0              # NEW: for adjusted emissions
    ) -> "AutoML":
        """
        Train the model on a dataset.
        If per_trial_budget_kg is set, we track emissions each epoch and abort
        the trial immediately if adjusted emissions exceed the allowance.
        """
        base_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=None
        )

        if subsample is not None:
            indices = np.random.choice(len(base_dataset), subsample, replace=False)
            base_dataset = Subset(base_dataset, indices)

        # Split dataset into train and val indices
        train_len = int(0.8 * len(base_dataset))
        val_len = len(base_dataset) - train_len
        train_indices, val_indices = random_split(
            range(len(base_dataset)), [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Compute transforms
        mean, std = calculate_mean_std(dataset_class)
        rand_augment = transforms.RandAugment(num_ops=1, magnitude=5)
        base_transform = get_transforms(mean, std, phase="test", backbone_name=self.backbone)

        train_transform = transforms.Compose([rand_augment, base_transform]) if self.use_augmentation else base_transform
        val_transform = base_transform  # no augmentation for validation

        train_indices = train_indices.indices if isinstance(train_indices, Subset) else train_indices
        val_indices = val_indices.indices if isinstance(val_indices, Subset) else val_indices

        train_set = TransformedSubset(base_dataset, train_indices, train_transform)
        val_set = TransformedSubset(base_dataset, val_indices, val_transform)

        # Class-balanced sampler
        train_targets = [train_set.base_dataset[i][1] for i in train_set.indices]
        class_counts = np.bincount(train_targets)
        weights = 1. / class_counts[train_targets]
        sampler = WeightedRandomSampler(weights, len(train_targets))
        print(f"[BALANCE] Applied WeightedRandomSampler. Counts: {np.bincount(train_targets)}", flush=True)

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

        # Dynamic LR/optimizer controller
        print(f"[DEBUG] DAC ENABLED for {dataset_class.__name__}")
        self.dac = DynamicAlgorithmController(optimizer, initial_lr=self.lr)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        patience = 10
        wait = 0
        best_model_state = None

        # --- NEW: epoch-level CO2 tracker and allowance ---
        epoch_tracker = EpochEmissionsTracker(project_name_prefix=f"{dataset_class.__name__}_trial")
        adjusted_emissions_so_far = 0.0
        budget_guard_enabled = per_trial_budget_kg is not None
        eff_w = max(1e-8, float(efficiency_weight))

        # Train loop
        model.train()
        for epoch in range(self.epochs):
            if budget_guard_enabled:
                epoch_tracker.start_epoch()

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

            epoch_loss = float(np.mean(loss_per_batch)) if loss_per_batch else 0.0
            epoch_acc = accuracy_score(all_targets, all_preds) if all_preds else 0.0
            self._history["loss"].append(epoch_loss)
            self._history["acc"].append(epoch_acc)

            # Validation
            val_loss_per_batch = []
            val_preds, val_targets = [], []
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

            val_loss = float(np.mean(val_loss_per_batch)) if val_loss_per_batch else 0.0
            val_acc = accuracy_score(val_targets, val_preds) if val_preds else 0.0
            self._history["val_loss"].append(val_loss)
            self._history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            print(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # DAC update (may change LR and/or optimizer)
            if self.dac:
                self.dac.update(epoch_loss)
                optimizer = self.dac.get_optimizer()

            # Keep best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f" Early stopping at epoch {epoch + 1} — Best Val Acc: {best_val_acc:.4f}")
                    if budget_guard_enabled:
                        # still close the epoch tracker properly
                        emitted = epoch_tracker.stop_epoch()
                        print(f"[CARBON][EPOCH-END-ES] emitted={emitted:.6f} kg")
                    break

            # --- NEW: close epoch tracker and check allowance ---
            if budget_guard_enabled:
                epoch_emitted = epoch_tracker.stop_epoch()
                adjusted_emissions_so_far = epoch_tracker.total_kg / eff_w
                print(f"[CARBON][EPOCH] emitted={epoch_emitted:.6f} kg, "
                      f"adjusted_total={adjusted_emissions_so_far:.6f} kg, "
                      f"allowance={per_trial_budget_kg:.6f} kg")
                if adjusted_emissions_so_far > float(per_trial_budget_kg):
                    print(f"[CARBON][STOP] Trial exceeded per-trial budget at epoch {epoch + 1}. Aborting trial early.")
                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    self._model = model.eval()
                    if trial:
                        trial.set_user_attr("epoch_level_emissions_kg", epoch_tracker.total_kg)
                        trial.set_user_attr("budget_pruned_epoch", epoch + 1)
                    raise RuntimeError("PER_TRIAL_CARBON_BUDGET_EXCEEDED")

            model.train()

        if best_model_state:
            model.load_state_dict(best_model_state)

        self._model = model.eval()
        self.device = self.device

        if trial:
            trial.set_user_attr("history", self._history)
            # record emissions if we had a tracker
            if 'epoch_tracker' in locals():
                trial.set_user_attr("epoch_level_emissions_kg", epoch_tracker.total_kg)

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
        predictions = np.concatenate(predictions) if predictions else np.array([])
        labels = np.concatenate(labels) if labels else np.array([])
        logger.info("Writing predictions to disk")
        return predictions, labels

    def predict(self, dataset_class: Any) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_on(dataset_class, split="test")

    def evaluate_on_val(self) -> Tuple[np.ndarray, np.ndarray]:
        data_loader = DataLoader(self._val_set, batch_size=self.batch_size, shuffle=False)
        predictions, labels = [], []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self._model(data)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred)
                labels.append(target.numpy())
        if predictions:
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
        else:
            predictions = np.array([])
            labels = np.array([])
        return predictions, labels
