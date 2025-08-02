"""
Enhanced Training Module with SMOTE Integration

This module extends the original training.py with optional SMOTE balancing
while maintaining full backward compatibility with existing code.

Usage:
- Import this instead of the original training module
- Enable SMOTE by setting use_smote=True in AutoML constructor
- All existing functionality remains unchanged when use_smote=False
"""

import logging
from typing import Any, Tuple
import torch
from copy import deepcopy 
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import random
import optuna
from model import get_model, get_transforms
from utils import calculate_mean_std
from torch.utils.data import Subset, random_split

# Import SMOTE functionality
from smote_balancer import SMOTEBalancer, SMOTEDatasetWrapper, check_imbalanced_learn_availability

logger = logging.getLogger(__name__)

class AutoML:
    """
    Enhanced AutoML class with optional SMOTE balancing for class imbalance handling.
    
    New parameters:
    - use_smote: Enable SMOTE balancing (default: False for backward compatibility)
    - smote_max_samples: Maximum samples for SMOTE (default: 2000 for memory efficiency)
    - smote_strategy: SMOTE sampling strategy (default: 'auto')
    - detect_imbalance: Automatically detect and report class imbalance (default: True)
    """
    
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
        optimizer = 'adam',
        # New SMOTE parameters
        use_smote: bool = False,
        smote_max_samples: int = 2000,
        smote_strategy: str = 'auto',
        detect_imbalance: bool = True
    ) -> None:
        """
        Initialize AutoML with optional SMOTE balancing.
        
        Args:
            seed: Random seed for reproducibility
            num_layers_to_freeze: Number of backbone layers to freeze
            lr: Learning rate
            use_augmentation: Use data augmentation
            backbone: Backbone architecture name
            batch_size: Training batch size
            epochs: Number of training epochs
            custom_head: Custom classifier head
            optimizer: Optimizer type ('adam' or 'sgd')
            use_smote: Enable SMOTE balancing for class imbalance
            smote_max_samples: Maximum samples for SMOTE (memory constraint)
            smote_strategy: SMOTE sampling strategy
            detect_imbalance: Automatically detect and log class imbalance
        """
        # Original parameters
        self.seed = seed
        self.num_layers_to_freeze = num_layers_to_freeze
        self.optimizer = optimizer
        self.lr = lr
        self.custom_head = custom_head
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        
        # New SMOTE parameters
        self.use_smote = use_smote
        self.smote_max_samples = smote_max_samples
        self.smote_strategy = smote_strategy
        self.detect_imbalance = detect_imbalance
        
        # State variables
        self._model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._imbalance_info = None
        
        # Check SMOTE availability
        if self.use_smote and not check_imbalanced_learn_availability():
            logger.warning("SMOTE requested but imbalanced-learn not available. Disabling SMOTE.")
            logger.warning("Install with: pip install imbalanced-learn")
            self.use_smote = False
        
        self._set_seed()

    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return self._history
    
    @property
    def imbalance_info(self):
        """Get information about dataset class imbalance."""
        return self._imbalance_info

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_dataset(self, dataset_class: Any, split: str = 'train', subsample: int = None) -> Any:
        """
        Create dataset with optional SMOTE balancing.
        
        Args:
            dataset_class: Dataset class to instantiate
            split: Dataset split ('train', 'test')
            subsample: Number of samples to subsample (applied before SMOTE)
        
        Returns:
            Dataset (potentially SMOTE-balanced)
        """
        # Calculate transforms
        mean, std = calculate_mean_std(dataset_class)
        transform = get_transforms(mean, std, phase="train" if split == 'train' else "test", backbone_name=self.backbone)
        
        # Create base dataset
        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=transform
        )
        
        # Apply subsampling if requested (before SMOTE to save memory)
        if subsample is not None:
            indices = np.random.choice(len(dataset), subsample, replace=False)
            dataset = Subset(dataset, indices)
            logger.info(f"Subsampled dataset to {subsample} samples")
        
        # Only apply SMOTE to training data
        if split == 'train':
            # Analyze class balance
            if self.detect_imbalance:
                self._imbalance_info = SMOTEBalancer.analyze_dataset_balance(dataset)
                logger.info(f"Dataset balance analysis: {self._imbalance_info}")
            
            # Apply SMOTE if enabled and imbalanced
            if self.use_smote:
                if self._imbalance_info and self._imbalance_info['is_imbalanced']:
                    logger.info(f"Applying SMOTE balancing (max {self.smote_max_samples} samples)...")
                    
                    # Create SMOTE-balanced wrapper
                    dataset = SMOTEDatasetWrapper(
                        original_dataset=dataset,
                        max_samples=self.smote_max_samples,
                        smote_strategy=self.smote_strategy,
                        random_state=self.seed,
                        enable_smote=True
                    )
                    
                    logger.info(f"SMOTE balancing completed. New dataset size: {len(dataset)}")
                else:
                    logger.info("Dataset is balanced or SMOTE not needed")
            else:
                if self._imbalance_info and self._imbalance_info['is_imbalanced']:
                    logger.warning(f"Class imbalance detected (ratio: {self._imbalance_info['imbalance_ratio']:.2f}) but SMOTE is disabled")
        
        return dataset

    def fit(self, dataset_class: Any, subsample: int = None, trial: optuna.trial.Trial = None) -> "AutoML":
        """
        Train the model with optional SMOTE balancing.
        
        Args:
            dataset_class: Dataset class to use for training
            subsample: Number of samples to use (applied before SMOTE)
            trial: Optuna trial for hyperparameter optimization
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting training with SMOTE={'enabled' if self.use_smote else 'disabled'}")
        
        # Create training dataset (potentially with SMOTE)
        dataset = self._create_dataset(dataset_class, split='train', subsample=subsample)
        
        # Split into train/validation
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(
            dataset, 
            [train_len, val_len], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        self._val_set = val_set
        
        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        # Create model
        model = get_model(
            self.backbone,
            num_classes=dataset_class.num_classes,
            grayscale=(dataset_class.channels == 1),
            custom_head=self.custom_head
        ).to(self.device)

        # Setup optimizer
        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()

        # Training history
        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        
        # Early stopping parameters
        best_val_acc = 0.0
        patience = 5
        wait = 0
        best_model_state = None

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            # Training phase
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

            # Validation phase
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

            # Logging
            log_msg = f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            if self.use_smote and self._imbalance_info:
                log_msg += f" [SMOTE: ratio {self._imbalance_info['imbalance_ratio']:.2f}]"
            
            logger.info(log_msg)
            print(log_msg)
            
            # Early stopping
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

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
       
        self._model = model.eval()
        self.device = self.device

        # Store trial history
        if trial:
            trial_history = self._history.copy()
            # Add SMOTE info to trial
            if self.use_smote and self._imbalance_info:
                trial_history['smote_applied'] = True
                trial_history['imbalance_ratio'] = self._imbalance_info['imbalance_ratio']
                trial_history['original_class_counts'] = self._imbalance_info['class_counts']
            else:
                trial_history['smote_applied'] = False
            
            trial.set_user_attr("history", trial_history)

        return self

    def predict_on(self, dataset_class: Any, split="test") -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on specified dataset split.
        Note: SMOTE is never applied to test data.
        """
        # Create test dataset (no SMOTE for test data)
        mean, std = calculate_mean_std(dataset_class)
        test_transform = get_transforms(mean, std, phase="test", backbone_name=self.backbone)

        dataset = dataset_class(
            root="./data",
            split=split,
            download=True,
            transform=test_transform
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
        logger.info("Predictions completed")
        
        return predictions, labels
    
    def predict(self, dataset_class: Any) -> np.ndarray:
        """Make predictions on test split."""
        preds, labels = self.predict_on(dataset_class, split="test")
        return preds, labels
    
    def evaluate_on_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate on validation set."""
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

    def get_training_summary(self) -> dict:
        """
        Get a summary of the training process including SMOTE information.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            'backbone': self.backbone,
            'epochs_trained': len(self._history['loss']) if hasattr(self, '_history') else 0,
            'final_train_acc': self._history['acc'][-1] if hasattr(self, '_history') and self._history['acc'] else None,
            'final_val_acc': self._history['val_acc'][-1] if hasattr(self, '_history') and self._history['val_acc'] else None,
            'smote_enabled': self.use_smote,
        }
        
        if self._imbalance_info:
            summary.update({
                'imbalance_detected': self._imbalance_info['is_imbalanced'],
                'imbalance_ratio': self._imbalance_info['imbalance_ratio'],
                'num_classes': self._imbalance_info['num_classes'],
                'class_distribution': self._imbalance_info['class_counts']
            })
        
        return summary