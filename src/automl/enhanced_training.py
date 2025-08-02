# enhanced_training.py
# Enhanced version of your training.py with CO2 tracking integration

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

# Import our CO2 tracking components
from co2_tracker import CO2Tracker, EnhancedVisualizer

logger = logging.getLogger(__name__)

class EnhancedAutoML:
    """
    Enhanced AutoML class with integrated CO2 tracking and visualization.
    Drop-in replacement for your existing AutoML class.
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
        # New CO2 tracking parameters
        co2_tracker: CO2Tracker = None,
        visualizer: EnhancedVisualizer = None,
        enable_co2_tracking: bool = True,
        region: str = "DE"
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
        
        # CO2 tracking setup
        self.enable_co2_tracking = enable_co2_tracking
        if self.enable_co2_tracking:
            self.co2_tracker = co2_tracker or CO2Tracker(region=region, tracking_interval=1.0)
            self.visualizer = visualizer or EnhancedVisualizer(self.co2_tracker)
        else:
            self.co2_tracker = None
            self.visualizer = None
            
        # Training metrics for enhanced visualization
        self.detailed_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'learning_rates': [], 'co2_per_epoch': [], 'power_per_epoch': [],
            'epoch_times': [], 'gpu_memory': []
        }
        
        self._set_seed()

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

    def _log_gpu_memory(self):
        """Log GPU memory usage if available."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0

    def _log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """Log detailed metrics for each epoch."""
        if self.enable_co2_tracking and self.co2_tracker:
            co2_stats = self.co2_tracker.get_current_stats()
            self.detailed_history['co2_per_epoch'].append(co2_stats.get('total_co2_g', 0))
            self.detailed_history['power_per_epoch'].append(co2_stats.get('avg_power_w', 0))
        else:
            self.detailed_history['co2_per_epoch'].append(0)
            self.detailed_history['power_per_epoch'].append(0)
            
        self.detailed_history['train_loss'].append(train_loss)
        self.detailed_history['train_acc'].append(train_acc)
        self.detailed_history['val_loss'].append(val_loss)
        self.detailed_history['val_acc'].append(val_acc)
        self.detailed_history['epoch_times'].append(epoch_time)
        self.detailed_history['gpu_memory'].append(self._log_gpu_memory())
        
        # Log to visualizer if available
        if self.visualizer:
            self.visualizer.log_training_step(
                epoch, 
                {
                    'loss': train_loss, 'accuracy': train_acc, 
                    'epoch_time': epoch_time, 'gpu_memory': self._log_gpu_memory()
                }, 
                'train'
            )
            self.visualizer.log_training_step(
                epoch, 
                {'loss': val_loss, 'accuracy': val_acc}, 
                'validation'
            )

    def fit(self, dataset_class: Any, subsample: int = None, trial: optuna.trial.Trial = None) -> "EnhancedAutoML":
        """Enhanced fit method with CO2 tracking."""
        
        # Start CO2 tracking for this training session
        if self.enable_co2_tracking and self.co2_tracker:
            self.co2_tracker.set_phase(f"training_{self.backbone}")
            
        import time
        training_start_time = time.time()

        mean, std = calculate_mean_std(dataset_class)
        self._transform = get_transforms(mean, std, phase="train", backbone_name=self.backbone)

        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        
        if subsample is not None:
            indices = np.random.choice(len(dataset), subsample, replace=False)
            dataset = Subset(dataset, indices)
            
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(self.seed))
        self._val_set = val_set
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
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

        criterion = nn.CrossEntropyLoss()
        
        # Enhanced history tracking
        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        
        best_val_acc = 0.0
        patience = 5
        wait = 0
        best_model_state = None

        print(f"🚀 Starting training with {len(train_set)} train samples, {len(val_set)} val samples")
        print(f"🔧 Backbone: {self.backbone}, Optimizer: {self.optimizer}, LR: {self.lr}")
        
        if self.enable_co2_tracking:
            print(f"🌱 CO2 tracking enabled - Region: {self.co2_tracker.region}")

        model.train()
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Training phase
            if self.enable_co2_tracking and self.co2_tracker:
                self.co2_tracker.set_phase(f"epoch_{epoch}_train")
                
            loss_per_batch = []
            all_preds = []
            all_targets = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
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
            if self.enable_co2_tracking and self.co2_tracker:
                self.co2_tracker.set_phase(f"epoch_{epoch}_val")
                
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
            
            epoch_time = time.time() - epoch_start_time
            
            # Log detailed metrics
            self._log_epoch_metrics(epoch, epoch_loss, epoch_acc, val_loss, val_acc, epoch_time)
            
            # Enhanced logging with CO2 info
            co2_info = ""
            if self.enable_co2_tracking and self.co2_tracker:
                co2_stats = self.co2_tracker.get_current_stats()
                co2_info = f", CO2: {co2_stats.get('total_co2_g', 0):.1f}g, Power: {co2_stats.get('avg_power_w', 0):.1f}W"
            
            logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s{co2_info}")
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s{co2_info}")
            
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
        
        training_time = time.time() - training_start_time
        
        # Final training summary
        if self.enable_co2_tracking and self.co2_tracker:
            final_co2_stats = self.co2_tracker.get_current_stats()
            print(f"🏁 Training completed! Time: {training_time:.1f}s, "
                  f"Total CO2: {final_co2_stats.get('total_co2_g', 0):.1f}g, "
                  f"Best Val Acc: {best_val_acc:.4f}")

        if trial:
            trial.set_user_attr("history", self._history)
            trial.set_user_attr("detailed_history", self.detailed_history)
            if self.enable_co2_tracking:
                trial.set_user_attr("co2_stats", self.co2_tracker.get_current_stats())

        return self

    def predict_on(self, dataset_class: Any, split="test") -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced predict method with CO2 tracking."""
        if self.enable_co2_tracking and self.co2_tracker:
            self.co2_tracker.set_phase(f"inference_{split}")
            
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
        logger.info("Writing predictions to disk")
        
        return predictions, labels
    
    def predict(self, dataset_class: Any) -> np.ndarray:
        preds, labels = self.predict_on(dataset_class, split="test")
        return preds, labels
    
    def evaluate_on_val(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced validation evaluation with CO2 tracking."""
        if self.enable_co2_tracking and self.co2_tracker:
            self.co2_tracker.set_phase("validation_eval")
            
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
    
    def create_training_visualizations(self, save_path: str = "training_analysis.png"):
        """Create detailed training process visualizations."""
        if not self.detailed_history['train_loss']:
            print("No training history available for visualization")
            return
            
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(len(self.detailed_history['train_loss']))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.detailed_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.detailed_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('📉 Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.detailed_history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.detailed_history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('📈 Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # CO2 emissions per epoch
        if self.enable_co2_tracking:
            axes[0, 2].plot(epochs, self.detailed_history['co2_per_epoch'], 'g-', linewidth=2)
            axes[0, 2].fill_between(epochs, self.detailed_history['co2_per_epoch'], alpha=0.3, color='green')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Cumulative CO2 (g)')
            axes[0, 2].set_title('🌱 CO2 Emissions Over Training')
            axes[0, 2].grid(alpha=0.3)
        
        # Power consumption
        if self.enable_co2_tracking:
            axes[1, 0].plot(epochs, self.detailed_history['power_per_epoch'], 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Average Power (W)')
            axes[1, 0].set_title('⚡ Power Consumption Per Epoch')
            axes[1, 0].grid(alpha=0.3)
        
        # Epoch timing
        axes[1, 1].bar(epochs, self.detailed_history['epoch_times'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('⏱️ Epoch Training Times')
        axes[1, 1].grid(alpha=0.3)
        
        # GPU memory usage
        if max(self.detailed_history['gpu_memory']) > 0:
            axes[1, 2].plot(epochs, self.detailed_history['gpu_memory'], 'red', linewidth=2)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('GPU Memory (GB)')
            axes[1, 2].set_title('🔧 GPU Memory Usage')
            axes[1, 2].grid(alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No GPU\nDetected', ha='center', va='center', 
                           transform=axes[1, 2].transAxes, fontsize=16)
            axes[1, 2].set_title('🔧 GPU Memory Usage')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training visualizations saved to {save_path}")
    
    def get_training_summary(self) -> dict:
        """Get comprehensive training summary including CO2 metrics."""
        if not self.detailed_history['train_loss']:
            return {}
            
        summary = {
            'final_train_acc': self.detailed_history['train_acc'][-1],
            'final_val_acc': self.detailed_history['val_acc'][-1],
            'best_val_acc': max(self.detailed_history['val_acc']),
            'total_epochs': len(self.detailed_history['train_loss']),
            'total_training_time': sum(self.detailed_history['epoch_times']),
            'avg_epoch_time': np.mean(self.detailed_history['epoch_times']),
            'backbone': self.backbone,
            'optimizer': self.optimizer,
            'learning_rate': self.lr,
            'batch_size': self.batch_size
        }
        
        if self.enable_co2_tracking and self.co2_tracker:
            co2_stats = self.co2_tracker.get_current_stats()
            summary.update({
                'total_co2_g': co2_stats.get('total_co2_g', 0),
                'avg_power_w': co2_stats.get('avg_power_w', 0),
                'co2_per_accuracy': co2_stats.get('total_co2_g', 0) / max(summary['best_val_acc'], 0.01),
                'energy_efficiency': summary['best_val_acc'] / max(co2_stats.get('total_co2_g', 1), 1)
            })
            
        return summary