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
from torch.utils.data import Subset
from RandAug import prepare_augmented_balanced_dataset
from DAC import DynamicAdjustmentController

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
      
        custom_head: nn.Module = None ,
        optimizer = 'adam'
    ) -> None:
        self.seed = seed
        self.num_layers_to_freeze = num_layers_to_freeze
        self.optimizer = optimizer
        self.lr = lr
        self.dac = None
        self.custom_head = custom_head
        self.backbone = backbone
        self.epochs = epochs
        
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def fit(self, dataset_class: Any, subsample: int = None, trial: optuna.trial.Trial = None) -> "AutoML":
      
        mean, std = calculate_mean_std(dataset_class)
        self.mean, self.std = mean, std  # Save for predict usage

#        Load raw dataset without transforms
        raw_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=None
        )

        if self.use_augmentation:
            n = trial.suggest_int("randaug_n", 1, 3) if trial else 2
            m = trial.suggest_int("randaug_m", 5, 15) if trial else 9

            dataset, sampler = prepare_augmented_balanced_dataset(
                dataset=raw_dataset,
                image_size=(224, 224),
                grayscale=(dataset_class.channels == 1),
                n=n,
                m=m,
                mean=mean,
                std=std,
                backbone_name=self.backbone
            )

        # Apply same transform before splitting
        if subsample is not None:
            indices = np.random.choice(len(dataset), subsample, replace=False)
            dataset = Subset(dataset, indices)

        # Split dataset into training and validation sets
        total_indices = list(range(len(dataset)))
        np.random.seed(self.seed)
        np.random.shuffle(total_indices)
        split = int(0.8 * len(dataset))
        train_indices = total_indices[:split]
        val_indices = total_indices[split:]

        train_raw = Subset(raw_dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        self._val_set = val_set

        # Augment only training split
        train_augmented, sampler = prepare_augmented_balanced_dataset(
            dataset=train_raw,
            image_size=(224, 224),
            grayscale=(dataset_class.channels == 1),
            n=n,
            m=m,
            mean=mean,
            std=std,
            backbone_name=self.backbone
        )

        train_loader = DataLoader(train_augmented, batch_size=self.batch_size, sampler=sampler)

        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        model = get_model(
            self.backbone,
            num_classes=dataset_class.num_classes,
            grayscale=(dataset_class.channels == 1),
            custom_head=self.custom_head
        ).to(self.device)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer:
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        # Enable DAC for datasets    
        if dataset_class.__name__ == ["SkinCancerDataset","EmotionsDataset","FlowersDataset","FashionDataset"]:
            self.dac = DynamicAdjustmentController(optimizer, initial_lr=self.lr)
        
        # Training Loop
        criterion = nn.CrossEntropyLoss()
        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
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

            # Validation Loop
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
          
                
            logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if self.dac:
                self.dac.update(epoch_loss)
                optimizer = self.dac.get_optimizer()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                
            model.train()

        # Save the Best Model State

        if best_model_state:
            model.load_state_dict(best_model_state)
            model = model.to(self.device) 
        self._model = model.eval()  
    

        if trial:
            trial.set_user_attr("history", self._history)
            trial.set_user_attr("randaug_n", n)
            trial.set_user_attr("randaug_m", m)

        return self


    def predict_on(self, dataset_class: Any, split="test") -> Tuple[np.ndarray, np.ndarray]:
        # Predict Labels on Test set.

        self._model= self._model.to(self.device).eval()
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
        
        preds, labels= self.predict_on(dataset_class, split="test")

        return preds, labels
    
    def evaluate_on_val(self) -> Tuple[np.ndarray, np.ndarray]:
        self._model=self._model.to(self.device).eval()

        data_loader = DataLoader(self._val_set, batch_size=100, shuffle=False)
        predictions, labels = [], []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self._model(data)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                predictions.append(pred)
                labels.append(target.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(labels)
 
