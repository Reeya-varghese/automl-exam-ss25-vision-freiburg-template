import torch
import random
import numpy as np
import logging
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score

from automl.dummy_model import get_resnet_model
from automl.utils import calculate_mean_std

logger = logging.getLogger(__name__)

class AutoML:
    def __init__(
        self,
        seed: int,
        lr: float = 0.001,
        batch_size: int = 32,
        epochs: int = 8,
        optimizer_name: str = "adam",
    ) -> None:
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self._model: nn.Module | None = None

    def fit(self, dataset_class: any) -> "AutoML":
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        mean, std = calculate_mean_std(dataset_class)
        tfs = [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        self._transform = transforms.Compose(tfs)

        # 2000 sample subset, 80/20 split
        full_dataset = dataset_class(root="./data", split="train", download=True, transform=self._transform)
        indices = np.random.choice(len(full_dataset), 2000, replace=False)
        subset = Subset(full_dataset, indices)
        train_len = int(0.8 * 2000)
        val_len = 2000 - train_len
        train_set, val_set = random_split(subset, [train_len, val_len], generator=torch.Generator().manual_seed(self.seed))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        model = get_resnet_model(
            "resnet18",  # fixed backbone
            num_classes=dataset_class.num_classes,
            num_layers_to_freeze=0,   # fully fine-tuned
            grayscale=(dataset_class.channels == 1)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr) if self.optimizer_name == "adam" else optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        
        # History for plotting
        self._history = {"loss": [], "acc": [], "val_acc": []}

        model.train()
        for epoch in range(self.epochs):
            loss_per_batch = []
            all_preds = []
            all_targets = []
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
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

            # Validation
            val_preds = []
            val_targets = []
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = torch.argmax(output, 1).cpu().numpy()
                    val_preds.extend(pred)
                    val_targets.extend(target.cpu().numpy())
            val_acc = accuracy_score(val_targets, val_preds)
            self._history["val_acc"].append(val_acc)
            logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        self._model = model.eval()
        self.device = device
        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = calculate_mean_std(dataset_class)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = dataset_class(root="./data", split="test", download=True, transform=test_transform)
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

def optuna_objective(trial, dataset_class, seed=42):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    automl = AutoML(
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        epochs=8,  # always 8 per trial during HPO
        optimizer_name=optimizer_name
    )
    automl.fit(dataset_class)
    preds, labels = automl.predict(dataset_class)
    acc = accuracy_score(labels, preds) if not np.isnan(labels).any() else 0
    trial.set_user_attr("history", automl._history)
    return acc

