import argparse
import logging
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score


import optuna
from optuna.samplers import NSGAIIISampler
from optuna.pruners import SuccessiveHalvingPruner

# --- Replace with your actual import paths ---
from dummy_model import get_resnet_model
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset
from torch.utils.data import Subset, random_split
# ---------------------------------------------

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
        epochs: int = 10,
        optimizer = 'adam',
    ) -> None:
        self.seed = seed
        self.num_layers_to_freeze = num_layers_to_freeze
        self.optimizer = optimizer
        self.lr = lr
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._model: nn.Module | None = None

    def fit(self, dataset_class: Any) -> "AutoML":
        import random
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

       
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        indices = np.random.choice(len(dataset), 2000, replace=False)
        subset = Subset(dataset, indices)
        train_len = int(0.8 * 2000)
        val_len = 2000 - train_len
        train_set, val_set = random_split(subset, [train_len, val_len], generator=torch.Generator().manual_seed(self.seed))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        model = get_resnet_model(
            self.backbone,
            num_classes=dataset_class.num_classes,
            num_layers_to_freeze=self.num_layers_to_freeze,
            grayscale=(dataset_class.channels == 1)
        ).to(device)
        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer:
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()

        self._history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
      
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

            val_loss_per_batch = []
            val_preds = []
            val_targets = []
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
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
            model.train()
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

        dataset = dataset_class(
        root="./data",
        split='test',
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
        return predictions, labels

def optuna_objective(trial, dataset_class, seed=42):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    backbone = "resnet18"  # Fixed as per screenshot
    use_augmentation = True
    epochs = 8  
    automl = AutoML(
        seed=seed,
        num_layers_to_freeze=0,
        lr=lr,
        use_augmentation=use_augmentation,
        backbone=backbone,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer
    )
    start = time.time()
    automl.fit(dataset_class)
    training_time = time.time() - start
    logger.info(f"Training time: {training_time:.2f} seconds")

    preds, labels = automl.predict(dataset_class)

    if not np.isnan(labels).any():
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
    else:
        acc = 0
        f1 = 0
    return acc, f1, training_time    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True, choices=["fashion", "flowers", "emotions"])
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Dataset selection
    if args.dataset == "fashion":
        dataset_class = FashionDataset
    elif args.dataset == "flowers":
        dataset_class = FlowersDataset
    elif args.dataset == "emotions":
        dataset_class = EmotionsDataset
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    

    reference_points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1/3, 1/3, 1/3]
    ])

    # GA + SH Hyperparameter Optimization
    sampler = NSGAIIISampler(
        population_size=40,
        mutation_prob=0.2,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )
    pruner = SuccessiveHalvingPruner()

    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
    ), n_trials=args.n_trials)

    pareto_trials = study.best_trials
    print(f"\n✅Pareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        print(f"✅Accuracy: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s | Params: {t.params}")

    # Retrain AutoML with best config and save predictions
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    automl = AutoML(
        seed=args.seed,
        num_layers_to_freeze=0,
        lr=best_params.get("lr", 0.001),
        use_augmentation=True,
        backbone="resnet18",
        batch_size=best_params.get("batch_size", 32),
        epochs=final_epochs,
        optimizer=best_params.get("optimizer", "adam"),
    )
    automl.fit(dataset_class)
    test_preds, test_labels = automl.predict(dataset_class)
    with args.output_path.open("wb") as f:
        np.save(f, test_preds)
    print(f"✅Predictions for best-accuracy config saved to {args.output_path}")
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"✅Accuracy of best config on test set: {acc:.4f} (F1: {f1:.4f})")
    else:
        print(f"No test split for dataset '{dataset_class.__name__}'")

# --- Visualizations ---
try:
    import optuna.visualization
    import matplotlib.pyplot as plt

    fig = optuna.visualization.plot_optimization_history(
        study,
        target=lambda t: t.values[0],   # 0 = accuracy
        arget_name="Accuracy"
    )
    fig.write_html("optuna_optimization_history.html")
    print("✅ Optuna optimization history plot saved as optuna_optimization_history.html")

    # Pareto front
    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=["Accuracy", "F1", "Training Time"],
        include_dominated_trials=False
    )
    fig.write_html("pareto_front.html")
    print("✅ Pareto front plot saved as pareto_front.html")

    # Hyperparameter importance (accuracy)
    fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="Accuracy")
    fig.write_html("param_importance.html")
    print("✅ Parameter importance plot saved as param_importance.html")

    # Parallel coordinate plot
    fig = optuna.visualization.plot_parallel_coordinate(
        study,
        target=lambda t: t.values[0],
        target_name="Accuracy"
    )
    fig.write_html("parallel_coords.html")
    print("✅ Parallel coordinate plot saved as parallel_coords.html")

    # Accuracy histogram
    all_accs = [t.values[0] for t in study.trials if t.values is not None]
    plt.figure()
    plt.hist(all_accs, bins=20, color='skyblue')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title('Distribution of Accuracy Across Trials')
    plt.tight_layout()
    plt.savefig("accuracy_hist_EA.png")
    plt.close()
    print("✅ Accuracy histogram saved as accuracy_hist.png")

    plt.figure(figsize=(10, 5))
    for t in study.trials:
        if "history" in t.user_attrs:
            plt.plot(t.user_attrs["history"]["acc"], alpha=0.3, label='train_acc' if t == study.trials[0] else "")
            plt.plot(t.user_attrs["history"]["val_acc"], alpha=0.3, linestyle='--', label='val_acc' if t == study.trials[0] else "")
    plt.title("Accuracy per Epoch (all trials)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("trials_accuracy_EA_SH.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    for t in study.trials:
        if "history" in t.user_attrs:
            plt.plot(t.user_attrs["history"]["loss"], alpha=0.3, label='train_loss' if t == study.trials[0] else "")
            plt.plot(t.user_attrs["history"]["val_loss"], alpha=0.3, linestyle='--', label='val_loss' if t == study.trials[0] else "")
    plt.title("Loss per Epoch (all trials)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("trials_loss_EA_SH.png")
    plt.close()
    print("✅ Per-epoch train/val accuracy and loss curves saved as trials_accuracy.png and trials_loss.png")
except Exception as e:
    print("Could not create one or more plots:", e)

