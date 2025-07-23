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
import random
from Zero_cost import ZeroCostCandidateGenerator
from training import AutoML
import optuna
from optuna.samplers import NSGAIIISampler

from Plots import (
    save_optuna_visualizations,
    save_accuracy_histogram,
    save_metric_curves
)
from model import get_model
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset
from torch.utils.data import Subset, random_split
# ---------------------------------------------
logger = logging.getLogger(__name__)

def optuna_objective(
    trial: optuna.Trial,
    dataset_class: Any,
    seed: int = 42,
    top_k_candidates: list[dict[str, Any]] = None
    ) -> Tuple[float, float, float]:
    candidate_lookup = {
        f"{c['backbone']}_{i}": (c['backbone'], c['head'])
        for i, c in enumerate(top_k_candidates)
    }
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
    backbone, head = candidate_lookup[candidate_id]
    use_augmentation = trial.suggest_categorical('use_augmentation', [True])
    epochs = 8  
    automl = AutoML(
        seed=seed,
        num_layers_to_freeze=0,
        lr=lr,
        use_augmentation=use_augmentation,
        backbone=backbone,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        custom_head=head
    )
    start = time.time()
    automl.fit(dataset_class, subsample=2000, trial=trial)
    training_time = time.time() - start
    logger.info(f"Training time: {training_time:.2f} seconds")
    trial.set_user_attr("head_type", head.__class__.__name__)
    trial.set_user_attr("backbone", backbone)

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
    
    mean, std = calculate_mean_std(dataset_class)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load and split dataset
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    # Run Zero-Cost Proxy search
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10, num_classes=dataset_class.num_classes)
    top_k_candidates = zcc.get_top_k_candidates()
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")

    print("✅ Top-K candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        print(f"[{i+1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}")
    
    # Reference points for NSGAIII
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
   

    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
        
    )
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
    ), n_trials=args.n_trials)

    pareto_trials = study.best_trials
    print(f"\n✅Pareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        print(f"✅Accuracy: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s | Params: {t.params}")

    # Retrain AutoML with best config and save predictions
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]
    automl = AutoML(
        seed=args.seed,
        num_layers_to_freeze=0,
        lr=best_params.get("lr", 0.001),
        use_augmentation=True,
        backbone=backbone,
        batch_size=best_params.get("batch_size", 32),
        epochs=final_epochs,
        optimizer=best_params.get("optimizer", "adam"),
        custom_head=head
    )
    automl.fit(dataset_class, subsample=None)
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
    print("✅AutoML training completed successfully!")
    
    # Save Optuna plots
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    print("✅Optuna visualizations saved successfully!")