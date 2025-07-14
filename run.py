from __future__ import annotations

from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np
from automl.automl import AutoML, optuna_objective
import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

import logging
from automl.vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset

logger = logging.getLogger(__name__)

def main(
    dataset_class,
    output_path: Path,
    seed: int,
    num_layers_to_freeze: int,
    learning_rate: float,
):
    logger.info("Fitting AutoML")
    print("Fitting AutoML")
    automl = AutoML(seed=seed, num_layers_to_freeze=num_layers_to_freeze, lr=learning_rate)
    automl.fit(dataset_class)
    test_preds, test_labels = automl.predict(dataset_class)
    logger.info("Writing predictions to disk")
    print("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
        print(f"Accuracy on test set: {acc}")
    else:
        logger.info(f"No test split for dataset '{dataset_class.__name__}'")
        print(f"No test split for dataset '{dataset_class.__name__}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo", action="store_true", help="Run Optuna TPE+SH hyperparameter search instead of single run.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials if --hpo is set.")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to run on.", choices=["fashion", "flowers", "emotions"])
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="The path to save the predictions to. By default this will just save to './predictions.npy'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate for optimizer.")
    parser.add_argument("--num-layers-to-freeze", type=int, default=0, help="Number of layers to freeze in the model.")
    parser.add_argument("--quiet", action="store_true", help="Whether to log only warnings and errors.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)
    logger.info(f"Running dataset {args.dataset}\n{args}")
    print(f"Running dataset {args.dataset}\n{args}")

    # 1. Select dataset class ONCE at top level
    match args.dataset:
        case "fashion":
            dataset_class = FashionDataset
        case "flowers":
            dataset_class = FlowersDataset
        case "emotions":
            dataset_class = EmotionsDataset
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")

    if args.hpo:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(),
            pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)
        )
        study.optimize(lambda trial: optuna_objective(
            trial,
            dataset_class=dataset_class,
            seed=args.seed,
            epochs=10,
            batch_size=64
        ), n_trials=args.n_trials)
        print("✅ Best trial:", study.best_trial.value)
        print("🏆 Best hyperparameters:", study.best_trial.params)

        # OPTIONAL: Retrain on best config and save predictions
        best_params = study.best_trial.params
        automl = AutoML(
            seed=args.seed,
            num_layers_to_freeze=best_params.get("num_layers_to_freeze", 0),
            lr=best_params.get("lr", 0.001),
            use_augmentation=best_params.get("use_augmentation", True),
            backbone=best_params.get("backbone", "resnet18"),
            batch_size=64,
            epochs=10
        )
        automl.fit(dataset_class)
        test_preds, test_labels = automl.predict(dataset_class)
        with args.output_path.open("wb") as f:
            np.save(f, test_preds)
        print(f"Predictions for best config saved to {args.output_path}")
        if not np.isnan(test_labels).any():
            acc = accuracy_score(test_labels, test_preds)
            print(f"Accuracy of best config on test set: {acc}")

    else:
        main(
            dataset_class=dataset_class,
            output_path=args.output_path,
            seed=args.seed,
            num_layers_to_freeze=args.num_layers_to_freeze,
            learning_rate=args.learning_rate
        )
