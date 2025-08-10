import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os
import random
from Zero_cost import ZeroCostCandidateGenerator
from training import AutoML
import optuna
from optuna.samplers import NSGAIIISampler
import traceback

from model import get_transforms
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from torch.utils.data import random_split

warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("codecarbon").propagate = False

logger = logging.getLogger(__name__)


class CarbonBudgetManager:
    """
    Tracks global budget and guides architecture choice (unchanged logic),
    but now we track and compare **adjusted** emissions to keep the sum <= total budget.
    """
    def __init__(self, total_budget_kg=0.15, total_trials=100):
        self.total_budget = float(total_budget_kg)
        self.total_trials = int(total_trials)
        self.used_budget_adjusted = 0.0  # sum of adjusted kgCO2
        self.trial_count = 0
        self.architecture_costs = {
            'resnet18': 0.001,
            'efficientnet_b0': 0.0015,
            'vit_base_patch16_224': 0.003
        }
        self.trial_history = []

    def get_architecture_probability(self, trial_number):
        self.trial_count = trial_number
        remaining_trials = max(1, self.total_trials - trial_number)
        remaining_budget = max(0.0, self.total_budget - self.used_budget_adjusted)
        budget_per_remaining_trial = remaining_budget / remaining_trials

        print(f"[CARBON BUDGET] Trial {trial_number}: "
              f"Used(adj): {self.used_budget_adjusted:.4f}kg, "
              f"Remaining: {remaining_budget:.4f}kg, "
              f"Per trial target: {budget_per_remaining_trial:.4f}kg")

        progress_ratio = trial_number / self.total_trials

        if progress_ratio < 0.3:
            high_threshold = 0.003
            medium_threshold = 0.0015
        elif progress_ratio < 0.7:
            high_threshold = 0.0025
            medium_threshold = 0.0012
        else:
            high_threshold = 0.002
            medium_threshold = 0.001

        if budget_per_remaining_trial > high_threshold:
            probs = {'resnet18': 0.25, 'efficientnet_b0': 0.35, 'vit_base_patch16_224': 0.40}
            budget_status = "GENEROUS"
        elif budget_per_remaining_trial > medium_threshold:
            probs = {'resnet18': 0.40, 'efficientnet_b0': 0.40, 'vit_base_patch16_224': 0.20}
            budget_status = "MODERATE"
        else:
            probs = {'resnet18': 0.60, 'efficientnet_b0': 0.35, 'vit_base_patch16_224': 0.05}
            budget_status = "LOW"

        print(f"[CARBON BUDGET] Status: {budget_status}, "
              f"ViT probability: {probs['vit_base_patch16_224']:.2f}")
        return probs

    def weighted_architecture_choice(self, architectures, trial_number):
        probs = self.get_architecture_probability(trial_number)
        available_probs = {arch: probs.get(arch, 0.1) for arch in architectures if arch in probs}
        if not available_probs:
            return random.choice(architectures)
        total_prob = sum(available_probs.values())
        normalized = {k: v / total_prob for k, v in available_probs.items()}
        r = random.random()
        cum = 0.0
        for arch, p in normalized.items():
            cum += p
            if r <= cum:
                print(f"[CARBON BUDGET] Selected architecture: {arch} (prob: {p:.3f})")
                return arch
        return list(available_probs.keys())[0]

    def get_dynamic_training_config(self, backbone_name, trial_number):
        progress_ratio = trial_number / self.total_trials
        remaining_budget = max(0.0, self.total_budget - self.used_budget_adjusted)
        remaining_trials = max(1, self.total_trials - trial_number)

        base_epochs = int(6 + (progress_ratio * 6))  # 6-12 epochs
        efficiency_weights = {'resnet18': 1.0, 'efficientnet_b0': 0.8, 'vit_base_patch16_224': 0.4}
        efficiency = efficiency_weights.get(backbone_name, 0.6)

        budget_per_remaining_trial = remaining_budget / remaining_trials
        if budget_per_remaining_trial > 0.002:
            epoch_multiplier = 1.2 if efficiency < 0.6 else 1.0
        elif budget_per_remaining_trial > 0.001:
            epoch_multiplier = 1.0
        else:
            epoch_multiplier = 0.7 if efficiency < 0.6 else 0.9

        final_epochs = max(4, int(base_epochs * epoch_multiplier))
        return {
            'epochs': final_epochs,
            'early_stopping_patience': max(3, final_epochs // 3),
            'efficiency_weight': efficiency
        }

    def update_used_budget(self, adjusted_trial_cost, trial_number, backbone_name, performance_metrics=None):
        """
        Update with **adjusted** kgCO2 so the running total never exceeds the total budget.
        """
        self.used_budget_adjusted += float(adjusted_trial_cost)

        self.trial_history.append({
            'trial_number': trial_number,
            'backbone': backbone_name,
            'adjusted_carbon': float(adjusted_trial_cost),
            'cumulative_adjusted': self.used_budget_adjusted,
            'performance': performance_metrics or {}
        })

        print(f"[CARBON BUDGET] Trial {trial_number} ({backbone_name}): "
              f"Adj cost: {adjusted_trial_cost:.4f}kg, "
              f"Total used(adj): {self.used_budget_adjusted:.4f}kg / {self.total_budget:.4f}kg "
              f"({100*self.used_budget_adjusted/self.total_budget:.1f}%)")

    def remaining_adjusted_budget(self) -> float:
        return max(0.0, self.total_budget - self.used_budget_adjusted)

    def get_budget_summary(self):
        return {
            'total_budget': self.total_budget,
            'used_budget_adjusted': self.used_budget_adjusted,
            'remaining_budget': self.remaining_adjusted_budget(),
            'utilization_percent': 100 * self.used_budget_adjusted / max(1e-8, self.total_budget),
            'trial_history': self.trial_history
        }


def get_architecture_efficiency_weight(backbone_name):
    return {
        'resnet18': 1.0,
        'efficientnet_b0': 0.8,
        'resnet50': 0.6,
        'vit_base_patch16_224': 0.4
    }.get(backbone_name, 0.5)


def get_progressive_config(trial_number, total_trials, enable_progressive=True):
    if not enable_progressive:
        return {'max_epochs': 8, 'min_batch_size': 32, 'prefer_efficient_arch': False}

    progress_ratio = trial_number / max(total_trials, 1)
    if progress_ratio < 0.3:
        return {'max_epochs': 8, 'min_batch_size': 32, 'prefer_efficient_arch': True}
    elif progress_ratio < 0.7:
        return {'max_epochs': 10, 'min_batch_size': 16, 'prefer_efficient_arch': False}
    else:
        return {'max_epochs': 12, 'min_batch_size': 16, 'prefer_efficient_arch': False}


def get_enhanced_reference_points():
    import numpy as np
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0.7, 0.3, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0.4, 0.4, 0.1, 0.05, 0.05],
        [0.3, 0.3, 0.2, 0.1, 0.1],
        [0.2, 0.2, 0.15, 0.25, 0.2],
        [0.1, 0.1, 0.1, 0.35, 0.35],
    ])


def optuna_objective(
        trial: optuna.Trial,
        dataset_class: Any,
        seed: int = 42,
        top_k_candidates: list[dict[str, Any]] = None,
        carbon_budget_kg: float = 0.1,
        enable_progressive: bool = True,
        carbon_budget_manager: CarbonBudgetManager | None = None,
        total_trials: int = 10
) -> Tuple[float, float, float, float, float]:
    """
    Objective with per-trial allowance and epoch-level hard stops.
    """
    import numpy as np

    try:
        # Build candidate lookup
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }

        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

        all_candidate_ids = list(candidate_lookup.keys())
        candidate_id = trial.suggest_categorical("candidate_id", all_candidate_ids)
        backbone, head = candidate_lookup[candidate_id]

        # Carbon-manager-guided architecture override
        if carbon_budget_manager:
            available_backbones = list(set(c['backbone'] for c in top_k_candidates))
            selected_backbone_from_budget = carbon_budget_manager.weighted_architecture_choice(
                available_backbones, trial.number
            )

            if backbone != selected_backbone_from_budget:
                preferred_candidates = [cid for cid, (bb, _) in candidate_lookup.items()
                                        if bb == selected_backbone_from_budget]
                if preferred_candidates:
                    import random as _rnd
                    candidate_id = _rnd.choice(preferred_candidates)
                    backbone, head = candidate_lookup[candidate_id]
                    print(f"[CARBON BUDGET] Switched to budget-preferred: {backbone}")

            training_cfg = carbon_budget_manager.get_dynamic_training_config(backbone, trial.number)
            epochs = training_cfg['epochs']
            efficiency_weight = training_cfg['efficiency_weight']
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        else:
            prog = get_progressive_config(trial.number, total_trials, enable_progressive)
            if prog['prefer_efficient_arch'] and get_architecture_efficiency_weight(backbone) < 0.8:
                efficient_candidates = [cid for cid, (bb, _) in candidate_lookup.items()
                                        if get_architecture_efficiency_weight(bb) >= 0.8]
                if efficient_candidates:
                    import random as _rnd
                    candidate_id = _rnd.choice(efficient_candidates)
                    backbone, head = candidate_lookup[candidate_id]
                    print(f"[PROGRESSIVE] Switched to efficient architecture: {backbone}")
            epochs = trial.suggest_int('epochs', 4, prog['max_epochs'])
            efficiency_weight = get_architecture_efficiency_weight(backbone)
            batch_size = trial.suggest_categorical('batch_size',
                                                   [16, 32, 64] if prog['min_batch_size'] <= 16 else [32, 64])

        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])

        # --- NEW: per-trial allowance (adjusted) ---
        per_trial_allowance = float(carbon_budget_kg) / max(1, int(total_trials))
        # Optional: if global remaining is lower than the equal share, cap to remaining
        if carbon_budget_manager:
            per_trial_allowance = min(per_trial_allowance, carbon_budget_manager.remaining_adjusted_budget())

        print(f"[TRIAL {trial.number}] Backbone: {backbone}, Epochs: {epochs}, Batch: {batch_size}, "
              f"Per-trial CO2 allowance (adjusted): {per_trial_allowance:.6f} kg")

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
        try:
            automl.fit(
                dataset_class,
                subsample=None,
                trial=trial,
                per_trial_budget_kg=per_trial_allowance,
                efficiency_weight=efficiency_weight
            )
        except RuntimeError as e:
            # Early budget stop inside an epoch
            if "PER_TRIAL_CARBON_BUDGET_EXCEEDED" in str(e):
                training_time = time.time() - start
                emissions_kg = float(trial.user_attrs.get("epoch_level_emissions_kg", 0.0))
                adjusted_carbon = emissions_kg / max(1e-8, float(efficiency_weight))
                trial.set_user_attr("emissions_kg", emissions_kg)
                trial.set_user_attr("adjusted_carbon", adjusted_carbon)
                trial.set_user_attr("efficiency_weight", efficiency_weight)

                # Update global manager with adjusted cost so total never exceeds total budget
                if carbon_budget_manager:
                    carbon_budget_manager.update_used_budget(
                        adjusted_carbon, trial.number, backbone,
                        performance_metrics=None
                    )

                raise optuna.TrialPruned(
                    f"Per-trial carbon budget exceeded mid-epoch: "
                    f"{adjusted_carbon:.6f}kg > {per_trial_allowance:.6f}kg"
                )
            else:
                raise

        training_time = time.time() - start

        # Evaluate on validation
        preds, labels = automl.evaluate_on_val()
        if labels.size > 0:
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
        else:
            acc, f1 = 0.0, 0.0

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        # Return dominated values to push trial away from Pareto front
        return 0.0, 0.0, 999.0, 999.0, 999.0

    # Read epoch-level emissions recorded by training
    emissions_kg = float(trial.user_attrs.get("epoch_level_emissions_kg", 0.0))
    adjusted_carbon = emissions_kg / max(1e-8, float(efficiency_weight))
    trial.set_user_attr("emissions_kg", emissions_kg)
    trial.set_user_attr("adjusted_carbon", adjusted_carbon)
    trial.set_user_attr("efficiency_weight", efficiency_weight)

    # Update global adjusted usage
    if carbon_budget_manager:
        carbon_budget_manager.update_used_budget(
            adjusted_carbon, trial.number, backbone,
            performance_metrics={'accuracy': acc, 'f1': f1}
        )

    # Final guard (should be redundant due to epoch-level check)
    if adjusted_carbon > per_trial_allowance:
        raise optuna.TrialPruned(
            f"Per-trial carbon budget exceeded post-run: "
            f"{adjusted_carbon:.6f}kg > {per_trial_allowance:.6f}kg"
        )

    # Return 5 objectives: accuracy, f1, time, adjusted_emissions, gpu_memory (0.0 placeholder)
    return acc, f1, training_time, adjusted_carbon, 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fashion", "flowers", "emotions", "skin_cancer"], )
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--carbon-budget", type=float, default=0.15, help="Total carbon budget in kg CO2eq")
    parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
    parser.add_argument("--enable-carbon-manager", action="store_true", help="Enable carbon budget manager")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Dataset choice
    if args.dataset == "fashion":
        dataset_class = FashionDataset
    elif args.dataset == "flowers":
        dataset_class = FlowersDataset
    elif args.dataset == "emotions":
        dataset_class = EmotionsDataset
    elif args.dataset == "skin_cancer":
        dataset_class = SkinCancerDataset
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    print(f"Enhanced AutoML (CO₂-aware) - {args.dataset.upper()}")
    print(f"Total Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
    print(f"Carbon Budget Manager: {'Enabled' if args.enable_carbon_manager else 'Disabled'}")

    # Initialize carbon budget manager
    carbon_budget_manager = None
    if args.enable_carbon_manager:
        carbon_budget_manager = CarbonBudgetManager(
            total_budget_kg=args.carbon_budget,
            total_trials=args.n_trials
        )
        print(f"Carbon Budget Manager initialized with {args.carbon_budget}kg budget "
              f"for {args.n_trials} trials")

    # Build a tiny sample for zero-cost proxies to evaluate heads
    mean, std = calculate_mean_std(dataset_class)
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="test", backbone_name=default_backbone)

    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len],
                                generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    # Zero-Cost Proxy search for top-k heads
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10,
                                     num_classes=dataset_class.num_classes)
    top_k_candidates = [c for c in zcc.get_top_k_candidates() if
                        c['backbone'] in ['resnet18', 'efficientnet_b0', 'vit_base_patch16_224']]
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    print(f"Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        eff = get_architecture_efficiency_weight(c['backbone'])
        print(f"[{i + 1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {eff:.1f}")

    # NSGA-III setup (5 objectives)
    ref_points = get_enhanced_reference_points()
    sampler = NSGAIIISampler(
        population_size=50,
        mutation_prob=0.15,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=ref_points
    )

    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    study.optimize(lambda t: optuna_objective(
        t,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
        carbon_budget_kg=args.carbon_budget,
        enable_progressive=args.enable_progressive,
        carbon_budget_manager=carbon_budget_manager,
        total_trials=args.n_trials
    ), n_trials=args.n_trials)

    # Budget summary
    if carbon_budget_manager:
        summary = carbon_budget_manager.get_budget_summary()
        print("\nCARBON BUDGET SUMMARY:")
        print(f"   Total Budget (adjusted): {summary['total_budget']:.4f} kg CO2eq")
        print(f"   Used (adjusted): {summary['used_budget_adjusted']:.4f} kg CO2eq")
        print(f"   Remaining: {summary['remaining_budget']:.4f} kg CO2eq")
        print(f"   Utilization: {summary['utilization_percent']:.1f}%")

    # Report Pareto front
    pareto_trials = study.best_trials
    print(f"\nPareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        eff = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        print(f"Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
              f"Carbon(raw): {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), GPU: {t.values[4]:.2f}, "
              f"Eff: {eff:.1f} | {t.params}")
