import argparse
import contextlib
import logging
import warnings
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import threading
import os
import random
from Zero_cost import ZeroCostCandidateGenerator
from training import AutoML
import optuna
from optuna.samplers import NSGAIIISampler
import traceback
from Plots import (
    save_accuracy_histogram,
    save_metric_curves
)
from model import get_transforms
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from torch.utils.data import random_split

warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("codecarbon").propagate = False

logger = logging.getLogger(__name__)

class CarbonGPUTracker:
    """
    This Function, Tracks carbon emissions and GPU usage during model training using CodeCarbon.

    Args:
        project_name (str): Name used to identify the CodeCarbon tracking log.

    """

    def __init__(self, project_name="automl_carbon_tracking"):
        self.project_name = project_name
        self.emissions_tracker = None
        self.gpu_available = torch.cuda.is_available()
        self.monitoring_active = False
        self.peak_gpu_memory = 0
        self.start_time = None

    def start_tracking(self, trial_id=None):
        """Start carbon and GPU tracking"""
        self.start_time = time.time()
        self.peak_gpu_memory = 0
        self.monitoring_active = True

        # Initialize CodeCarbon tracker
        tracker_name = f"{self.project_name}_trial_{trial_id}" if trial_id else self.project_name
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            from codecarbon import EmissionsTracker
            self.emissions_tracker = EmissionsTracker(
                project_name=tracker_name,
                measure_power_secs=15,
                save_to_file=True,
                log_level="error"
            )
        self.emissions_tracker.start()

        # Reset GPU memory stats
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()

        # Start GPU monitoring
        self.gpu_monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
        self.gpu_monitor_thread.start()

    def _monitor_gpu(self):
        """Monitor GPU memory in background"""
        while self.monitoring_active:
            if self.gpu_available:
                try:
                    current_memory = torch.cuda.memory_allocated() / 1024 ** 3  # GB
                    self.peak_gpu_memory = max(self.peak_gpu_memory, current_memory)
                except Exception:
                    pass
            time.sleep(2)

    def stop_tracking(self):
        """Stop tracking and return metrics"""
        self.monitoring_active = False

        total_emissions = 0
        if self.emissions_tracker:
            try:
                total_emissions = self.emissions_tracker.stop()
            except Exception:
                total_emissions = 0

        training_time = time.time() - self.start_time if self.start_time else 0

        return {
            'emissions_kg': total_emissions,
            'training_time': training_time,
            'peak_gpu_memory_gb': self.peak_gpu_memory
        }


class CarbonBudgetManager:
    """
    this class  adjusts architecture selection 
    based on remaining budget and trial progress.
    """
    
    def __init__(self, total_budget_kg=0.15, total_trials=100):
        self.total_budget = total_budget_kg
        self.total_trials = total_trials
        self.used_budget = 0.0
        self.trial_count = 0
        self.architecture_costs = {
            'resnet18': 0.001,           # Estimated carbon cost per trial
            'efficientnet_b0': 0.0015,
            'vit_base_patch16_224': 0.003
        }
        self.trial_history = []
        
    def get_architecture_probability(self, trial_number):
        """
        Dynamically adjust architecture selection based on remaining budget.
        
        Args:
            trial_number (int): Current trial number
            
        Returns:
            dict: Architecture probabilities for weighted selection
        """
        self.trial_count = trial_number
        remaining_trials = max(1, self.total_trials - trial_number)  
        remaining_budget = max(0, self.total_budget - self.used_budget)
        
        # Calculate budget per remaining trial
        budget_per_remaining_trial = remaining_budget / remaining_trials
        
        print(f"[CARBON BUDGET] Trial {trial_number}: "
              f"Used: {self.used_budget:.4f}kg, "
              f"Remaining: {remaining_budget:.4f}kg, "
              f"Per trial: {budget_per_remaining_trial:.4f}kg")
        
        # Adaptive thresholds based on progress
        progress_ratio = trial_number / self.total_trials
        
        # Early stage: more conservative thresholds
        if progress_ratio < 0.3:
            high_threshold = 0.003
            medium_threshold = 0.0015
        # Middle stage: balanced thresholds
        elif progress_ratio < 0.7:
            high_threshold = 0.0025
            medium_threshold = 0.0012
        # Late stage: more aggressive thresholds
        else:
            high_threshold = 0.002
            medium_threshold = 0.001
        
        # Determine architecture probabilities based on budget availability
        if budget_per_remaining_trial > high_threshold:
            # favor complex models for potential high performance
            probs = {
                'resnet18': 0.25,
                'efficientnet_b0': 0.35,
                'vit_base_patch16_224': 0.40
            }
            budget_status = "GENEROUS"
        elif budget_per_remaining_trial > medium_threshold:
            # Moderate budget - balanced approach
            probs = {
                'resnet18': 0.40,
                'efficientnet_b0': 0.40,
                'vit_base_patch16_224': 0.20
            }
            budget_status = "MODERATE"
        else:
            # Low budget - prioritize efficient models
            probs = {
                'resnet18': 0.60,
                'efficientnet_b0': 0.35,
                'vit_base_patch16_224': 0.05
            }
            budget_status = "LOW"
        
        print(f"[CARBON BUDGET] Status: {budget_status}, "
              f"ViT probability: {probs['vit_base_patch16_224']:.2f}")
        
        return probs
    
    def weighted_architecture_choice(self, architectures, trial_number):
        """
        Select architecture using weighted random selection based on carbon budget.
        
        Args:
            architectures (list): Available architecture names
            trial_number (int): Current trial number
            
        Returns:
            str: Selected architecture name
        """
        probs = self.get_architecture_probability(trial_number)
        
        # Filter available architectures and their probabilities
        available_probs = {arch: probs.get(arch, 0.1) for arch in architectures if arch in probs}
        
        if not available_probs:
            # Fallback to uniform selection if no matches
            return random.choice(architectures)
        
        # Normalize probabilities
        total_prob = sum(available_probs.values())
        normalized_probs = {arch: prob/total_prob for arch, prob in available_probs.items()}
        
        # Weighted random selection
        rand_val = random.random()
        cumulative = 0.0
        for arch, prob in normalized_probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                print(f"[CARBON BUDGET] Selected architecture: {arch} (prob: {prob:.3f})")
                return arch
        
        # Fallback
        return list(available_probs.keys())[0]
    
    def get_dynamic_training_config(self, backbone_name, trial_number):
        """
        Get training configuration based on architecture efficiency and budget.
        
        Args:
            backbone_name (str): Selected backbone architecture
            trial_number (int): Current trial number
            
        Returns:
            dict: Training configuration
        """
        progress_ratio = trial_number / self.total_trials
        remaining_budget = max(0, self.total_budget - self.used_budget)
        remaining_trials = max(1, self.total_trials - trial_number)
        
        # Base epochs increase with progress
        base_epochs = int(6 + (progress_ratio * 6))  # 6 - 12 epochs over time
        
        # Architecture efficiency multiplier
        efficiency_weights = {
            'resnet18': 1.0,
            'efficientnet_b0': 0.8,
            'vit_base_patch16_224': 0.4
        }
        efficiency = efficiency_weights.get(backbone_name, 0.6)
        
        # Budget-aware epoch adjustment
        budget_per_remaining_trial = remaining_budget / remaining_trials
        if budget_per_remaining_trial > 0.002:
            # Generous budget - allow longer training for complex models
            if efficiency < 0.6:  # Complex models (ViT)
                epoch_multiplier = 1.2
            else:
                epoch_multiplier = 1.0
        elif budget_per_remaining_trial > 0.001:
            # Moderate budget - standard training
            epoch_multiplier = 1.0
        else:
            # Low budget - shorter training, especially for complex models
            if efficiency < 0.6:
                epoch_multiplier = 0.7
            else:
                epoch_multiplier = 0.9
        
        final_epochs = max(4, int(base_epochs * epoch_multiplier))
        
        return {
            'epochs': final_epochs,
            'early_stopping_patience': max(3, final_epochs // 3),
            'efficiency_weight': efficiency
        }
    
    def update_used_budget(self, trial_carbon_cost, trial_number, backbone_name, performance_metrics=None):
        """
        Update budget tracking after each trial.
        
        Args:
            trial_carbon_cost (float): Actual carbon cost of the trial
            trial_number (int): Trial number
            backbone_name (str): Architecture used
            performance_metrics (dict): Optional performance metrics for analysis
        """
        self.used_budget += trial_carbon_cost
        
        # Track trial history for analysis
        trial_record = {
            'trial_number': trial_number,
            'backbone': backbone_name,
            'carbon_cost': trial_carbon_cost,
            'cumulative_budget': self.used_budget,
            'performance': performance_metrics
        }
        self.trial_history.append(trial_record)
        
        print(f"[CARBON BUDGET] Trial {trial_number} ({backbone_name}): "
              f"Cost: {trial_carbon_cost:.4f}kg, "
              f"Total used: {self.used_budget:.4f}kg / {self.total_budget:.4f}kg "
              f"({100*self.used_budget/self.total_budget:.1f}%)")
        
        # Warn if budget is running low
        if self.used_budget > 0.8 * self.total_budget:
            remaining_trials = self.total_trials - trial_number - 1
            if remaining_trials > 0:
                print(f"[CARBON BUDGET] ⚠️ WARNING: {100*self.used_budget/self.total_budget:.1f}% "
                      f"budget used with {remaining_trials} trials remaining!")
    
    def get_budget_summary(self):
        """Get summary of budget usage"""
        return {
            'total_budget': self.total_budget,
            'used_budget': self.used_budget,
            'remaining_budget': self.total_budget - self.used_budget,
            'utilization_percent': 100 * self.used_budget / self.total_budget,
            'trial_history': self.trial_history
        }


def get_architecture_efficiency_weight(backbone_name):
    """
    Assign a predefined efficiency weight based on backbone architecture.

    Args:
        backbone_name (str): Model architecture name.
    Returns:
        float: Efficiency weight.
    """
    efficiency_weights = {
        'resnet18': 1.0,  # Most efficient
        'efficientnet_b0': 0.8,  # Good efficiency 
        'resnet50': 0.6,  # Moderate efficiency 
        'vit_base_patch16_224': 0.4  # Least efficient
    }
    return efficiency_weights.get(backbone_name, 0.5)


# Progressive training strategy.
def get_progressive_config(trial_number, total_trials, enable_progressive=True):
    """
    Return training config with progressive scaling based on trial number.

    Args:
        trial_number (int): Current Optuna trial number.
        total_trials (int): Total planned trials.
        enable_progressive (bool): Whether to use progressive strategy.
    Returns:
        dict: Training configuration.
    """
    if not enable_progressive:
        return {
            'max_epochs': 8,  # Original default
            'min_batch_size': 32,
            'prefer_efficient_arch': False
        }

    progress_ratio = trial_number / max(total_trials, 1)

    if progress_ratio < 0.3:
        return {
            'max_epochs': 8,
            'min_batch_size': 32,
            'prefer_efficient_arch': True
        }
    elif progress_ratio < 0.7:
        return {
            'max_epochs': 10,
            'min_batch_size': 16,
            'prefer_efficient_arch': False
        }
    else:
        return {
            'max_epochs': 12,
            'min_batch_size': 16,
            'prefer_efficient_arch': False
        }


def get_enhanced_reference_points():
    """
        Defines reference points for NSGA-III multi-objective optimization.

        Returns:
            np.ndarray: Reference points array.
        """
    return np.array([
        # Performance solutions
        [1, 0, 0, 0, 0],  # Pure accuracy
        [0, 1, 0, 0, 0],  # Pure F1
        [0.7, 0.3, 0, 0, 0],  # Balanced performance

        # Efficiency solutions
        [0, 0, 1, 0, 0],  # Pure speed
        [0, 0, 0, 1, 0],  # Pure carbon efficiency
        [0, 0, 0, 0, 1],  # Pure GPU efficiency

        # Balanced sustainability solutions
        [0.4, 0.4, 0.1, 0.05, 0.05],  # Performance + sustainability
        [0.3, 0.3, 0.2, 0.1, 0.1],  # Balanced all objectives
        [0.2, 0.2, 0.15, 0.25, 0.2],  # Sustainability 
        [0.1, 0.1, 0.1, 0.35, 0.35],  # Green AI 
    ])


def optuna_objective(
        trial: optuna.Trial,
        dataset_class: Any,
        seed: int = 42,
        top_k_candidates: list[dict[str, Any]] = None,
        carbon_budget_kg: float = 0.1,
        enable_progressive: bool = True,
        carbon_budget_manager: CarbonBudgetManager = None,
        total_trials: int = 10
) -> Tuple[float, float, float, float, float]:
    """
    Objective function for Optuna multi-objective optimization with carbon and GPU tracking.
    """

    tracker = CarbonGPUTracker(f"trial_{trial.number}")
    tracker.start_tracking(trial.number)

    try:
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
        
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

        # FIX: Always use the full candidate list for Optuna suggest_categorical
        # This ensures consistent choices across all trials
        all_candidate_ids = list(candidate_lookup.keys())
        candidate_id = trial.suggest_categorical("candidate_id", all_candidate_ids)
        
        # Get backbone and head from the suggested candidate
        backbone, head = candidate_lookup[candidate_id]

        # NEW: Apply carbon budget manager filtering AFTER Optuna selection
        if carbon_budget_manager:
            available_backbones = list(set(c['backbone'] for c in top_k_candidates))
            
            # Check if the selected backbone aligns with carbon budget strategy
            selected_backbone_from_budget = carbon_budget_manager.weighted_architecture_choice(
                available_backbones, trial.number
            )
            
            # If the Optuna-selected backbone doesn't match budget preference,
            # find an alternative candidate with the preferred backbone
            if backbone != selected_backbone_from_budget:
                # Find candidates with the budget-preferred backbone
                preferred_candidates = [cid for cid, (bb, _) in candidate_lookup.items() 
                                      if bb == selected_backbone_from_budget]
                
                if preferred_candidates:
                    # Randomly select from preferred candidates to maintain some diversity
                    import random
                    candidate_id = random.choice(preferred_candidates)
                    backbone, head = candidate_lookup[candidate_id]
                    print(f"[CARBON BUDGET] Switched from {backbone} to budget-preferred {selected_backbone_from_budget}")
                # If no preferred candidates available, keep the original Optuna selection
            
            # Get dynamic training configuration from carbon budget manager
            training_config = carbon_budget_manager.get_dynamic_training_config(backbone, trial.number)
            epochs = training_config['epochs']
            efficiency_weight = training_config['efficiency_weight']
            
            # Use all batch size options for carbon budget manager
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
        else:
            # ORIGINAL: Progressive configuration (kept for backward compatibility)
            progressive_config = get_progressive_config(trial.number, total_trials, enable_progressive)

            # Check if we need to enforce efficient architecture preference
            if progressive_config['prefer_efficient_arch']:
                # Check if the selected backbone is efficient enough
                if get_architecture_efficiency_weight(backbone) < 0.8:
                    # Find efficient candidates
                    efficient_candidates = [cid for cid, (bb, _) in candidate_lookup.items()
                                          if get_architecture_efficiency_weight(bb) >= 0.8]
                    if efficient_candidates:
                        # Replace with a random efficient candidate
                        import random
                        candidate_id = random.choice(efficient_candidates)
                        backbone, head = candidate_lookup[candidate_id]
                        print(f"[PROGRESSIVE] Switched to efficient architecture: {backbone}")

            # Get epochs from progressive config
            epochs = trial.suggest_int('epochs', 4, progressive_config['max_epochs'])
            efficiency_weight = get_architecture_efficiency_weight(backbone)
            
            # Progressive batch size
            batch_size_options = [16, 32, 64] if progressive_config['min_batch_size'] <= 16 else [32, 64]
            batch_size = trial.suggest_categorical('batch_size', batch_size_options)

        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])

        # Log the final selection
        print(f"[TRIAL {trial.number}] Selected: {backbone}, Epochs: {epochs}, Batch: {batch_size}")

        # Your existing AutoML training
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
        # hyperparameter tuning with Optuna on train set and val set
        automl.fit(dataset_class, subsample=None, trial=trial)
        training_time = time.time() - start
        logger.info(f"Training time: {training_time:.2f} seconds")
        trial.set_user_attr("head_type", head.__class__.__name__)
        trial.set_user_attr("backbone", backbone)

        preds, labels = automl.evaluate_on_val()

        if not np.isnan(labels).any():
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
        else:
            acc = 0
            f1 = 0

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        traceback.print_exc()
        tracker.stop_tracking()
        return 0.0, 0.0, 999.0, 999.0, 999.0

    # Get sustainability metrics
    sustainability_metrics = tracker.stop_tracking()

    # Apply architecture efficiency weighting to carbon cost
    adjusted_carbon = sustainability_metrics['emissions_kg'] / efficiency_weight

    # Update carbon budget manager
    if carbon_budget_manager:
        performance_metrics = {'accuracy': acc, 'f1': f1}
        carbon_budget_manager.update_used_budget(
            sustainability_metrics['emissions_kg'], 
            trial.number, 
            backbone, 
            performance_metrics
        )

    trial.set_user_attr("emissions_kg", sustainability_metrics['emissions_kg'])
    trial.set_user_attr("adjusted_carbon", adjusted_carbon)
    trial.set_user_attr("peak_gpu_memory_gb", sustainability_metrics['peak_gpu_memory_gb'])
    trial.set_user_attr("efficiency_weight", efficiency_weight)

    # Carbon budget constraint with efficiency weighting
    if adjusted_carbon > carbon_budget_kg:
        trial.set_user_attr("carbon_budget_exceeded", True)
        raise optuna.TrialPruned(f"Carbon budget exceeded: {adjusted_carbon:.4f} kg (efficiency-adjusted)")

    # Returns 5 objectives: accuracy, f1, time, adjusted_emissions, gpu_memory
    return (
        acc,
        f1,
        training_time,
        adjusted_carbon,
        sustainability_metrics['peak_gpu_memory_gb']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fashion", "flowers", "emotions", "skin_cancer"], )
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    # NEW: Carbon-aware parameters
    parser.add_argument("--carbon-budget", type=float, default=0.15, help="Carbon budget in kg CO2eq")
    parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
    parser.add_argument("--enable-carbon-manager", action="store_true", help="Enable carbon budget manager")
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
    elif args.dataset == "skin_cancer":
        dataset_class = SkinCancerDataset
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    print(f"Enhanced AutoML with Combined Carbon Strategies - {args.dataset.upper()}")
    print(f"Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
    print(f"Carbon Budget Manager: {'Enabled' if args.enable_carbon_manager else 'Disabled'}")

    # NEW: Initialize carbon budget manager
    carbon_budget_manager = None
    if args.enable_carbon_manager:
        carbon_budget_manager = CarbonBudgetManager(
            total_budget_kg=args.carbon_budget,
            total_trials=args.n_trials
        )
        print(f"🎯 Carbon Budget Manager initialized with {args.carbon_budget}kg budget for {args.n_trials} trials")

    mean, std = calculate_mean_std(dataset_class)

    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"

    transform = get_transforms(mean, std, phase="test", backbone_name=default_backbone)

    # Load and split dataset
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    # Run Zero-Cost Proxy search
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10,
                                     num_classes=dataset_class.num_classes)
    top_k_candidates = [c for c in zcc.get_top_k_candidates() if
                        c['backbone'] in ['resnet18', 'efficientnet_b0', 'vit_base_patch16_224']]
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    print(f"Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")

    print("Top-K candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        efficiency = get_architecture_efficiency_weight(c['backbone'])
        print(
            f"[{i + 1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {efficiency:.1f}")

    # reference points for carbon-aware optimization
    reference_points = get_enhanced_reference_points()

    # NSGA-III sampler with larger population for 5 objectives
    sampler = NSGAIIISampler(
        population_size=50,
        mutation_prob=0.15,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )

    # study with 5 objectives
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    # Global carbon tracking
    global_tracker = CarbonGPUTracker("global_optimization")
    global_tracker.start_tracking()

    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
        carbon_budget_kg=args.carbon_budget,
        enable_progressive=args.enable_progressive,
        carbon_budget_manager=carbon_budget_manager,
        total_trials=args.n_trials
    ), n_trials=args.n_trials)

    global_metrics = global_tracker.stop_tracking()

    # Print carbon budget summary if manager was used
    if carbon_budget_manager:
        budget_summary = carbon_budget_manager.get_budget_summary()
        print(f"\n📊 CARBON BUDGET SUMMARY:")
        print(f"   Total Budget: {budget_summary['total_budget']:.4f} kg CO2eq")
        print(f"   Used Budget: {budget_summary['used_budget']:.4f} kg CO2eq")
        print(f"   Remaining: {budget_summary['remaining_budget']:.4f} kg CO2eq")
        print(f"   Utilization: {budget_summary['utilization_percent']:.1f}%")
        

        backbone_usage = {}
        for record in budget_summary['trial_history']:
            backbone = record['backbone']
            backbone_usage[backbone] = backbone_usage.get(backbone, 0) + 1
        
        print(f"\n🏗️ ARCHITECTURE USAGE:")
        for arch, count in backbone_usage.items():
            percentage = 100 * count / len(budget_summary['trial_history'])
            print(f"   {arch}: {count} trials ({percentage:.1f}%)")

    pareto_trials = study.best_trials
    print(f"\nPareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        print(f"Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
              f"Carbon: {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), GPU: {t.values[4]:.2f}GB, "
              f"Eff: {efficiency:.1f} | {t.params}")

  
    print(f"\nSOLUTION ANALYSIS:")
    print("=" * 60)

    carbon_efficient = [t for t in pareto_trials if t.values[3] < args.carbon_budget * 0.5]
    high_performance = [t for t in pareto_trials if t.values[0] > 0.85 and t.values[1] > 0.8]
    balanced = [t for t in pareto_trials if t not in carbon_efficient and t not in high_performance]

    print(f"Carbon Efficient ({len(carbon_efficient)}): Low carbon footprint solutions")
    for t in carbon_efficient[:3]:
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, Carbon={t.values[3]:.4f}kg")

    print(f"High Performance ({len(high_performance)}): Best accuracy solutions")
    for t in high_performance[:3]:
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, F1={t.values[1]:.3f}")

    print(f"Balanced ({len(balanced)}): Good trade-offs")

    # select best candidate from pareto front
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]

    print(f"\nSelected Best Accuracy Solution: Trial {best_acc_trial.number}")
    print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
    print(f"   Sustainability: Carbon={best_acc_trial.values[3]:.4f}kg, GPU={best_acc_trial.values[4]:.2f}GB")

    # Final training on training and Validation
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

    final_tracker = CarbonGPUTracker("final_training")
    final_tracker.start_tracking()

    automl.fit(dataset_class, subsample=None)
    test_preds, test_labels = automl.predict(dataset_class)

    final_metrics = final_tracker.stop_tracking()

    if args.dataset == "skin_cancer":
        output_path = Path("final_test_preds.npy")
    else:
        output_path = args.output_path

    with output_path.open("wb") as f:
        np.save(f, test_preds)

    print(f"\nFINAL RESULTS:")
    print(f"Predictions saved to: {output_path}")
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"Final Test Performance: Acc={acc:.4f}, F1={f1:.4f}")
    else:
        print(f"No test split for dataset '{dataset_class.__name__}'")

    total_emissions = global_metrics['emissions_kg'] + final_metrics['emissions_kg']
    efficiency_used = get_architecture_efficiency_weight(backbone)
    carbon_saved_estimate = total_emissions * (1 - efficiency_used) if efficiency_used < 1.0 else 0

    print(f"\nSUSTAINABILITY SUMMARY:")
    print(f"   Total Carbon Footprint: {total_emissions:.4f} kg CO2eq")
    print(f"   HPO Phase: {global_metrics['emissions_kg']:.4f} kg")
    print(f"   Final Training: {final_metrics['emissions_kg']:.4f} kg")
    print(f"   Architecture Efficiency: {efficiency_used:.1f} (1.0 = most efficient)")
    if carbon_saved_estimate > 0:
        print(f"   Estimated Carbon Saved: {carbon_saved_estimate:.4f} kg CO2eq")
    print(f"Peak GPU Memory: {max(global_metrics['peak_gpu_memory_gb'], final_metrics['peak_gpu_memory_gb']):.2f} GB")

    print("Enhanced AutoML with Combined Carbon Strategies completed successfully!")

    save_accuracy_histogram(study)
    save_metric_curves(study)
    print("Optuna visualizations saved successfully!")
