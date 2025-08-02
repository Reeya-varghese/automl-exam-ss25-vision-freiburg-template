"""
Enhanced EA_SH_CES with SMOTE Integration

This file integrates SMOTE class balancing into the existing carbon-aware
multi-objective AutoML pipeline. It maintains all existing functionality
while adding optional SMOTE support for handling class imbalanced datasets.

Key changes:
- Added SMOTE-related command line arguments
- Enhanced objective function to use SMOTE when enabled
- Updated logging to include class imbalance information
- Maintains full backward compatibility
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import threading
from codecarbon import EmissionsTracker

from Zero_cost import ZeroCostCandidateGenerator
# Import the enhanced training module with SMOTE support
from training_with_smote import AutoML
import optuna
from optuna.samplers import NSGAIIISampler

from Plots import (
    save_optuna_visualizations,
    save_accuracy_histogram,
    save_metric_curves
)
from model import get_transforms
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
from torch.utils.data import random_split

# Import SMOTE utilities
from smote_balancer import SMOTEBalancer, check_imbalanced_learn_availability

# ---------------------------------------------
logger = logging.getLogger(__name__)

class CarbonGPUTracker:
    """Smart carbon emissions and GPU tracking"""
    
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
        self.emissions_tracker = EmissionsTracker(
            project_name=tracker_name,
            measure_power_secs=15,
            save_to_file=True,
            log_level="WARNING"
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
                    current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
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

# Architecture efficiency weights based on research
def get_architecture_efficiency_weight(backbone_name):
    """Efficiency weights based on parameter count and energy research"""
    efficiency_weights = {
        'resnet18': 1.0,        # Most efficient (11M params)
        'efficientnet_b0': 0.8, # Good efficiency (5M params, but complex ops)
        'resnet50': 0.6,        # Moderate efficiency (25M params)
        'vit_base_patch16_224': 0.4  # Least efficient (86M params, attention heavy)
    }
    return efficiency_weights.get(backbone_name, 0.5)

# Progressive training strategy
def get_progressive_config(trial_number, total_trials, enable_progressive=True):
    """Start with efficient configs, gradually allow more complex ones"""
    if not enable_progressive:
        return {
            'max_epochs': 8,
            'min_batch_size': 32,
            'prefer_efficient_arch': False
        }
    
    progress_ratio = trial_number / max(total_trials, 1)
    
    if progress_ratio < 0.3:  # First 30% of trials: focus on efficiency
        return {
            'max_epochs': 8,
            'min_batch_size': 32,
            'prefer_efficient_arch': True
        }
    elif progress_ratio < 0.7:  # Middle 40%: balanced approach
        return {
            'max_epochs': 10,
            'min_batch_size': 16,
            'prefer_efficient_arch': False
        }
    else:  # Final 30%: allow complex models if carbon budget permits
        return {
            'max_epochs': 12,
            'min_batch_size': 16,
            'prefer_efficient_arch': False
        }

# Enhanced reference points for carbon-aware optimization
def get_enhanced_reference_points():
    """Enhanced reference points for 5-objective optimization with carbon focus"""
    return np.array([
        # Performance-focused solutions
        [1, 0, 0, 0, 0],           # Pure accuracy
        [0, 1, 0, 0, 0],           # Pure F1
        [0.7, 0.3, 0, 0, 0],       # Balanced performance
        
        # Efficiency-focused solutions  
        [0, 0, 1, 0, 0],           # Pure speed
        [0, 0, 0, 1, 0],           # Pure carbon efficiency
        [0, 0, 0, 0, 1],           # Pure GPU efficiency
        
        # Balanced sustainability solutions
        [0.4, 0.4, 0.1, 0.05, 0.05],  # Performance + minimal sustainability
        [0.3, 0.3, 0.2, 0.1, 0.1],    # Balanced all objectives
        [0.2, 0.2, 0.15, 0.25, 0.2],  # Sustainability-focused
        [0.1, 0.1, 0.1, 0.35, 0.35],  # Green AI focused
    ])

def optuna_objective(
    trial: optuna.Trial,
    dataset_class: Any,
    seed: int = 42,
    top_k_candidates: list[dict[str, Any]] = None,
    carbon_budget_kg: float = 0.1,
    enable_progressive: bool = True,
    total_trials: int = 10,
    # NEW SMOTE parameters
    enable_smote: bool = False,
    smote_max_samples: int = 2000,
    smote_strategy: str = 'auto'
    ) -> Tuple[float, float, float, float, float]:
    """Enhanced objective with carbon and GPU tracking + SMOTE support"""
    
    # Start carbon tracking
    tracker = CarbonGPUTracker(f"trial_{trial.number}")
    tracker.start_tracking(trial.number)
    
    try:
        # Existing hyperparameter suggestions
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        # Progressive configuration
        progressive_config = get_progressive_config(trial.number, total_trials, enable_progressive)
        
        # Dynamic resource optimization
        if progressive_config['prefer_efficient_arch']:
            efficient_candidates = [cid for cid, (backbone, _) in candidate_lookup.items() 
                                  if get_architecture_efficiency_weight(backbone) >= 0.8]
            if efficient_candidates:
                candidate_id = trial.suggest_categorical("candidate_id", efficient_candidates)
            else:
                candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
        else:
            candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
        
        epochs = trial.suggest_int('epochs', 4, progressive_config['max_epochs'])
        batch_size_options = [16, 32, 64] if progressive_config['min_batch_size'] <= 16 else [32, 64]
        batch_size = trial.suggest_categorical('batch_size', batch_size_options)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        backbone, head = candidate_lookup[candidate_id]
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])
        
        # Get architecture efficiency weight
        efficiency_weight = get_architecture_efficiency_weight(backbone)
        
        # Enhanced AutoML with SMOTE support
        automl = AutoML(
            seed=seed,
            num_layers_to_freeze=0,
            lr=lr,
            use_augmentation=use_augmentation,
            backbone=backbone,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            custom_head=head,
            # NEW: SMOTE parameters
            use_smote=enable_smote,
            smote_max_samples=smote_max_samples,
            smote_strategy=smote_strategy,
            detect_imbalance=True  # Always detect imbalance for logging
        )
        
        start = time.time()
        print(f"Trial {trial.number}: {backbone} with {head.__class__.__name__}")
        if enable_smote:
            print(f"SMOTE enabled (max {smote_max_samples} samples)")

        automl.fit(dataset_class, subsample=2000, trial=trial)
        training_time = time.time() - start
        
        # Log training summary including SMOTE info
        training_summary = automl.get_training_summary()
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Training summary: {training_summary}")
        
        # Store trial metadata
        trial.set_user_attr("head_type", head.__class__.__name__)
        trial.set_user_attr("backbone", backbone)
        trial.set_user_attr("smote_enabled", enable_smote)
        if automl.imbalance_info:
            trial.set_user_attr("imbalance_detected", automl.imbalance_info['is_imbalanced'])
            trial.set_user_attr("imbalance_ratio", automl.imbalance_info['imbalance_ratio'])
            trial.set_user_attr("class_distribution", automl.imbalance_info['class_counts'])

        preds, labels = automl.evaluate_on_val()
        
        if not np.isnan(labels).any():
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
        else:
            acc = 0
            f1 = 0
            
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        tracker.stop_tracking()
        return 0.0, 0.0, 999.0, 999.0, 999.0
    
    # Get sustainability metrics
    sustainability_metrics = tracker.stop_tracking()
    
    # Apply architecture efficiency weighting to carbon cost
    adjusted_carbon = sustainability_metrics['emissions_kg'] / efficiency_weight
    
    # Add sustainability tracking to user attributes
    trial.set_user_attr("emissions_kg", sustainability_metrics['emissions_kg'])
    trial.set_user_attr("adjusted_carbon", adjusted_carbon)
    trial.set_user_attr("peak_gpu_memory_gb", sustainability_metrics['peak_gpu_memory_gb'])
    trial.set_user_attr("efficiency_weight", efficiency_weight)
    
    # Carbon budget constraint with efficiency weighting
    if adjusted_carbon > carbon_budget_kg:
        trial.set_user_attr("carbon_budget_exceeded", True)
        raise optuna.TrialPruned(f"Carbon budget exceeded: {adjusted_carbon:.4f} kg (efficiency-adjusted)")
    
    # Return 5 objectives: accuracy, f1, time, adjusted_emissions, gpu_memory
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
    parser.add_argument("--dataset", type=str, required=True, choices=["fashion", "flowers", "emotions", "skin_cancer"])
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Carbon-aware parameters
    parser.add_argument("--carbon-budget", type=float, default=0.15, help="Carbon budget in kg CO2eq")
    parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
    
    # NEW: SMOTE parameters
    parser.add_argument("--enable-smote", action="store_true", help="Enable SMOTE for class imbalance handling")
    parser.add_argument("--smote-max-samples", type=int, default=2000, help="Maximum samples for SMOTE (memory constraint)")
    parser.add_argument("--smote-strategy", type=str, default="auto", choices=["auto", "minority", "not minority", "all"], help="SMOTE sampling strategy")
    parser.add_argument("--detect-imbalance-only", action="store_true", help="Only detect and report imbalance without applying SMOTE")
    
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Check SMOTE availability
    if args.enable_smote and not check_imbalanced_learn_availability():
        logger.error("SMOTE requested but imbalanced-learn not available!")
        logger.error("Install with: pip install imbalanced-learn")
        logger.info("Continuing without SMOTE...")
        args.enable_smote = False

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
    
    print(f"🌱 Enhanced AutoML with Combined Carbon Strategies + SMOTE - {args.dataset.upper()}")
    print(f"📊 Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"🔄 Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
    print(f"⚖️ SMOTE Balancing: {'Enabled' if args.enable_smote else 'Disabled'}")
    if args.enable_smote:
        print(f"   - Max samples for SMOTE: {args.smote_max_samples}")
        print(f"   - SMOTE strategy: {args.smote_strategy}")
    
    # Quick dataset balance analysis
    if args.detect_imbalance_only or args.enable_smote:
        print(f"\n📈 Analyzing dataset class balance...")
        temp_dataset = dataset_class(root="./data", split='train', download=True, transform=None)
        balance_info = SMOTEBalancer.analyze_dataset_balance(temp_dataset)
        
        print(f"   - Total classes: {balance_info['num_classes']}")
        print(f"   - Class distribution: {balance_info['class_counts']}")
        print(f"   - Imbalance ratio: {balance_info['imbalance_ratio']:.2f}")
        print(f"   - Requires balancing: {'Yes' if balance_info['is_imbalanced'] else 'No'}")
        
        if args.detect_imbalance_only:
            print("✅ Imbalance analysis completed. Exiting (--detect-imbalance-only)")
            exit(0)
    
    mean, std = calculate_mean_std(dataset_class)
    
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="train", backbone_name=default_backbone)

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
    for i, c in enumerate(top_k_candidates):
        efficiency = get_architecture_efficiency_weight(c['backbone'])
        print(f"[{i+1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {efficiency:.1f}")
    
    # Enhanced reference points for carbon-aware optimization
    reference_points = get_enhanced_reference_points()

    # Enhanced NSGA-III sampler
    sampler = NSGAIIISampler(
        population_size=50,
        mutation_prob=0.15,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )
   
    # Enhanced study with 5 objectives
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
        total_trials=args.n_trials,
        # NEW: SMOTE parameters
        enable_smote=args.enable_smote,
        smote_max_samples=args.smote_max_samples,
        smote_strategy=args.smote_strategy
    ), n_trials=args.n_trials)

    global_metrics = global_tracker.stop_tracking()

    pareto_trials = study.best_trials
    print(f"\n✅ Pareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        smote_status = "SMOTE" if t.user_attrs.get('smote_enabled', False) else "Original"
        imbalance_ratio = t.user_attrs.get('imbalance_ratio', 'N/A')
        
        print(f"✅ Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
              f"Carbon: {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), GPU: {t.values[4]:.2f}GB, "
              f"Eff: {efficiency:.1f}, Data: {smote_status} (IR: {imbalance_ratio}) | {t.params}")

    # Enhanced solution analysis
    print(f"\n🎯 SOLUTION ANALYSIS:")
    print("="*60)
    
    # Categorize solutions
    carbon_efficient = [t for t in pareto_trials if t.values[3] < args.carbon_budget * 0.5]
    high_performance = [t for t in pareto_trials if t.values[0] > 0.85 and t.values[1] > 0.8]
    smote_trials = [t for t in pareto_trials if t.user_attrs.get('smote_enabled', False)]
    
    print(f"🌱 Carbon Efficient ({len(carbon_efficient)}): Low carbon footprint solutions")
    print(f"🚀 High Performance ({len(high_performance)}): Best accuracy solutions")
    print(f"⚖️ SMOTE Enhanced ({len(smote_trials)}): Class-balanced solutions")

    # Select best accuracy solution
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]
    
    print(f"\n🎯 Selected Best Accuracy Solution: Trial {best_acc_trial.number}")
    print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
    print(f"   Sustainability: Carbon={best_acc_trial.values[3]:.4f}kg, GPU={best_acc_trial.values[4]:.2f}GB")
    smote_used = best_acc_trial.user_attrs.get('smote_enabled', False)
    print(f"   Data Processing: {'SMOTE balanced' if smote_used else 'Original data'}")
    
    # Final training with SMOTE if it was beneficial
    automl = AutoML(
        seed=args.seed,
        num_layers_to_freeze=0,
        lr=best_params.get("lr", 0.001),
        use_augmentation=True,
        backbone=backbone,
        batch_size=best_params.get("batch_size", 32),
        epochs=final_epochs,
        optimizer=best_params.get("optimizer", "adam"),
        custom_head=head,
        # Use SMOTE if it was used in the best trial
        use_smote=smote_used,
        smote_max_samples=args.smote_max_samples,
        smote_strategy=args.smote_strategy
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

    print(f"\n✅ FINAL RESULTS:")
    print(f"📊 Predictions saved to: {output_path}")
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"✅ Final Test Performance: Acc={acc:.4f}, F1={f1:.4f}")
    else:
        print(f"No test split for dataset '{dataset_class.__name__}'")
    
    # Enhanced carbon summary
    total_emissions = global_metrics['emissions_kg'] + final_metrics['emissions_kg']
    efficiency_used = get_architecture_efficiency_weight(backbone)
    carbon_saved_estimate = total_emissions * (1 - efficiency_used) if efficiency_used < 1.0 else 0
    
    print(f"\n🌱 SUSTAINABILITY SUMMARY:")
    print(f"   Total Carbon Footprint: {total_emissions:.4f} kg CO2eq")
    print(f"   HPO Phase: {global_metrics['emissions_kg']:.4f} kg")
    print(f"   Final Training: {final_metrics['emissions_kg']:.4f} kg")
    print(f"   Architecture Efficiency: {efficiency_used:.1f} (1.0 = most efficient)")
    print(f"   Estimated Carbon Saved: {carbon_saved_estimate:.4f} kg CO2eq")
    print(f"🖥️ Peak GPU Memory: {max(global_metrics['peak_gpu_memory_gb'], final_metrics['peak_gpu_memory_gb']):.2f} GB")
    
    # Carbon efficiency metrics
    if total_emissions > 0:
        carbon_efficiency = acc / (total_emissions * 1000)  # Accuracy per gram CO2
        print(f"📈 Carbon Efficiency: {carbon_efficiency:.1f} accuracy points per gram CO2")
    
    # SMOTE summary
    final_summary = automl.get_training_summary()
    print(f"\n⚖️ CLASS BALANCE SUMMARY:")
    print(f"   SMOTE Applied: {'Yes' if final_summary['smote_enabled'] else 'No'}")
    if final_summary.get('imbalance_detected'):
        print(f"   Imbalance Ratio: {final_summary['imbalance_ratio']:.2f}")
        print(f"   Class Distribution: {final_summary['class_distribution']}")
    
    print("✅ Enhanced AutoML with SMOTE and Carbon Strategies completed successfully!")
    
    # Save Optuna plots
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    print("✅ Optuna visualizations saved successfully!")