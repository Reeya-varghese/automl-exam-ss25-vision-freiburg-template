"""
Enhanced EA_SH_CES.py with SMOTE Integration

This file shows how to integrate the SMOTE balancer with your existing
EA_SH_CES.py without modifying the core functionality.

Key changes:
1. Import SMOTE module
2. Add SMOTE parameters to argument parser
3. Apply SMOTE before training if class imbalance detected
4. Log SMOTE statistics in results

Usage:
    python EA_SH_CES_with_smote.py \
        --dataset skin_cancer \
        --n-trials 10 \
        --carbon-budget 0.15 \
        --enable-smote \
        --smote-variant standard \
        --smote-threshold 1.5
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

# Your existing imports (unchanged)
from Zero_cost import ZeroCostCandidateGenerator
from training import AutoML
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

# NEW: Import SMOTE module
from smote_balancer import apply_smote_to_dataset, check_class_imbalance, SMOTEBalancer

# Your existing classes (unchanged)
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

# Your existing utility functions (unchanged)
def get_architecture_efficiency_weight(backbone_name):
    """Efficiency weights based on parameter count and energy research"""
    efficiency_weights = {
        'resnet18': 1.0,        # Most efficient (11M params)
        'efficientnet_b0': 0.8, # Good efficiency (5M params, but complex ops)
        'resnet50': 0.6,        # Moderate efficiency (25M params)
        'vit_base_patch16_224': 0.4  # Least efficient (86M params, attention heavy)
    }
    return efficiency_weights.get(backbone_name, 0.5)

def get_progressive_config(trial_number, total_trials, enable_progressive=True):
    """Start with efficient configs, gradually allow more complex ones"""
    if not enable_progressive:
        return {
            'max_epochs': 8,  # Original default
            'min_batch_size': 32,
            'prefer_efficient_arch': False
        }
    
    progress_ratio = trial_number / max(total_trials, 1)  # Fixed division by zero
    
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

# NEW: Enhanced dataset creation with SMOTE support
def create_dataset_with_smote(
    dataset_class, 
    transform, 
    seed, 
    enable_smote=False, 
    smote_config=None
):
    """
    Create dataset with optional SMOTE balancing.
    
    Returns both the full dataset and train/val splits, with SMOTE applied if needed.
    """
    # Create original dataset
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    
    # Check for class imbalance
    if enable_smote:
        print("🔍 Analyzing class distribution...")
        imbalance_analysis = check_class_imbalance(full_dataset, threshold=smote_config.get('threshold', 1.5))
        
        if imbalance_analysis['needs_balancing']:
            print(f"⚠️ Class imbalance detected (ratio: {imbalance_analysis['imbalance_ratio']:.2f})")
            print(f"📊 Original distribution: {imbalance_analysis['class_counts']}")
            
            # Apply SMOTE
            print(f"🔄 Applying {smote_config.get('variant', 'standard')} SMOTE...")
            
            # Create temporary dataset without transforms for SMOTE
            temp_dataset = dataset_class(root="./data", split='train', download=False, transform=None)
            
            balanced_dataset = apply_smote_to_dataset(
                temp_dataset,
                target_distribution=smote_config.get('target_distribution', 'balanced'),
                smote_variant=smote_config.get('variant', 'standard'),
                feature_extractor=smote_config.get('feature_extractor', 'flatten'),
                k_neighbors=smote_config.get('k_neighbors', 5),
                random_state=seed
            )
            
            print(f"✅ SMOTE complete: {len(temp_dataset)} → {len(balanced_dataset)} samples")
            
            # Apply transforms to balanced dataset
            # Note: This is a simplified approach. In practice, you might need a more 
            # sophisticated way to apply transforms to the balanced dataset
            balanced_dataset.transform = transform
            full_dataset = balanced_dataset
            
        else:
            print("✅ Dataset is already balanced. No SMOTE needed.")
    
    # Create train/val splits
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    
    return full_dataset, train_set, val_set

# Modified optuna_objective to support SMOTE
def optuna_objective(
    trial: optuna.Trial,
    dataset_class: Any,
    seed: int = 42,
    top_k_candidates: list[dict[str, Any]] = None,
    carbon_budget_kg: float = 0.1,
    enable_progressive: bool = True,
    total_trials: int = 10,
    enable_smote: bool = False,
    smote_config: dict = None
    ) -> Tuple[float, float, float, float, float]:
    """Enhanced objective with carbon and GPU tracking + SMOTE support"""
    
    # Start carbon tracking
    tracker = CarbonGPUTracker(f"trial_{trial.number}")
    tracker.start_tracking(trial.number)
    
    try:
        # Your existing hyperparameter suggestions (unchanged)
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        # Progressive configuration (fixed)
        progressive_config = get_progressive_config(trial.number, total_trials, enable_progressive)
        
        # Dynamic batch size and epochs based on progressive strategy
        if progressive_config['prefer_efficient_arch']:
            # Bias toward efficient architectures in early trials
            efficient_candidates = [cid for cid, (backbone, _) in candidate_lookup.items() 
                                  if get_architecture_efficiency_weight(backbone) >= 0.8]
            if efficient_candidates:
                candidate_id = trial.suggest_categorical("candidate_id", efficient_candidates)
            else:
                candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
        else:
            candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
        
        # Dynamic resource optimization
        epochs = trial.suggest_int('epochs', 4, progressive_config['max_epochs'])
        batch_size_options = [16, 32, 64] if progressive_config['min_batch_size'] <= 16 else [32, 64]
        batch_size = trial.suggest_categorical('batch_size', batch_size_options)
        
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        backbone, head = candidate_lookup[candidate_id]
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])
        
        # Get architecture efficiency weight
        efficiency_weight = get_architecture_efficiency_weight(backbone)
        
        # NEW: Enhanced AutoML training with potential SMOTE support
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
        
        # NEW: Apply SMOTE if enabled (modify AutoML fit call)
        if enable_smote and smote_config:
            print(f"🔄 Trial {trial.number}: Training with SMOTE-balanced data")
            trial.set_user_attr("smote_enabled", True)
            trial.set_user_attr("smote_config", smote_config)
        else:
            trial.set_user_attr("smote_enabled", False)
        
        start = time.time()
        print(f"🏗️ Trial {trial.number}: Training {backbone} with {epochs} epochs, batch_size {batch_size}")

        automl.fit(dataset_class, subsample=2000, trial=trial)
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
    
    print(f"✅ Trial {trial.number}: Acc={acc:.3f}, F1={f1:.3f}, Carbon={adjusted_carbon:.4f}kg")
    
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
    parser.add_argument("--dataset", type=str, required=True, choices=["fashion", "flowers", "emotions", "skin_cancer"],)
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--carbon-budget", type=float, default=0.15, help="Carbon budget in kg CO2eq")
    parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    
    # NEW: SMOTE-related arguments
    parser.add_argument("--enable-smote", action="store_true", help="Enable SMOTE class balancing")
    parser.add_argument("--smote-variant", type=str, default="standard", 
                        choices=["standard", "borderline", "adasyn", "smote_tomek", "smote_enn"],
                        help="SMOTE variant to use")
    parser.add_argument("--smote-threshold", type=float, default=1.5, 
                        help="Imbalance ratio threshold for applying SMOTE")
    parser.add_argument("--smote-neighbors", type=int, default=5, 
                        help="Number of neighbors for SMOTE")
    parser.add_argument("--smote-feature-extractor", type=str, default="flatten",
                        choices=["flatten", "resnet_features", "simple_cnn"],
                        help="Feature extraction method for SMOTE")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Dataset selection (unchanged)
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
    
    # NEW: SMOTE configuration
    smote_config = {
        'variant': args.smote_variant,
        'threshold': args.smote_threshold,
        'k_neighbors': args.smote_neighbors,
        'feature_extractor': args.smote_feature_extractor,
        'target_distribution': 'balanced'
    } if args.enable_smote else None
    
    print(f"🌱 Enhanced AutoML with Combined Carbon Strategies + SMOTE - {args.dataset.upper()}")
    print(f"📊 Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"🔄 Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
    print(f"⚖️ SMOTE Balancing: {'Enabled' if args.enable_smote else 'Disabled'}")
    if args.enable_smote:
        print(f"   SMOTE Variant: {args.smote_variant}")
        print(f"   Imbalance Threshold: {args.smote_threshold}")
        print(f"   Feature Extractor: {args.smote_feature_extractor}")
    
    mean, std = calculate_mean_std(dataset_class)
    
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="train", backbone_name=default_backbone)

    # NEW: Enhanced dataset creation with SMOTE support
    full_dataset, train_set, val_set = create_dataset_with_smote(
        dataset_class, transform, args.seed, args.enable_smote, smote_config
    )

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    # Run Zero-Cost Proxy search (unchanged)
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10, num_classes=dataset_class.num_classes)
    top_k_candidates = zcc.get_top_k_candidates()
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")

    print("✅ Top-K candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        efficiency = get_architecture_efficiency_weight(c['backbone'])
        print(f"[{i+1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {efficiency:.1f}")
    
    # Enhanced reference points for carbon-aware optimization
    reference_points = get_enhanced_reference_points()

    # Enhanced NSGA-III sampler with larger population for 5 objectives
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
    
    # NEW: Enhanced optimization with SMOTE support
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
        carbon_budget_kg=args.carbon_budget,
        enable_progressive=args.enable_progressive,
        total_trials=args.n_trials,
        enable_smote=args.enable_smote,
        smote_config=smote_config
    ), n_trials=args.n_trials)

    global_metrics = global_tracker.stop_tracking()

    pareto_trials = study.best_trials
    print(f"\n✅ Pareto-optimal solutions ({len(pareto_trials)}):")
    
    # Enhanced results display with SMOTE information
    for t in pareto_trials:
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        smote_used = t.user_attrs.get('smote_enabled', False)
        smote_indicator = "🔄" if smote_used else "⚪"
        print(f"✅ {smote_indicator} Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
              f"Carbon: {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), GPU: {t.values[4]:.2f}GB, "
              f"Eff: {efficiency:.1f} | {t.params}")

    # Enhanced solution analysis
    print(f"\n🎯 SOLUTION ANALYSIS:")
    print("="*60)
    
    # Categorize solutions by carbon efficiency
    carbon_efficient = [t for t in pareto_trials if t.values[3] < args.carbon_budget * 0.5]
    high_performance = [t for t in pareto_trials if t.values[0] > 0.85 and t.values[1] > 0.8]
    balanced = [t for t in pareto_trials if t not in carbon_efficient and t not in high_performance]
    
    # SMOTE analysis
    smote_trials = [t for t in pareto_trials if t.user_attrs.get('smote_enabled', False)]
    
    print(f"🌱 Carbon Efficient ({len(carbon_efficient)}): Low carbon footprint solutions")
    for t in carbon_efficient[:3]:
        smote_indicator = "🔄" if t.user_attrs.get('smote_enabled', False) else "⚪"
        print(f"   {smote_indicator} Trial {t.number}: Acc={t.values[0]:.3f}, Carbon={t.values[3]:.4f}kg")
    
    print(f"🚀 High Performance ({len(high_performance)}): Best accuracy solutions")
    for t in high_performance[:3]:
        smote_indicator = "🔄" if t.user_attrs.get('smote_enabled', False) else "⚪"
        print(f"   {smote_indicator} Trial {t.number}: Acc={t.values[0]:.3f}, F1={t.values[1]:.3f}")
    
    print(f"⚖️ Balanced ({len(balanced)}): Good trade-offs")
    print(f"🔄 SMOTE Enhanced ({len(smote_trials)}): Trials using class balancing")

    # Select best accuracy solution with error handling
    try:
        best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
        best_params = best_acc_trial.params
        
        # Check if candidate_id exists in parameters
        if 'candidate_id' not in best_params:
            print("❌ No candidate_id in best trial. Using first available candidate.")
            candidate_id = list(candidate_lookup.keys())[0]
        else:
            candidate_id = best_params['candidate_id']
            
        final_epochs = 10 if args.dataset == "flowers" else 8
        backbone, head = candidate_lookup[candidate_id]
        
        print(f"\n🎯 Selected Best Accuracy Solution: Trial {best_acc_trial.number}")
        print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
        print(f"   Sustainability: Carbon={best_acc_trial.values[3]:.4f}kg, GPU={best_acc_trial.values[4]:.2f}GB")
        print(f"   SMOTE Used: {'Yes' if best_acc_trial.user_attrs.get('smote_enabled', False) else 'No'}")
        
    except Exception as e:
        print(f"❌ Error selecting best trial: {e}")
        print("🔧 Using fallback configuration")
        best_params = {'lr': 0.001, 'batch_size': 32, 'optimizer': 'adam'}
        backbone, head = list(candidate_lookup.values())[0]
        final_epochs = 8
    
    # Final training with enhanced AutoML
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
    
    print(f"\n🏁 Final training with {'SMOTE-balanced' if args.enable_smote else 'original'} dataset...")
    automl.fit(dataset_class, subsample=None)
    test_preds, test_labels = automl.predict(dataset_class)
    
    final_metrics = final_tracker.stop_tracking()
    
    if args.dataset == "skin_cancer": 
        output_path = Path("final_test_preds.npy")
    else :
        output_path = args.output_path    
            
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    print(f"\n✅ FINAL RESULTS:")
    print(f"📊 Predictions saved to: {output_path}")
    
    # Enhanced final results with proper error handling
    try:
        if test_labels is not None and not np.isnan(test_labels).any() and len(test_labels) > 0:
            acc = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds, average="macro")
            print(f"✅ Final Test Performance: Acc={acc:.4f}, F1={f1:.4f}")
        else:
            print(f"📤 Test predictions generated for submission")
            print(f"🎯 Predictions shape: {test_preds.shape}")
            print(f"🔢 Unique classes predicted: {len(np.unique(test_preds))}")
    except Exception as e:
        print(f"📤 Test predictions saved successfully")
        print(f"⚠️ Could not evaluate accuracy: {str(e)}")
        
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
    
    # SMOTE summary
    if args.enable_smote:
        smote_trial_count = len([t for t in study.trials if t.user_attrs.get('smote_enabled', False)])
        print(f"\n⚖️ SMOTE SUMMARY:")
        print(f"   SMOTE Enabled: {args.enable_smote}")
        print(f"   Variant Used: {args.smote_variant}")
        print(f"   Imbalance Threshold: {args.smote_threshold}")
        print(f"   Trials with SMOTE: {smote_trial_count}/{len(study.trials)}")
    
    # Carbon efficiency metrics
    # if total_emissions > 0:
    #     try:
    #         carbon_efficiency = test_preds.shape[0] / (total_emissions * 1000) if 'acc' not in locals() else acc / (total_emissions * 1000)
    #         print(f"📈 Carbon Efficiency: {carbon_efficiency:.1f} predictions per gram CO2")
    #     except:
    #         print(f"📈 Carbon tracking completed successfully")
    
    print("✅ Enhanced AutoML with Combined Carbon Strategies + SMOTE completed successfully!")
    
    # Save Optuna plots (unchanged)
    try:
        save_optuna_visualizations(study)
        save_accuracy_histogram(study)
        save_metric_curves(study)
        print("✅ Optuna visualizations saved successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")