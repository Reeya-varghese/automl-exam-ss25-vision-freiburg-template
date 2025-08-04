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
import matplotlib.pyplot as plt
import pandas as pd

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


# REMOVED: Architecture efficiency weights - now only for tracking
def get_architecture_efficiency_weight(backbone_name):
    """Efficiency weights for tracking purposes only (no longer used for pruning)"""
    efficiency_weights = {
        'resnet18': 1.0,  # Most efficient (11M params)
        'efficientnet_b0': 0.8,  # Good efficiency (5M params, but complex ops)
        'resnet50': 0.6,  # Moderate efficiency (25M params)
        'vit_base_patch16_224': 0.4  # Least efficient (86M params, attention heavy)
    }
    return efficiency_weights.get(backbone_name, 0.5)


# REMOVED: Progressive training strategy - now using standard configuration
def get_standard_config():
    """Standard configuration without progressive restrictions"""
    return {
        'max_epochs': 12,  # Allow full epoch range
        'min_batch_size': 16,
        'prefer_efficient_arch': False  # No architecture preference
    }


# Standard reference points for optimization (without carbon focus)
def get_standard_reference_points():
    """Standard reference points for 3-objective optimization (accuracy, f1, time)"""
    return np.array([
        # Performance-focused solutions
        [1, 0, 0],  # Pure accuracy
        [0, 1, 0],  # Pure F1
        [0.7, 0.3, 0],  # Balanced performance
        
        # Speed-focused solutions
        [0, 0, 1],  # Pure speed
        [0.5, 0.5, 0],  # Balanced performance
        [0.3, 0.3, 0.4],  # Balanced with speed consideration
        [0.6, 0.2, 0.2],  # Accuracy focused with some speed
        [0.2, 0.6, 0.2],  # F1 focused with some speed
    ])


def optuna_objective(
        trial: optuna.Trial,
        dataset_class: Any,
        seed: int = 42,
        top_k_candidates: list[dict[str, Any]] = None,
        carbon_tracker_data: dict = None  # For collecting carbon data
) -> Tuple[float, float, float]:
    """Standard objective without carbon constraints"""

    # Start carbon tracking
    tracker = CarbonGPUTracker(f"trial_{trial.number}")
    tracker.start_tracking(trial.number)

    try:
        # Standard hyperparameter suggestions (unchanged)
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

        # REMOVED: Progressive configuration - using standard config
        standard_config = get_standard_config()

        # Standard candidate selection (no efficiency bias)
        candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))

        # Standard resource optimization (no restrictions)
        # epochs = trial.suggest_int('epochs', 4, standard_config['max_epochs'])
        epochs = trial.suggest_int('epochs', 10, 18)
        batch_size_options = [16, 32, 64]
        # batch_size = trial.suggest_categorical('batch_size', batch_size_options)
        batch_size = trial.suggest_categorical('batch_size', [8, 16])

        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        backbone, head = candidate_lookup[candidate_id]
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])

        # Get architecture efficiency weight for tracking only
        efficiency_weight = get_architecture_efficiency_weight(backbone)

        # Standard AutoML training (unchanged)
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
        print(f"Training with head: {head}")

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
        return 0.0, 0.0, 999.0

    # Get sustainability metrics for tracking
    sustainability_metrics = tracker.stop_tracking()

    # REMOVED: Carbon budget constraint - no pruning based on emissions
    
    # Store carbon data for analysis
    if carbon_tracker_data is not None:
        carbon_tracker_data['trials'].append({
            'trial_number': trial.number,
            'emissions_kg': sustainability_metrics['emissions_kg'],
            'training_time': sustainability_metrics['training_time'],
            'peak_gpu_memory_gb': sustainability_metrics['peak_gpu_memory_gb'],
            'backbone': backbone,
            'efficiency_weight': efficiency_weight,
            'accuracy': acc,
            'f1': f1,
            'epochs': epochs,
            'batch_size': batch_size
        })

    # Add sustainability tracking to user attributes (for analysis)
    trial.set_user_attr("emissions_kg", sustainability_metrics['emissions_kg'])
    trial.set_user_attr("peak_gpu_memory_gb", sustainability_metrics['peak_gpu_memory_gb'])
    trial.set_user_attr("efficiency_weight", efficiency_weight)

    # Return 3 objectives: accuracy, f1, time (removed carbon objectives)
    return (
        acc,
        f1,
        training_time
    )


def plot_carbon_emissions(carbon_data, output_dir="./plots"):
    """Generate comprehensive carbon emission plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    df = pd.DataFrame(carbon_data['trials'])
    
    # Plot 1: Cumulative emissions over trials
    plt.figure(figsize=(12, 8))
    cumulative_emissions = df['emissions_kg'].cumsum()
    plt.subplot(2, 2, 1)
    plt.plot(df['trial_number'], cumulative_emissions, 'b-', marker='o', markersize=4)
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative CO2 Emissions (kg)')
    plt.title('Cumulative Carbon Emissions Over Trials')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Emissions per trial
    plt.subplot(2, 2, 2)
    plt.bar(df['trial_number'], df['emissions_kg'], alpha=0.7, color='orange')
    plt.xlabel('Trial Number')
    plt.ylabel('CO2 Emissions per Trial (kg)')
    plt.title('Carbon Emissions per Trial')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Emissions vs Performance
    plt.subplot(2, 2, 3)
    plt.scatter(df['emissions_kg'], df['accuracy'], alpha=0.6, c=df['trial_number'], cmap='viridis')
    plt.xlabel('CO2 Emissions (kg)')
    plt.ylabel('Accuracy')
    plt.title('Carbon Emissions vs Accuracy')
    plt.colorbar(label='Trial Number')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Emissions by backbone
    plt.subplot(2, 2, 4)
    backbone_emissions = df.groupby('backbone')['emissions_kg'].mean()
    plt.bar(range(len(backbone_emissions)), backbone_emissions.values, alpha=0.7, color='green')
    plt.xlabel('Backbone Architecture')
    plt.ylabel('Average CO2 Emissions (kg)')
    plt.title('Average Emissions by Architecture')
    plt.xticks(range(len(backbone_emissions)), backbone_emissions.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/carbon_emissions_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed plot
    plt.figure(figsize=(14, 6))
    
    # Training time vs emissions
    plt.subplot(1, 2, 1)
    plt.scatter(df['training_time'], df['emissions_kg'], 
                c=df['accuracy'], cmap='RdYlGn', alpha=0.7, s=60)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.title('Training Time vs Carbon Emissions\n(Color = Accuracy)')
    plt.colorbar(label='Accuracy')
    plt.grid(True, alpha=0.3)
    
    # GPU memory vs emissions
    plt.subplot(1, 2, 2)
    plt.scatter(df['peak_gpu_memory_gb'], df['emissions_kg'], 
                c=df['epochs'], cmap='plasma', alpha=0.7, s=60)
    plt.xlabel('Peak GPU Memory (GB)')
    plt.ylabel('CO2 Emissions (kg)')
    plt.title('GPU Memory vs Carbon Emissions\n(Color = Epochs)')
    plt.colorbar(label='Epochs')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/carbon_detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n🌱 CARBON EMISSIONS ANALYSIS:")
    print(f"=" * 50)
    print(f"Total CO2 Emissions: {df['emissions_kg'].sum():.4f} kg")
    print(f"Average per Trial: {df['emissions_kg'].mean():.4f} kg")
    print(f"Min per Trial: {df['emissions_kg'].min():.4f} kg")
    print(f"Max per Trial: {df['emissions_kg'].max():.4f} kg")
    print(f"Standard Deviation: {df['emissions_kg'].std():.4f} kg")
    
    print(f"\n📊 EMISSIONS BY ARCHITECTURE:")
    for arch, emissions in backbone_emissions.items():
        count = len(df[df['backbone'] == arch])
        print(f"  {arch}: {emissions:.4f} kg (avg), {count} trials")
    
    # Calculate carbon efficiency
    df['carbon_efficiency'] = df['accuracy'] / (df['emissions_kg'] * 1000)  # accuracy per gram CO2
    print(f"\n⚡ CARBON EFFICIENCY:")
    print(f"Best: {df['carbon_efficiency'].max():.2f} accuracy points per gram CO2")
    print(f"Average: {df['carbon_efficiency'].mean():.2f} accuracy points per gram CO2")
    print(f"Worst: {df['carbon_efficiency'].min():.2f} accuracy points per gram CO2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["fashion", "flowers", "emotions", "skin_cancer"], )
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
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

    print(f"🌱 AutoML with Carbon Emission Tracking (No Pruning) - {args.dataset.upper()}")
    print(f"📊 Will track and analyze CO2 emissions without constraints")

    mean, std = calculate_mean_std(dataset_class)

    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="train", backbone_name=default_backbone)

    # Load and split dataset (unchanged)
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    # Run Zero-Cost Proxy search (unchanged)
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10,
                                     num_classes=dataset_class.num_classes)
    top_k_candidates = [c for c in zcc.get_top_k_candidates() if
                        c['backbone'] in ['resnet18', 'efficientnet_b0', 'vit_base_patch16_224']]
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        efficiency = get_architecture_efficiency_weight(c['backbone'])
        print(
            f"[{i + 1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {efficiency:.1f}")

    # MODIFIED: Standard reference points (3 objectives instead of 5)
    reference_points = get_standard_reference_points()

    # MODIFIED: Standard NSGA-III sampler for 3 objectives
    sampler = NSGAIIISampler(
        population_size=30,  # Reduced for 3 objectives
        mutation_prob=0.1,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )

    # MODIFIED: Study with 3 objectives (removed carbon objectives)
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],  # accuracy, f1, time
        sampler=sampler,
    )

    # Initialize carbon tracking data
    carbon_tracker_data = {'trials': []}

    # Global carbon tracking
    global_tracker = CarbonGPUTracker("global_optimization")
    global_tracker.start_tracking()

    # MODIFIED: Optimization without carbon constraints
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
        carbon_tracker_data=carbon_tracker_data  # Pass carbon tracking data
    ), n_trials=args.n_trials)

    global_metrics = global_tracker.stop_tracking()

    pareto_trials = study.best_trials
    print(f"\n✅ Pareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        actual_carbon = t.user_attrs.get('emissions_kg', 0)
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        print(f"✅ Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
              f"Carbon: {actual_carbon:.4f}kg, Eff: {efficiency:.1f} | {t.params}")

    # Generate carbon emission analysis plots
    plot_carbon_emissions(carbon_tracker_data)

    # Select best accuracy solution
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]

    print(f"\n🎯 Selected Best Accuracy Solution: Trial {best_acc_trial.number}")
    print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
    actual_carbon = best_acc_trial.user_attrs.get('emissions_kg', 0)
    print(f"   Carbon Footprint: {actual_carbon:.4f}kg")

    # Final training
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

    print(f"\n✅ FINAL RESULTS:")
    print(f"📊 Predictions saved to: {output_path}")
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"✅ Final Test Performance: Acc={acc:.4f}, F1={f1:.4f}")
    else:
        print(f"No test split for dataset '{dataset_class.__name__}'")

    # Final carbon summary
    total_emissions = global_metrics['emissions_kg'] + final_metrics['emissions_kg']
    
    print(f"\n🌱 TOTAL CARBON FOOTPRINT SUMMARY:")
    print(f"   Total Carbon Footprint: {total_emissions:.4f} kg CO2eq")
    print(f"   HPO Phase: {global_metrics['emissions_kg']:.4f} kg")
    print(f"   Final Training: {final_metrics['emissions_kg']:.4f} kg")
    print(f"🖥️ Peak GPU Memory: {max(global_metrics['peak_gpu_memory_gb'], final_metrics['peak_gpu_memory_gb']):.2f} GB")

    # Calculate total carbon efficiency
    if total_emissions > 0 and not np.isnan(test_labels).any():
        carbon_efficiency = acc / (total_emissions * 1000)  # Accuracy per gram CO2
        print(f"📈 Overall Carbon Efficiency: {carbon_efficiency:.1f} accuracy points per gram CO2")

    print("✅ AutoML with Carbon Emission Tracking completed successfully!")

    # Save standard Optuna plots
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    print("✅ All visualizations saved successfully!")