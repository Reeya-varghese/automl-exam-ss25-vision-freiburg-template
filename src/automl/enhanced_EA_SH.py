# enhanced_EA_SH.py
# Enhanced version of your EA_SH.py with complete CO2 tracking integration

import argparse
import logging
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from Zero_cost import ZeroCostCandidateGenerator
from enhanced_training import EnhancedAutoML  # Use enhanced version
from co2_tracker import CO2Tracker, EnhancedVisualizer  # Import tracking components
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

logger = logging.getLogger(__name__)

# Global CO2 tracker and visualizer instances
co2_tracker = None
visualizer = None

def enhanced_optuna_objective(
    trial: optuna.Trial,
    dataset_class: Any,
    seed: int = 42,
    top_k_candidates: list[dict[str, Any]] = None
    ) -> Tuple[float, float, float]:
    """
    Enhanced Optuna objective with CO2 tracking and detailed logging.
    """
    global co2_tracker, visualizer
    
    # Set tracking phase for this trial
    if co2_tracker:
        co2_tracker.set_phase(f"trial_{trial.number}")
    
    candidate_lookup = {
        f"{c['backbone']}_{i}": (c['backbone'], c['head'])
        for i, c in enumerate(top_k_candidates)
    }
    
    # Hyperparameter suggestions
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    candidate_id = trial.suggest_categorical("candidate_id", list(candidate_lookup.keys()))
    backbone, head = candidate_lookup[candidate_id]
    use_augmentation = trial.suggest_categorical('use_augmentation', [True])
    epochs = 8  
    
    # Create enhanced AutoML instance with CO2 tracking
    automl = EnhancedAutoML(
        seed=seed,
        num_layers_to_freeze=0,
        lr=lr,
        use_augmentation=use_augmentation,
        backbone=backbone,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        custom_head=head,
        co2_tracker=co2_tracker,
        visualizer=visualizer,
        enable_co2_tracking=True
    )
    
    start = time.time()
    print(f"\n🔥 Trial {trial.number}: {backbone} with custom head")
    print(f"⚙️ Params: LR={lr:.6f}, Batch={batch_size}, Optimizer={optimizer}")

    # Training with CO2 tracking
    automl.fit(dataset_class, subsample=2000, trial=trial)
    training_time = time.time() - start
    
    logger.info(f"Training time: {training_time:.2f} seconds")
    trial.set_user_attr("head_type", head.__class__.__name__)
    trial.set_user_attr("backbone", backbone)
    trial.set_user_attr("training_time", training_time)
    
    # Get CO2 statistics for this trial
    if co2_tracker:
        co2_stats = co2_tracker.get_current_stats()
        trial.set_user_attr("co2_emissions", co2_stats.get('total_co2_g', 0))
        trial.set_user_attr("avg_power", co2_stats.get('avg_power_w', 0))
        print(f"🌱 Trial CO2: {co2_stats.get('total_co2_g', 0):.1f}g, "
              f"Power: {co2_stats.get('avg_power_w', 0):.1f}W")

    # Evaluation
    preds, labels = automl.evaluate_on_val()
    
    if not np.isnan(labels).any():
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        
        # Calculate efficiency metrics
        co2_efficiency = acc / max(co2_stats.get('total_co2_g', 1), 1) if co2_tracker else 0
        trial.set_user_attr("co2_efficiency", co2_efficiency)
        
        print(f"📊 Results: Acc={acc:.4f}, F1={f1:.4f}, Time={training_time:.1f}s")
        if co2_tracker:
            print(f"♻️ Efficiency: {co2_efficiency:.6f} acc/g_CO2")
    else:
        acc = 0
        f1 = 0
        co2_efficiency = 0
    
    # Log trial data for visualization
    if visualizer:
        visualizer.log_trial({
            'trial_id': trial.number,
            'params': trial.params,
            'values': [acc, f1, training_time],
            'co2_emissions': co2_stats.get('total_co2_g', 0) if co2_tracker else 0,
            'co2_efficiency': co2_efficiency,
            'backbone': backbone,
            'head_type': head.__class__.__name__
        })
    
    return acc, f1, training_time    

def create_enhanced_visualizations(study, dataset_name, co2_tracker, visualizer):
    """Create comprehensive visualizations including CO2 analysis."""
    
    print("\n📊 Creating enhanced visualizations...")
    
    # 1. Original Optuna visualizations
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    
    # 2. Comprehensive CO2 dashboard
    visualizer.create_comprehensive_dashboard(f"{dataset_name}_co2_dashboard.png")
    
    # 3. 3D search space visualization
    visualizer.create_search_space_3d(f"{dataset_name}_search_space_3d.png")
    
    # 4. Create custom CO2-focused analysis
    create_co2_analysis_plots(study, dataset_name)
    
    # 5. Create Pareto front with CO2 considerations
    create_pareto_co2_analysis(study, dataset_name)
    
    print("✅ All enhanced visualizations created!")

def create_co2_analysis_plots(study, dataset_name):
    """Create CO2-specific analysis plots."""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract trial data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            co2_emissions = trial.user_attrs.get('co2_emissions', 0)
            co2_efficiency = trial.user_attrs.get('co2_efficiency', 0)
            training_time = trial.user_attrs.get('training_time', 0)
            backbone = trial.user_attrs.get('backbone', 'unknown')
            
            trials_data.append({
                'trial_id': trial.number,
                'accuracy': trial.values[0] if trial.values else 0,
                'f1': trial.values[1] if len(trial.values) > 1 else 0,
                'training_time': training_time,
                'co2_emissions': co2_emissions,
                'co2_efficiency': co2_efficiency,
                'backbone': backbone,
                'lr': trial.params.get('lr', 0),
                'batch_size': trial.params.get('batch_size', 32)
            })
    
    if not trials_data:
        print("No completed trials found for CO2 analysis")
        return
    
    import pandas as pd
    df = pd.DataFrame(trials_data)
    
    # 1. CO2 vs Accuracy scatter plot
    axes[0, 0].scatter(df['co2_emissions'], df['accuracy'], 
                      c=df['training_time'], cmap='viridis', s=60, alpha=0.7)
    axes[0, 0].set_xlabel('CO2 Emissions (g)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('🎯 Accuracy vs CO2 Emissions\n(Color = Training Time)')
    
    # Add trend line
    if len(df) > 3:
        z = np.polyfit(df['co2_emissions'], df['accuracy'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df['co2_emissions'], p(df['co2_emissions']), 
                       "r--", alpha=0.8, label=f'Trend')
        axes[0, 0].legend()
    
    # 2. CO2 efficiency by backbone
    backbone_efficiency = df.groupby('backbone')['co2_efficiency'].agg(['mean', 'std']).reset_index()
    bars = axes[0, 1].bar(backbone_efficiency['backbone'], backbone_efficiency['mean'],
                         yerr=backbone_efficiency['std'], capsize=5, alpha=0.7)
    axes[0, 1].set_xlabel('Architecture')
    axes[0, 1].set_ylabel('CO2 Efficiency (Acc/g)')
    axes[0, 1].set_title('♻️ CO2 Efficiency by Architecture')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean_val in zip(bars, backbone_efficiency['mean']):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                       f'{mean_val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Hyperparameter impact on CO2
    lr_co2 = df.groupby('lr')['co2_emissions'].mean().reset_index()
    axes[1, 0].semilogx(lr_co2['lr'], lr_co2['co2_emissions'], 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Average CO2 Emissions (g)')
    axes[1, 0].set_title('📈 Learning Rate Impact on CO2')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Multi-objective Pareto analysis
    # Normalize metrics for comparison
    normalized_acc = (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min())
    normalized_co2 = 1 - (df['co2_emissions'] - df['co2_emissions'].min()) / (df['co2_emissions'].max() - df['co2_emissions'].min())
    combined_score = 0.7 * normalized_acc + 0.3 * normalized_co2
    
    scatter = axes[1, 1].scatter(df['accuracy'], df['co2_emissions'], 
                                c=combined_score, cmap='RdYlGn', s=80, alpha=0.7)
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('CO2 Emissions (g)')
    axes[1, 1].set_title('🏆 Multi-Objective Score\n(Green = Better)')
    
    # Highlight best combined score
    best_idx = np.argmax(combined_score)
    axes[1, 1].scatter(df.iloc[best_idx]['accuracy'], df.iloc[best_idx]['co2_emissions'],
                      s=200, c='red', marker='*', label='Best Combined')
    axes[1, 1].legend()
    
    plt.colorbar(scatter, ax=axes[1, 1], label='Combined Score')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_co2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"💚 CO2 analysis plots saved to {dataset_name}_co2_analysis.png")

def create_pareto_co2_analysis(study, dataset_name):
    """Create Pareto front analysis considering CO2 emissions."""
    import matplotlib.pyplot as plt
    
    # Extract objectives: accuracy (maximize), CO2 (minimize)
    objectives = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
            accuracy = trial.values[0]
            co2 = trial.user_attrs.get('co2_emissions', 0)
            objectives.append([accuracy, -co2])  # Negative CO2 for maximization
    
    if len(objectives) < 3:
        print("Not enough trials for Pareto analysis")
        return
    
    objectives = np.array(objectives)
    
    # Find Pareto front
    def is_pareto_optimal(costs, i):
        return np.all(np.any(costs[i] >= costs, axis=1))
    
    pareto_mask = [is_pareto_optimal(objectives, i) for i in range(len(objectives))]
    pareto_front = objectives[pareto_mask]
    
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(objectives[:, 0], -objectives[:, 1], 
               alpha=0.6, s=50, c='lightblue', label='All Trials')
    
    # Plot Pareto front
    plt.scatter(pareto_front[:, 0], -pareto_front[:, 1], 
               alpha=0.8, s=100, c='red', label='Pareto Front')
    
    # Connect Pareto front points
    if len(pareto_front) > 1:
        sorted_pareto = pareto_front[np.argsort(pareto_front[:, 0])]
        plt.plot(sorted_pareto[:, 0], -sorted_pareto[:, 1], 
                'r--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Accuracy')
    plt.ylabel('CO2 Emissions (g)')
    plt.title(f'🎯 Pareto Front: Accuracy vs CO2 Emissions\n({dataset_name} Dataset)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add annotations for Pareto optimal points
    for i, point in enumerate(pareto_front):
        plt.annotate(f'P{i+1}', (point[0], -point[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🎯 Pareto front analysis saved to {dataset_name}_pareto_front.png")
    
    # Print Pareto optimal solutions
    print(f"\n🏆 Pareto Optimal Solutions for {dataset_name}:")
    print("=" * 50)
    for i, point in enumerate(pareto_front):
        print(f"Solution P{i+1}: Accuracy = {point[0]:.4f}, CO2 = {-point[1]:.1f}g")

def print_enhanced_summary(study, co2_tracker, dataset_name):
    """Print comprehensive summary including CO2 metrics."""
    
    print(f"\n🎊 ENHANCED AUTOML SUMMARY - {dataset_name.upper()}")
    print("=" * 60)
    
    # Basic study statistics
    pareto_trials = study.best_trials
    total_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    print(f"📊 Optimization Statistics:")
    print(f"   • Total Trials: {total_trials}")
    print(f"   • Completed Trials: {completed_trials}")
    print(f"   • Pareto Optimal Solutions: {len(pareto_trials)}")
    
    # CO2 and environmental impact
    if co2_tracker:
        co2_summary = co2_tracker.get_summary()
        print(f"\n🌍 Environmental Impact:")
        print(f"   • Total CO2 Emissions: {co2_summary.get('total_co2_g', 0):.1f}g")
        print(f"   • Equivalent Car Distance: {co2_summary.get('car_equivalent_km', 0):.2f} km")
        print(f"   • Tree Offset Required: {co2_summary.get('tree_offset_years', 0):.3f} tree-years")
        print(f"   • Average Power Consumption: {co2_summary.get('avg_power_w', 0):.1f}W")
        print(f"   • Total Energy: {co2_summary.get('avg_power_w', 0) * co2_summary.get('duration_hours', 0) / 1000:.2f} kWh")
        print(f"   • Carbon Intensity ({co2_summary.get('region', 'N/A')}): {co2_summary.get('carbon_intensity', 0)} gCO2/kWh")
    
    # Best solutions analysis
    print(f"\n🏆 Top Pareto Solutions:")
    print("-" * 40)
    for i, trial in enumerate(pareto_trials[:5]):  # Top 5
        co2_emissions = trial.user_attrs.get('co2_emissions', 0)
        backbone = trial.user_attrs.get('backbone', 'Unknown')
        co2_efficiency = trial.user_attrs.get('co2_efficiency', 0)
        
        print(f"   Solution {i+1}:")
        print(f"     • Accuracy: {trial.values[0]:.4f}")
        print(f"     • F1 Score: {trial.values[1]:.4f}")
        print(f"     • Training Time: {trial.values[2]:.1f}s")
        print(f"     • CO2 Emissions: {co2_emissions:.1f}g")
        print(f"     • Architecture: {backbone}")
        print(f"     • CO2 Efficiency: {co2_efficiency:.6f} acc/g")
        print(f"     • Hyperparameters: {trial.params}")
        print()
    
    # Architecture analysis
    backbone_stats = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            backbone = trial.user_attrs.get('backbone', 'Unknown')
            co2 = trial.user_attrs.get('co2_emissions', 0)
            
            if backbone not in backbone_stats:
                backbone_stats[backbone] = {'count': 0, 'total_co2': 0, 'best_acc': 0}
            
            backbone_stats[backbone]['count'] += 1
            backbone_stats[backbone]['total_co2'] += co2
            backbone_stats[backbone]['best_acc'] = max(
                backbone_stats[backbone]['best_acc'], 
                trial.values[0] if trial.values else 0
            )
    
    print(f"🏗️ Architecture Performance Summary:")
    print("-" * 40)
    for backbone, stats in backbone_stats.items():
        avg_co2 = stats['total_co2'] / stats['count']
        efficiency = stats['best_acc'] / max(avg_co2, 0.1)
        print(f"   {backbone}:")
        print(f"     • Trials: {stats['count']}")
        print(f"     • Best Accuracy: {stats['best_acc']:.4f}")
        print(f"     • Avg CO2 per Trial: {avg_co2:.1f}g")
        print(f"     • Efficiency: {efficiency:.6f} acc/g")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["fashion", "flowers", "emotions", "skin_cancer"])
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), 
                       help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    parser.add_argument("--co2-region", type=str, default="DE", 
                       help="Region for CO2 calculation (DE, US, FR, etc.)")
    parser.add_argument("--disable-co2", action="store_true", 
                       help="Disable CO2 tracking (for comparison)")
    parser.add_argument("--create-detailed-plots", action="store_true", 
                       help="Create additional detailed visualizations")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Initialize CO2 tracking
    if not args.disable_co2:
        print(f"🌱 Initializing CO2 tracking for region: {args.co2_region}")
        co2_tracker = CO2Tracker(region=args.co2_region, tracking_interval=0.5)
        visualizer = EnhancedVisualizer(co2_tracker)
        co2_tracker.start_tracking("automl_optimization")
    else:
        print("⚠️ CO2 tracking disabled")
        co2_tracker = None
        visualizer = None

    # Dataset selection
    dataset_mapping = {
        "fashion": FashionDataset,
        "flowers": FlowersDataset,
        "emotions": EmotionsDataset,
        "skin_cancer": SkinCancerDataset
    }
    
    dataset_class = dataset_mapping[args.dataset]
    
    print(f"🎯 Selected dataset: {args.dataset} ({dataset_class.__name__})")
    print(f"📊 Number of classes: {dataset_class.num_classes}")
    print(f"🖼️ Image dimensions: {dataset_class.width}x{dataset_class.height}x{dataset_class.channels}")
    
    mean, std = calculate_mean_std(dataset_class)
    
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="train", backbone_name=default_backbone)

    # Load and split dataset
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len], 
                               generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))

    print(f"🔍 Starting Zero-Cost Proxy candidate generation...")
    
    # Set tracking phase
    if co2_tracker:
        co2_tracker.set_phase("zero_cost_proxy_search")

    # Run Zero-Cost Proxy search
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, 
                                    top_k=10, num_classes=dataset_class.num_classes)
    top_k_candidates = zcc.get_top_k_candidates()
    
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        print(f"[{i+1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}")
    
    # Set tracking phase for optimization
    if co2_tracker:
        co2_tracker.set_phase("hyperparameter_optimization")
    
    # Reference points for NSGAIII (accuracy, f1, time)
    reference_points = np.array([
        [1, 1, 0],      # High accuracy, high F1, low time
        [1, 0, 0],      # High accuracy only
        [0, 1, 0],      # High F1 only
        [0, 0, 1],      # Low time only
        [0.5, 0.5, 0.5] # Balanced
    ])

    # Multi-objective optimization with NSGA-III
    sampler = NSGAIIISampler(
        population_size=40,
        mutation_prob=0.2,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )

    # Create study with three objectives: maximize accuracy, maximize F1, minimize time
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
    )
    
    print(f"🚀 Starting {args.n_trials} trials of multi-objective optimization...")
    print(f"🎯 Objectives: Maximize Accuracy, Maximize F1, Minimize Training Time")
    
    # Run optimization
    study.optimize(
        lambda trial: enhanced_optuna_objective(
            trial,
            dataset_class=dataset_class,
            seed=args.seed,
            top_k_candidates=top_k_candidates,
        ), 
        n_trials=args.n_trials
    )

    # Set final tracking phase
    if co2_tracker:
        co2_tracker.set_phase("final_model_training")

    # Analysis and results
    pareto_trials = study.best_trials
    print(f"\n✅ Optimization completed!")
    print(f"🏆 Found {len(pareto_trials)} Pareto-optimal solutions:")
    
    for i, t in enumerate(pareto_trials[:5]):  # Show top 5
        co2_info = ""
        if 'co2_emissions' in t.user_attrs:
            co2_info = f", CO2: {t.user_attrs['co2_emissions']:.1f}g"
        print(f"  [{i+1}] Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, "
              f"Time: {t.values[2]:.1f}s{co2_info}")

    # Select best model (highest accuracy from Pareto front)
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) 
                       for i, c in enumerate(top_k_candidates)}
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]
    
    print(f"\n🏅 Training final model with best configuration:")
    print(f"   Architecture: {backbone}")
    print(f"   Parameters: {best_params}")
    
    # Train final model
    final_automl = EnhancedAutoML(
        seed=args.seed,
        num_layers_to_freeze=0,
        lr=best_params.get("lr", 0.001),
        use_augmentation=True,
        backbone=backbone,
        batch_size=best_params.get("batch_size", 32),
        epochs=final_epochs,
        optimizer=best_params.get("optimizer", "adam"),
        custom_head=head,
        co2_tracker=co2_tracker,
        visualizer=visualizer,
        enable_co2_tracking=not args.disable_co2
    )
    
    final_automl.fit(dataset_class, subsample=None)
    test_preds, test_labels = final_automl.predict(dataset_class)
    
    # Save predictions
    output_path = Path("final_test_preds.npy") if args.dataset == "skin_cancer" else args.output_path
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    print(f"✅ Predictions saved to {output_path}")
    
    # Final evaluation
    if not np.isnan(test_labels).any():
        final_acc = accuracy_score(test_labels, test_preds)
        final_f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"🎯 Final test performance: Accuracy = {final_acc:.4f}, F1 = {final_f1:.4f}")
    else:
        print(f"ℹ️ No test labels available for {dataset_class.__name__}")

    # Stop CO2 tracking and create visualizations
    if co2_tracker:
        co2_summary = co2_tracker.stop_tracking()
        print(f"🌱 CO2 tracking completed!")
        
        # Create all visualizations
        create_enhanced_visualizations(study, args.dataset, co2_tracker, visualizer)
        
        # Save CO2 data
        co2_tracker.save_data(f"{args.dataset}_co2_tracking.json")
        
        # Create detailed training visualizations
        if args.create_detailed_plots:
            final_automl.create_training_visualizations(f"{args.dataset}_training_details.png")
    
    # Print comprehensive summary
    print_enhanced_summary(study, co2_tracker, args.dataset)
    
    print(f"\n🎉 Enhanced AutoML pipeline completed successfully!")
    print(f"📁 Check the generated files for detailed analysis and visualizations.")
    
    if co2_tracker:
        co2_final = co2_tracker.get_summary()
        print(f"🌍 Final environmental impact: {co2_final.get('total_co2_g', 0):.1f}g CO2 "
              f"≈ {co2_final.get('car_equivalent_km', 0):.2f}km driving")