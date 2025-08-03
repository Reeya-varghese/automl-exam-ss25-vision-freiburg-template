import argparse
import logging
from pathlib import Path
from typing import Any, Tuple
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from Plots import show_class_distribution_cli
from collections import namedtuple
from RandAug import AugmentDataset, compute_class_distribution
from CO2emission import (
    CarbonGPUTracker,
    get_architecture_efficiency_weight,
    get_enhanced_reference_points,
    get_progressive_config
)
from Zero_cost import ZeroCostCandidateGenerator
from training import AutoML
import optuna
from optuna.samplers import NSGAIIISampler
from Plots import (
    save_optuna_visualizations,
    save_accuracy_histogram,
    save_metric_curves,
    save_carbon_gpu_plots
)
from model import get_transforms
from utils import calculate_mean_std
from vision_datasets import FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset
import logging
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("codecarbon").propagate = False

logger = logging.getLogger(__name__)


def optuna_objective(
    trial: optuna.Trial,
    dataset_class: Any,
    seed: int = 42,
    
    top_k_candidates: list[dict[str, Any]] = None,
    carbon_budget_kg: float = 0.1,
    enable_progressive: bool = True,
    total_trials: int = 10 
    ) -> Tuple[float, float, float,float, float]:
    """
    Objective function for Optuna multi-objective optimization.
    Returns: accuracy, f1, training time, adjusted carbon, peak memory.
    """
    # Map candidate ID to corresponding backbone and head
    candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
    #
    STATIC_CANDIDATE_IDS = list(candidate_lookup.keys())
    STATIC_BATCH_SIZE = [64, 32, 16]

    # Tracking Crabon emissions and GPU usage
    tracker = CarbonGPUTracker(project_name=f"trial_{trial.number}")
    tracker.start_tracking(trial_id=trial.number)

    try:
      
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        progressive_config = get_progressive_config(trial.number, total_trials, enable_progressive)

        candidate_id = trial.suggest_categorical("candidate_id", STATIC_CANDIDATE_IDS)
        if candidate_id not in candidate_lookup:
            raise optuna.TrialPruned(f"Invalid candidate_id: {candidate_id}")
        backbone, head = candidate_lookup[candidate_id]


        if progressive_config['prefer_efficient_arch']:
            efficiency = get_architecture_efficiency_weight(backbone)
            if efficiency < 0.8:
                raise optuna.TrialPruned(f"Backbone {backbone} below efficiency threshold.")        
   
        efficiency_weight = get_architecture_efficiency_weight(backbone)

        head = head.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
     
        batch_size = trial.suggest_categorical('batch_size', STATIC_BATCH_SIZE)

        #if progressive_config['min_batch_size'] > 16 and batch_size == 16:
         #   raise optuna.TrialPruned("Batch size 16 disallowed by progressive config.")

        epochs = trial.suggest_int('epochs',8, progressive_config['max_epochs'])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        use_augmentation = trial.suggest_categorical('use_augmentation', [True])

        
      
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
        print(head)

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
        raise optuna.TrialPruned(f"Failed due to error: {e}")
    
    sustainability_metrics = tracker.stop_tracking()
    efficiency_weight = get_architecture_efficiency_weight(backbone)
    adjusted_carbon = sustainability_metrics['emissions_kg'] / efficiency_weight

    trial.set_user_attr("emissions_kg", sustainability_metrics['emissions_kg'])
    trial.set_user_attr("training_time", sustainability_metrics['training_time'])
    trial.set_user_attr("peak_gpu_memory_gb", sustainability_metrics['peak_gpu_memory_gb'])
    trial.set_user_attr("adjusted_carbon", adjusted_carbon)
    trial.set_user_attr("efficiency_weight", efficiency_weight)

    if adjusted_carbon > carbon_budget_kg:
        trial.set_user_attr("carbon_budget_exceeded", True)
        raise optuna.TrialPruned(f"Carbon budget exceeded: {adjusted_carbon:.4f} kg")

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
    
    print(f"🌱 Enhanced AutoML with Combined Carbon Strategies - {args.dataset.upper()}")
    print(f"📊 Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"🔄 Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
      
    mean, std = calculate_mean_std(dataset_class)

     
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
   
    raw_dataset = dataset_class(root="./data", split='train', download=True, transform=None)

    class_names = raw_dataset.classes if hasattr(raw_dataset, "classes") else None
    show_class_distribution_cli(raw_dataset, class_names, title="Before Augmentation")


    augmented_dataset, sampler = AugmentDataset(
        dataset=raw_dataset,
        image_size=(224, 224),
        grayscale=(dataset_class.channels == 1),
        n=2,
        m=9,
        mean=mean,
        std=std
    )


    sample_loader = DataLoader(augmented_dataset, sampler=sampler, batch_size=8)
    real_input, real_target = next(iter(sample_loader))



    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10, num_classes=dataset_class.num_classes)
    top_k_candidates = zcc.get_top_k_candidates()
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")

    print("✅ Top-K candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        print(f"[{i+1}] Backbone: {c['backbone']}, "
      f"Jacobian: {c['jacobian_score']:.4f}, "
      f"GradNorm: {c['gradnorm_score']:.4f}, "
      f"Total: {c['combined_score']:.4f}")


    reference_points = get_enhanced_reference_points()



    opsampler = NSGAIIISampler(
        population_size=50,
        mutation_prob=0.15,
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=args.seed,
        reference_points=reference_points
    )
   

    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize","minimize", "minimize"],
        sampler=opsampler,
        
    )
    global_tracker = CarbonGPUTracker("global_optimization")
    global_tracker.start_tracking()
    
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        carbon_budget_kg=args.carbon_budget,
        top_k_candidates=top_k_candidates,
    ), n_trials=args.n_trials)

    global_metrics=global_tracker.stop_tracking()

    pareto_trials = study.best_trials
    print(f"\n✅Pareto-optimal solutions ({len(pareto_trials)}):")
    for t in pareto_trials:
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        print(f"✅ Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.2f}s, "
            f"Carbon: {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), "
            f"GPU: {t.values[4]:.2f}GB, "
            f"Eff: {efficiency:.1f} | {t.params}")
    
    
    print(f"\n🎯 SOLUTION ANALYSIS:")
    print("="*60)

    carbon_efficient = [t for t in pareto_trials if t.values[3] < args.carbon_budget * 0.5]
    high_performance = [t for t in pareto_trials if t.values[0] > 0.85 and t.values[1] > 0.8]
    balanced = [t for t in pareto_trials if t not in carbon_efficient and t not in high_performance]
    
    print(f"🌱 Carbon Efficient ({len(carbon_efficient)}): Low carbon footprint solutions")
    for t in carbon_efficient[:3]:  # Show top 3
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, Carbon={t.values[3]:.4f}kg")
    
    print(f"🚀 High Performance ({len(high_performance)}): Best accuracy solutions")
    for t in high_performance[:3]:  # Show top 3
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, F1={t.values[1]:.3f}")
    
    print(f"⚖️ Balanced ({len(balanced)}): Good trade-offs")
    # Retrain AutoML with best config and save predictions
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]

    print(f"\n🎯 Selected Best Accuracy Solution: Trial {best_acc_trial.number}")
    print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
    print(f"   Sustainability: Carbon={best_acc_trial.values[3]:.4f}kg, GPU={best_acc_trial.values[4]:.2f}GB")
    
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
    else :
        output_path = args.output_path    
            
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    print(f"✅Predictions for best-accuracy config saved to {args.output_path}")
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"✅Accuracy of best config on test set: {acc:.4f} (F1: {f1:.4f})")
    else:
        print(f"No test split for dataset '{dataset_class.__name__}'")
    print("✅AutoML training completed successfully!")
    
    
    total_emissions = global_metrics['emissions_kg'] + final_metrics['emissions_kg']
    efficiency_used = get_architecture_efficiency_weight(backbone)
    carbon_saved_estimate = total_emissions * (1 - efficiency_used) if efficiency_used < 1.0 else 0
    
    print(f"\n🌱 SUSTAINABILITY SUMMARY:")
    print(f"   Total Carbon Footprint: {total_emissions:.4f} kg CO2eq")
    print(f"   HPO Phase: {global_metrics['emissions_kg']:.4f} kg")
    print(f"   Final Training: {final_metrics['emissions_kg']:.4f} kg")
    print(f"   Architecture Efficiency: {efficiency_used:.1f} (1.0 = most efficient)")
    print(f"   Estimated Carbon Saved: {carbon_saved_estimate:.4f} kg CO2eq")
    print(f"   Peak GPU Memory: {max(global_metrics['peak_gpu_memory_gb'], final_metrics['peak_gpu_memory_gb']):.2f} GB")
    
    
    # Save Optuna plots
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    save_carbon_gpu_plots(study, filename_prefix="optuna")
    print("✅Optuna visualizations saved successfully!")