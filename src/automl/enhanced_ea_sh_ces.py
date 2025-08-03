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
import seaborn as sns
import pandas as pd
import json

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

# Import enhanced CO2 tracking
from co2_tracker import CO2Tracker, EnhancedVisualizer

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

# NEW: Enhanced CO2 Analysis Class
class CarbonAwareAnalyzer:
    """Analyzes carbon reduction strategies and tracks detailed emissions"""
    
    def __init__(self, dataset_name, enhanced_co2_tracker, carbon_budget_kg):
        self.dataset_name = dataset_name
        self.enhanced_tracker = enhanced_co2_tracker
        self.carbon_budget_kg = carbon_budget_kg
        self.stage_emissions = {}
        self.trial_emissions = []
        self.carbon_savings = []
        self.efficiency_timeline = []
        self.budget_compliance = []
        
    def log_stage_emission(self, stage_name, emissions_kg):
        """Log emissions for each stage of the process"""
        self.stage_emissions[stage_name] = emissions_kg
        print(f"🌱 Stage '{stage_name}': {emissions_kg:.4f} kg CO2")
        
    def log_trial_emission(self, trial_number, emissions_kg, was_pruned, efficiency_weight):
        """Log per-trial emissions with pruning and efficiency info"""
        self.trial_emissions.append({
            'trial': trial_number,
            'emissions_kg': emissions_kg,
            'was_pruned': was_pruned,
            'efficiency_weight': efficiency_weight,
            'cumulative_emissions': sum([t['emissions_kg'] for t in self.trial_emissions]) + emissions_kg,
            'budget_remaining': self.carbon_budget_kg - (sum([t['emissions_kg'] for t in self.trial_emissions]) + emissions_kg)
        })
        
    def calculate_carbon_savings(self, baseline_emissions_kg):
        """Calculate carbon savings compared to baseline approach"""
        current_total = sum([t['emissions_kg'] for t in self.trial_emissions])
        savings = baseline_emissions_kg - current_total
        savings_percentage = (savings / baseline_emissions_kg) * 100 if baseline_emissions_kg > 0 else 0
        
        self.carbon_savings.append({
            'baseline_kg': baseline_emissions_kg,
            'actual_kg': current_total,
            'savings_kg': savings,
            'savings_percentage': savings_percentage
        })
        
        return savings, savings_percentage
        
    def create_comprehensive_carbon_analysis(self, study, save_prefix="carbon_analysis"):
        """Create comprehensive carbon analysis plots"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Stage-wise Emissions Breakdown
        self._create_stage_emissions_plot(save_prefix)
        
        # 2. Trial-by-trial Carbon Efficiency
        self._create_trial_efficiency_plot(save_prefix)
        
        # 3. Carbon Budget Compliance
        self._create_budget_compliance_plot(save_prefix)
        
        # 4. Carbon Reduction Strategies Effectiveness
        self._create_strategy_effectiveness_plot(study, save_prefix)
        
        # 5. Comprehensive Carbon Dashboard
        self._create_carbon_dashboard(study, save_prefix)
        
    def _create_stage_emissions_plot(self, save_prefix):
        """Create detailed stage emissions breakdown"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stage emissions pie chart
        if self.stage_emissions:
            stages = list(self.stage_emissions.keys())
            emissions = list(self.stage_emissions.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
            
            wedges, texts, autotexts = ax1.pie(emissions, labels=stages, autopct='%1.2f%%',
                                              colors=colors, startangle=90)
            ax1.set_title('🌱 CO2 Emissions by Stage', fontsize=14, fontweight='bold')
            
            # Add emission values to labels
            for i, (stage, emission) in enumerate(zip(stages, emissions)):
                texts[i].set_text(f'{stage}\n({emission:.4f} kg)')
        
        # Cumulative emissions over stages
        if self.stage_emissions:
            stages = list(self.stage_emissions.keys())
            emissions = list(self.stage_emissions.values())
            cumulative = np.cumsum(emissions)
            
            bars = ax2.bar(range(len(stages)), emissions, alpha=0.7, color=colors)
            ax2.plot(range(len(stages)), cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative')
            ax2.set_xlabel('Process Stages')
            ax2.set_ylabel('CO2 Emissions (kg)')
            ax2.set_title('📈 Cumulative CO2 Emissions', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(stages)))
            ax2.set_xticklabels([s[:10] + '...' if len(s) > 10 else s for s in stages], rotation=45)
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Add value labels on bars
            for bar, emission in zip(bars, emissions):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emissions)*0.01,
                        f'{emission:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_stage_emissions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💚 Stage emissions analysis saved to {save_prefix}_stage_emissions.png")
        
    def _create_trial_efficiency_plot(self, save_prefix):
        """Create trial-by-trial efficiency analysis"""
        if not self.trial_emissions:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        trials = [t['trial'] for t in self.trial_emissions]
        emissions = [t['emissions_kg'] for t in self.trial_emissions]
        cumulative = [t['cumulative_emissions'] for t in self.trial_emissions]
        efficiency_weights = [t['efficiency_weight'] for t in self.trial_emissions]
        pruned = [t['was_pruned'] for t in self.trial_emissions]
        
        # 1. Per-trial emissions with pruning info
        colors = ['red' if p else 'green' for p in pruned]
        bars = ax1.bar(trials, emissions, color=colors, alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('CO2 Emissions (kg)')
        ax1.set_title('🔥 Per-Trial CO2 Emissions\n(Red=Pruned, Green=Completed)', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add efficiency weight annotations
        for i, (bar, eff) in enumerate(zip(bars, efficiency_weights)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emissions)*0.01,
                    f'{eff:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Cumulative emissions vs budget
        ax2.plot(trials, cumulative, 'b-', linewidth=2, marker='o', label='Cumulative CO2')
        ax2.axhline(y=self.carbon_budget_kg, color='red', linestyle='--', linewidth=2, label=f'Budget ({self.carbon_budget_kg} kg)')
        ax2.fill_between(trials, 0, self.carbon_budget_kg, alpha=0.2, color='green', label='Within Budget')
        ax2.fill_between(trials, self.carbon_budget_kg, max(cumulative + [self.carbon_budget_kg]), alpha=0.2, color='red', label='Over Budget')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Cumulative CO2 (kg)')
        ax2.set_title('📊 Carbon Budget Compliance', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Efficiency weight impact
        scatter = ax3.scatter(efficiency_weights, emissions, c=trials, cmap='viridis', s=60, alpha=0.7)
        ax3.set_xlabel('Architecture Efficiency Weight')
        ax3.set_ylabel('Trial CO2 Emissions (kg)')
        ax3.set_title('⚙️ Architecture Efficiency vs Emissions', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Trial Number')
        ax3.grid(alpha=0.3)
        
        # 4. Carbon savings over time
        if len(self.carbon_savings) > 0:
            savings_data = self.carbon_savings[-1]  # Latest calculation
            categories = ['Baseline\nApproach', 'Carbon-Aware\nApproach', 'Savings']
            values = [savings_data['baseline_kg'], savings_data['actual_kg'], savings_data['savings_kg']]
            colors_bar = ['lightcoral', 'lightgreen', 'gold']
            
            bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8)
            ax4.set_ylabel('CO2 Emissions (kg)')
            ax4.set_title(f'💰 Carbon Savings: {savings_data["savings_percentage"]:.1f}%', fontweight='bold')
            
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{val:.4f} kg', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_trial_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"⚡ Trial efficiency analysis saved to {save_prefix}_trial_efficiency.png")
        
    def _create_budget_compliance_plot(self, save_prefix):
        """Create carbon budget compliance visualization"""
        if not self.trial_emissions:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        trials = [t['trial'] for t in self.trial_emissions]
        budget_remaining = [t['budget_remaining'] for t in self.trial_emissions]
        cumulative = [t['cumulative_emissions'] for t in self.trial_emissions]
        
        # 1. Budget remaining over trials
        colors = ['green' if br >= 0 else 'red' for br in budget_remaining]
        bars = ax1.bar(trials, budget_remaining, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Budget Remaining (kg CO2)')
        ax1.set_title('💳 Carbon Budget Status\n(Green=Within Budget, Red=Exceeded)', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # 2. Budget utilization percentage
        budget_used_pct = [(self.carbon_budget_kg - br) / self.carbon_budget_kg * 100 for br in budget_remaining]
        ax2.plot(trials, budget_used_pct, 'b-', linewidth=2, marker='s', markersize=6)
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Budget Limit')
        ax2.fill_between(trials, 0, 100, alpha=0.2, color='green', label='Safe Zone')
        ax2.fill_between(trials, 100, max(budget_used_pct + [100]), alpha=0.2, color='red', label='Over Budget')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Budget Used (%)')
        ax2.set_title('📈 Budget Utilization Progress', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_budget_compliance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💳 Budget compliance analysis saved to {save_prefix}_budget_compliance.png")
        
    def _create_strategy_effectiveness_plot(self, study, save_prefix):
        """Analyze effectiveness of different carbon reduction strategies"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract strategy data from completed trials
        strategies_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                backbone = trial.user_attrs.get('backbone', 'unknown')
                efficiency_weight = trial.user_attrs.get('efficiency_weight', 1.0)
                emissions = trial.user_attrs.get('emissions_kg', 0)
                adjusted_carbon = trial.values[3] if len(trial.values) > 3 else emissions
                accuracy = trial.values[0] if trial.values else 0
                
                strategies_data.append({
                    'backbone': backbone,
                    'efficiency_weight': efficiency_weight,
                    'emissions': emissions,
                    'adjusted_carbon': adjusted_carbon,
                    'accuracy': accuracy,
                    'trial': trial.number
                })
        
        if not strategies_data:
            return
            
        df = pd.DataFrame(strategies_data)
        
        # 1. Strategy effectiveness by architecture
        arch_efficiency = df.groupby('backbone').agg({
            'emissions': 'mean',
            'accuracy': 'mean',
            'efficiency_weight': 'mean'
        }).reset_index()
        
        x = np.arange(len(arch_efficiency))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, arch_efficiency['emissions'], width, label='Avg CO2 (kg)', alpha=0.8, color='lightcoral')
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, arch_efficiency['accuracy'], width, label='Avg Accuracy', alpha=0.8, color='lightgreen')
        
        ax1.set_xlabel('Architecture')
        ax1.set_ylabel('CO2 Emissions (kg)', color='red')
        ax1_twin.set_ylabel('Accuracy', color='green')
        ax1.set_title('🏗️ Architecture Strategy Effectiveness', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(arch_efficiency['backbone'], rotation=45)
        
        # Add efficiency weight annotations
        for i, (bar, eff) in enumerate(zip(bars1, arch_efficiency['efficiency_weight'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'Eff: {eff:.1f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        # 2. Carbon efficiency over time
        df_sorted = df.sort_values('trial')
        carbon_efficiency = df_sorted['accuracy'] / (df_sorted['emissions'] * 1000 + 1)  # acc per gram
        rolling_efficiency = pd.Series(carbon_efficiency).rolling(window=3, min_periods=1).mean()
        
        ax2.plot(df_sorted['trial'], rolling_efficiency, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Carbon Efficiency (Acc/gram CO2)')
        ax2.set_title('📈 Carbon Efficiency Improvement', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Progressive strategy impact
        if hasattr(self, 'progressive_enabled') and self.progressive_enabled:
            early_trials = df[df['trial'] < len(df) * 0.3]
            late_trials = df[df['trial'] >= len(df) * 0.7]
            
            categories = ['Early Trials\n(Efficiency Focus)', 'Late Trials\n(Performance Focus)']
            avg_emissions = [early_trials['emissions'].mean(), late_trials['emissions'].mean()]
            avg_accuracy = [early_trials['accuracy'].mean(), late_trials['accuracy'].mean()]
            
            x = np.arange(len(categories))
            bars1 = ax3.bar(x - 0.2, avg_emissions, 0.4, label='Avg CO2 (kg)', alpha=0.8, color='lightcoral')
            ax3_twin = ax3.twinx()
            bars2 = ax3_twin.bar(x + 0.2, avg_accuracy, 0.4, label='Avg Accuracy', alpha=0.8, color='lightgreen')
            
            ax3.set_ylabel('CO2 Emissions (kg)', color='red')
            ax3_twin.set_ylabel('Accuracy', color='green')
            ax3.set_title('🔄 Progressive Strategy Impact', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
        
        # 4. Strategy comparison summary
        strategy_summary = {
            'Total Trials': len(df),
            'Avg CO2 per Trial': f"{df['emissions'].mean():.4f} kg",
            'Best Accuracy': f"{df['accuracy'].max():.4f}",
            'Most Efficient Arch': arch_efficiency.loc[arch_efficiency['efficiency_weight'].idxmax(), 'backbone'],
            'Carbon Savings': f"{self.carbon_savings[-1]['savings_percentage']:.1f}%" if self.carbon_savings else "N/A"
        }
        
        ax4.axis('off')
        summary_text = "🎯 STRATEGY SUMMARY\n" + "="*25 + "\n"
        for key, value in strategy_summary.items():
            summary_text += f"{key}: {value}\n"
        
        ax4.text(0.1, 0.8, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_strategy_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"🎯 Strategy effectiveness analysis saved to {save_prefix}_strategy_effectiveness.png")
        
    def _create_carbon_dashboard(self, study, save_prefix):
        """Create comprehensive carbon-aware AutoML dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Use the enhanced visualizer's dashboard but add carbon-specific elements
        if hasattr(self, 'enhanced_tracker') and self.enhanced_tracker:
            # Call the original comprehensive dashboard
            self.enhanced_tracker.create_comprehensive_dashboard(f'{save_prefix}_enhanced_dashboard.png')
        
        # Create additional carbon-aware specific dashboard
        plt.subplot(2, 3, 1)
        self._plot_carbon_reduction_timeline()
        
        plt.subplot(2, 3, 2)
        self._plot_efficiency_vs_performance(study)
        
        plt.subplot(2, 3, 3)
        self._plot_budget_vs_results()
        
        plt.subplot(2, 3, 4)
        self._plot_pruning_effectiveness()
        
        plt.subplot(2, 3, 5)
        self._plot_carbon_roi()
        
        plt.subplot(2, 3, 6)
        self._plot_environmental_impact()
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_carbon_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"🌍 Carbon-aware dashboard saved to {save_prefix}_carbon_dashboard.png")
        
    def _plot_carbon_reduction_timeline(self):
        """Plot carbon reduction over time"""
        if not self.trial_emissions:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=plt.gca().transAxes)
            return
            
        trials = [t['trial'] for t in self.trial_emissions]
        cumulative = [t['cumulative_emissions'] for t in self.trial_emissions]
        
        # Calculate what baseline would have been
        avg_emission = np.mean([t['emissions_kg'] for t in self.trial_emissions])
        baseline_cumulative = [avg_emission * (i + 1) for i in range(len(trials))]
        
        plt.plot(trials, baseline_cumulative, 'r--', label='Baseline Approach', linewidth=2)
        plt.plot(trials, cumulative, 'g-', label='Carbon-Aware Approach', linewidth=2)
        plt.fill_between(trials, cumulative, baseline_cumulative, alpha=0.3, color='green', label='CO2 Saved')
        
        plt.xlabel('Trial Number')
        plt.ylabel('Cumulative CO2 (kg)')
        plt.title('🌱 Carbon Reduction Timeline', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
    def _plot_efficiency_vs_performance(self, study):
        """Plot efficiency weights vs performance"""
        efficiency_weights = []
        accuracies = []
        emissions = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                eff = trial.user_attrs.get('efficiency_weight', 1.0)
                acc = trial.values[0] if trial.values else 0
                emiss = trial.user_attrs.get('emissions_kg', 0)
                
                efficiency_weights.append(eff)
                accuracies.append(acc)
                emissions.append(emiss)
        
        if efficiency_weights:
            scatter = plt.scatter(efficiency_weights, accuracies, c=emissions, cmap='RdYlGn_r', s=60, alpha=0.7)
            plt.colorbar(scatter, label='CO2 Emissions (kg)')
            plt.xlabel('Architecture Efficiency Weight')
            plt.ylabel('Accuracy')
            plt.title('⚙️ Efficiency vs Performance', fontweight='bold')
            plt.grid(alpha=0.3)
        
    def _plot_budget_vs_results(self):
        """Plot budget compliance vs results"""
        if not self.trial_emissions:
            return
            
        budget_used = [(self.carbon_budget_kg - t['budget_remaining']) / self.carbon_budget_kg * 100 
                      for t in self.trial_emissions]
        trials = [t['trial'] for t in self.trial_emissions]
        
        colors = ['green' if bu <= 100 else 'red' for bu in budget_used]
        plt.bar(trials, budget_used, color=colors, alpha=0.7)
        plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Budget Limit')
        plt.xlabel('Trial Number')
        plt.ylabel('Budget Used (%)')
        plt.title('💳 Budget vs Results', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
    def _plot_pruning_effectiveness(self):
        """Plot pruning effectiveness"""
        if not self.trial_emissions:
            return
            
        completed = sum(1 for t in self.trial_emissions if not t['was_pruned'])
        pruned = sum(1 for t in self.trial_emissions if t['was_pruned'])
        
        if completed + pruned > 0:
            plt.pie([completed, pruned], labels=['Completed', 'Pruned'], 
                   autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            plt.title('✂️ Trial Pruning Effectiveness', fontweight='bold')
        
    def _plot_carbon_roi(self):
        """Plot carbon return on investment"""
        if not self.trial_emissions or not self.carbon_savings:
            return
            
        savings = self.carbon_savings[-1]
        roi_metrics = {
            'CO2 Saved (kg)': savings['savings_kg'],
            'Efficiency Gain (%)': savings['savings_percentage'],
            'Trials Completed': len([t for t in self.trial_emissions if not t['was_pruned']]),
            'Budget Compliance': 'Yes' if self.trial_emissions[-1]['budget_remaining'] >= 0 else 'No'
        }
        
        plt.axis('off')
        roi_text = "💰 CARBON ROI\n" + "="*15 + "\n"
        for key, value in roi_metrics.items():
            roi_text += f"{key}: {value}\n"
        
        plt.text(0.1, 0.8, roi_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
    def _plot_environmental_impact(self):
        """Plot environmental impact context"""
        if not self.stage_emissions:
            return
            
        total_kg = sum(self.stage_emissions.values())
        
        # Environmental context
        contexts = {
            'This AutoML': total_kg,
            'Car (10km)': 1.2,  # kg CO2
            'Train (100km)': 1.4,
            'Flight (1000km)': 90.0
        }
        
        names = list(contexts.keys())
        values = list(contexts.values())
        colors = ['red' if name == 'This AutoML' else 'lightblue' for name in names]
        
        bars = plt.bar(names, values, color=colors, alpha=0.8)
        plt.ylabel('CO2 Emissions (kg)')
        plt.title('🌍 Environmental Context', fontweight='bold')
        plt.xticks(rotation=45)
        
        # Highlight our emissions
        bars[0].set_edgecolor('darkred')
        bars[0].set_linewidth(3)

    def save_carbon_analysis_report(self, save_path="carbon_analysis_report.json"):
        """Save detailed carbon analysis report"""
        report = {
            'dataset': self.dataset_name,
            'carbon_budget_kg': self.carbon_budget_kg,
            'stage_emissions': self.stage_emissions,
            'trial_emissions': self.trial_emissions,
            'carbon_savings': self.carbon_savings,
            'total_emissions_kg': sum(self.stage_emissions.values()),
            'total_trials': len(self.trial_emissions),
            'trials_completed': len([t for t in self.trial_emissions if not t['was_pruned']]),
            'trials_pruned': len([t for t in self.trial_emissions if t['was_pruned']]),
            'budget_compliance': self.trial_emissions[-1]['budget_remaining'] >= 0 if self.trial_emissions else True,
            'average_efficiency_weight': np.mean([t['efficiency_weight'] for t in self.trial_emissions]) if self.trial_emissions else 0,
            'carbon_efficiency_score': sum([t['emissions_kg'] for t in self.trial_emissions]) / len(self.trial_emissions) if self.trial_emissions else 0
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📊 Carbon analysis report saved to {save_path}")
        
        return report

# NEW: Architecture efficiency weights based on research
def get_architecture_efficiency_weight(backbone_name):
    """Efficiency weights based on parameter count and energy research"""
    efficiency_weights = {
        'resnet18': 1.0,        # Most efficient (11M params)
        'efficientnet_b0': 0.8, # Good efficiency (5M params, but complex ops)
        'resnet50': 0.6,        # Moderate efficiency (25M params)
        'vit_base_patch16_224': 0.4  # Least efficient (86M params, attention heavy)
    }
    return efficiency_weights.get(backbone_name, 0.5)

# NEW: Progressive training strategy
def get_progressive_config(trial_number, total_trials, enable_progressive=True):
    """Start with efficient configs, gradually allow more complex ones"""
    if not enable_progressive:
        return {
            'max_epochs': 8,  # Original default
            'min_batch_size': 32,
            'prefer_efficient_arch': False
        }
    
    progress_ratio = trial_number / max(total_trials, 1)  # Fix: avoid division by zero
    
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

# NEW: Enhanced reference points for carbon-aware optimization
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
    carbon_analyzer: CarbonAwareAnalyzer = None,
    enhanced_co2_tracker: CO2Tracker = None,
    enhanced_visualizer: EnhancedVisualizer = None
    ) -> Tuple[float, float, float, float, float]:
    """Enhanced objective with carbon and GPU tracking + detailed CO2 analysis"""
    
    # Start carbon tracking for this trial
    tracker = CarbonGPUTracker(f"trial_{trial.number}")
    tracker.start_tracking(trial.number)
    
    # Enhanced CO2 tracking phase
    if enhanced_co2_tracker:
        enhanced_co2_tracker.set_phase(f"trial_{trial.number}_start")
    
    try:
        # Your existing hyperparameter suggestions (unchanged)
        candidate_lookup = {
            f"{c['backbone']}_{i}": (c['backbone'], c['head'])
            for i, c in enumerate(top_k_candidates)
        }
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        # Progressive configuration
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
        
        print(f"\n🔥 Trial {trial.number}: {backbone} (Efficiency: {efficiency_weight:.1f})")
        print(f"⚙️ Config: LR={lr:.6f}, Batch={batch_size}, Epochs={epochs}, Optimizer={optimizer}")
        
        # Enhanced CO2 tracking for training phase
        if enhanced_co2_tracker:
            enhanced_co2_tracker.set_phase(f"trial_{trial.number}_training")
        
        # Your existing AutoML training (unchanged)
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
        
        # Enhanced CO2 tracking for evaluation phase
        if enhanced_co2_tracker:
            enhanced_co2_tracker.set_phase(f"trial_{trial.number}_evaluation")
        
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
        sustainability_metrics = tracker.stop_tracking()
        
        # Log failed trial to carbon analyzer
        if carbon_analyzer:
            carbon_analyzer.log_trial_emission(
                trial.number, 
                sustainability_metrics['emissions_kg'], 
                was_pruned=True, 
                efficiency_weight=0.5
            )
        
        return 0.0, 0.0, 999.0, 999.0, 999.0
    
    # Get sustainability metrics
    sustainability_metrics = tracker.stop_tracking()
    
    # Apply architecture efficiency weighting to carbon cost
    adjusted_carbon = sustainability_metrics['emissions_kg'] / efficiency_weight
    
    # Enhanced logging with detailed CO2 info
    print(f"🌱 Trial CO2: {sustainability_metrics['emissions_kg']:.4f} kg (adjusted: {adjusted_carbon:.4f} kg)")
    print(f"⚡ Power/GPU: {sustainability_metrics['peak_gpu_memory_gb']:.2f} GB peak")
    print(f"📊 Results: Acc={acc:.4f}, F1={f1:.4f}, Time={training_time:.1f}s")
    
    # Add sustainability tracking to user attributes
    trial.set_user_attr("emissions_kg", sustainability_metrics['emissions_kg'])
    trial.set_user_attr("adjusted_carbon", adjusted_carbon)
    trial.set_user_attr("peak_gpu_memory_gb", sustainability_metrics['peak_gpu_memory_gb'])
    trial.set_user_attr("efficiency_weight", efficiency_weight)
    trial.set_user_attr("progressive_config", progressive_config)
    
    # Enhanced logging to carbon analyzer
    if carbon_analyzer:
        carbon_analyzer.log_trial_emission(
            trial.number, 
            sustainability_metrics['emissions_kg'], 
            was_pruned=False, 
            efficiency_weight=efficiency_weight
        )
    
    # Enhanced logging to visualizer
    if enhanced_visualizer:
        enhanced_visualizer.log_trial({
            'trial_id': trial.number,
            'params': trial.params,
            'values': [acc, f1, training_time, adjusted_carbon, sustainability_metrics['peak_gpu_memory_gb']],
            'co2_emissions': sustainability_metrics['emissions_kg'],
            'co2_efficiency': acc / (sustainability_metrics['emissions_kg'] * 1000 + 1),
            'backbone': backbone,
            'head_type': head.__class__.__name__,
            'efficiency_weight': efficiency_weight
        })
    
    # Carbon budget constraint with efficiency weighting
    if adjusted_carbon > carbon_budget_kg:
        trial.set_user_attr("carbon_budget_exceeded", True)
        print(f"🚫 Carbon budget exceeded: {adjusted_carbon:.4f} kg > {carbon_budget_kg} kg")
        
        # Log pruned trial
        if carbon_analyzer:
            carbon_analyzer.log_trial_emission(
                trial.number, 
                sustainability_metrics['emissions_kg'], 
                was_pruned=True, 
                efficiency_weight=efficiency_weight
            )
        
        raise optuna.TrialPruned(f"Carbon budget exceeded: {adjusted_carbon:.4f} kg (efficiency-adjusted)")
    
    # Calculate efficiency metrics for reporting
    carbon_efficiency = acc / (sustainability_metrics['emissions_kg'] * 1000 + 1)  # acc per gram
    print(f"♻️ Carbon Efficiency: {carbon_efficiency:.4f} acc/gram CO2")
    
    # Return 5 objectives: accuracy, f1, time, adjusted_emissions, gpu_memory
    return (
        acc, 
        f1, 
        training_time,
        adjusted_carbon,  # Use efficiency-adjusted carbon cost
        sustainability_metrics['peak_gpu_memory_gb']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--dataset", type=str, required=True, choices=["fashion", "flowers", "emotions", "skin_cancer"],)
    parser.add_argument("--output-path", type=Path, default=Path("predictions.npy"), help="Path to save predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    # Carbon-aware parameters
    parser.add_argument("--carbon-budget", type=float, default=0.15, help="Carbon budget in kg CO2eq")
    parser.add_argument("--enable-progressive", action="store_true", help="Enable progressive training strategy")
    parser.add_argument("--baseline-emissions", type=float, default=0.5, help="Baseline emissions for comparison (kg)")
    parser.add_argument("--save-carbon-analysis", action="store_true", help="Save detailed carbon analysis")
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and errors.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    # Dataset selection (unchanged)
    dataset_mapping = {
        "fashion": FashionDataset,
        "flowers": FlowersDataset, 
        "emotions": EmotionsDataset,
        "skin_cancer": SkinCancerDataset
    }
    dataset_class = dataset_mapping[args.dataset]
    
    print(f"🌱 ENHANCED Carbon-Aware AutoML - {args.dataset.upper()}")
    print(f"📊 Carbon Budget: {args.carbon_budget} kg CO2eq")
    print(f"🔄 Progressive Training: {'Enabled' if args.enable_progressive else 'Disabled'}")
    print(f"📈 Baseline Comparison: {args.baseline_emissions} kg CO2eq")
    print(f"=" * 60)
    
    # Initialize enhanced CO2 tracking system
    enhanced_co2_tracker = CO2Tracker(region="DE", tracking_interval=0.5)
    enhanced_visualizer = EnhancedVisualizer(enhanced_co2_tracker)
    carbon_analyzer = CarbonAwareAnalyzer(args.dataset, enhanced_visualizer, args.carbon_budget)
    
    # Start global enhanced tracking
    enhanced_co2_tracker.start_tracking("carbon_aware_automl")
    carbon_analyzer.log_stage_emission("initialization", 0.001)  # Minimal initialization cost
    
    mean, std = calculate_mean_std(dataset_class)
    
    grayscale = dataset_class.channels == 1
    default_backbone = "resnet18" if grayscale else "vit_base_patch16_224"
    transform = get_transforms(mean, std, phase="train", backbone_name=default_backbone)

    # Load and split dataset (unchanged)
    enhanced_co2_tracker.set_phase("dataset_preparation")
    full_dataset = dataset_class(root="./data", split='train', download=True, transform=transform)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, _ = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))

    sample_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    real_input, real_target = next(iter(sample_loader))
    
    # Track dataset preparation stage
    dataset_prep_stats = enhanced_co2_tracker.get_current_stats()
    carbon_analyzer.log_stage_emission("dataset_preparation", dataset_prep_stats['total_co2_g'] / 1000)

    # Run Zero-Cost Proxy search
    enhanced_co2_tracker.set_phase("zero_cost_proxy_search")
    print(f"🔍 Starting Zero-Cost Proxy candidate generation...")
    zcc = ZeroCostCandidateGenerator(real_input, real_target, num_candidates=100, top_k=10, num_classes=dataset_class.num_classes)
    top_k_candidates = zcc.get_top_k_candidates()
    candidate_lookup = {f"{c['backbone']}_{i}": (c['backbone'], c['head']) for i, c in enumerate(top_k_candidates)}
    
    # Track zero-cost proxy stage
    zcp_stats = enhanced_co2_tracker.get_current_stats()
    carbon_analyzer.log_stage_emission("zero_cost_proxy", (zcp_stats['total_co2_g'] - dataset_prep_stats['total_co2_g']) / 1000)
    
    print(f"✅ Found {len(top_k_candidates)} top-k candidates based on Zero-Cost scores:")
    for i, c in enumerate(top_k_candidates):
        efficiency = get_architecture_efficiency_weight(c['backbone'])
        print(f"[{i+1}] Backbone: {c['backbone']}, Combined Score: {c['combined_score']:.4f}, Efficiency: {efficiency:.1f}")
    
    # Enhanced reference points for carbon-aware optimization
    reference_points = get_enhanced_reference_points()

    # Enhanced NSGA-III sampler with larger population for 5 objectives
    sampler = NSGAIIISampler(
        population_size=50,  # Increased for 5 objectives
        mutation_prob=0.15,  # Slightly reduced for stability
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
    
    # Start HPO phase tracking
    enhanced_co2_tracker.set_phase("hyperparameter_optimization")
    hpo_start_stats = enhanced_co2_tracker.get_current_stats()
    
    # Global carbon tracking
    global_tracker = CarbonGPUTracker("global_optimization")
    global_tracker.start_tracking()
    
    print(f"\n🚀 Starting {args.n_trials} trials of carbon-aware multi-objective optimization...")
    print(f"🎯 Objectives: Maximize Accuracy, Maximize F1, Minimize Time, Minimize Carbon, Minimize GPU Memory")
    
    study.optimize(lambda trial: optuna_objective(
        trial,
        dataset_class=dataset_class,
        seed=args.seed,
        top_k_candidates=top_k_candidates,
        carbon_budget_kg=args.carbon_budget,
        enable_progressive=args.enable_progressive,
        total_trials=args.n_trials,
        carbon_analyzer=carbon_analyzer,
        enhanced_co2_tracker=enhanced_co2_tracker,
        enhanced_visualizer=enhanced_visualizer
    ), n_trials=args.n_trials)

    # Track HPO completion
    hpo_end_stats = enhanced_co2_tracker.get_current_stats()
    carbon_analyzer.log_stage_emission("hyperparameter_optimization", (hpo_end_stats['total_co2_g'] - hpo_start_stats['total_co2_g']) / 1000)

    global_metrics = global_tracker.stop_tracking()

    pareto_trials = study.best_trials
    print(f"\n✅ Carbon-Aware Optimization completed!")
    print(f"🏆 Found {len(pareto_trials)} Pareto-optimal solutions:")
    for i, t in enumerate(pareto_trials[:5]):  # Show top 5
        efficiency = t.user_attrs.get('efficiency_weight', 1.0)
        actual_carbon = t.user_attrs.get('emissions_kg', t.values[3])
        carbon_efficiency = t.values[0] / (actual_carbon * 1000 + 1)
        print(f"  [{i+1}] Acc: {t.values[0]:.4f}, F1: {t.values[1]:.4f}, Time: {t.values[2]:.1f}s")
        print(f"      CO2: {actual_carbon:.4f}kg (adj: {t.values[3]:.4f}), GPU: {t.values[4]:.2f}GB")
        print(f"      Efficiency: {efficiency:.1f}, Carbon Eff: {carbon_efficiency:.4f} acc/g")

    # Calculate and display carbon savings
    if args.baseline_emissions > 0:
        total_actual_emissions = sum([t.user_attrs.get('emissions_kg', 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        savings, savings_pct = carbon_analyzer.calculate_carbon_savings(args.baseline_emissions)
        print(f"\n🌱 CARBON IMPACT ANALYSIS:")
        print(f"   Baseline Approach: {args.baseline_emissions:.4f} kg CO2")
        print(f"   Carbon-Aware Approach: {total_actual_emissions:.4f} kg CO2")
        print(f"   Carbon Savings: {savings:.4f} kg CO2 ({savings_pct:.1f}%)")

    # Enhanced solution analysis
    print(f"\n🎯 ENHANCED SOLUTION ANALYSIS:")
    print("="*60)
    
    # Categorize solutions by carbon efficiency
    carbon_efficient = [t for t in pareto_trials if t.values[3] < args.carbon_budget * 0.5]
    high_performance = [t for t in pareto_trials if t.values[0] > 0.85 and t.values[1] > 0.8]
    balanced = [t for t in pareto_trials if t not in carbon_efficient and t not in high_performance]
    
    print(f"🌱 Carbon Efficient ({len(carbon_efficient)}): Ultra-low carbon solutions")
    for t in carbon_efficient[:3]:
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, Carbon={t.values[3]:.4f}kg")
    
    print(f"🚀 High Performance ({len(high_performance)}): Best accuracy solutions")
    for t in high_performance[:3]:
        print(f"   Trial {t.number}: Acc={t.values[0]:.3f}, F1={t.values[1]:.3f}")
    
    print(f"⚖️ Balanced ({len(balanced)}): Optimal trade-offs")

    # Select best solution (configurable strategy)
    best_acc_trial = max(pareto_trials, key=lambda t: t.values[0])  
    best_params = best_acc_trial.params
    final_epochs = 10 if args.dataset == "flowers" else 8
    best_id = best_acc_trial.params['candidate_id']
    backbone, head = candidate_lookup[best_id]
    
    print(f"\n🎯 Selected Best Solution: Trial {best_acc_trial.number}")
    print(f"   Performance: Acc={best_acc_trial.values[0]:.4f}, F1={best_acc_trial.values[1]:.4f}")
    print(f"   Sustainability: Carbon={best_acc_trial.values[3]:.4f}kg, GPU={best_acc_trial.values[4]:.2f}GB")
    print(f"   Architecture: {backbone} (Efficiency: {best_acc_trial.user_attrs.get('efficiency_weight', 1.0):.1f})")
    
    # Final training with enhanced tracking
    enhanced_co2_tracker.set_phase("final_model_training")
    final_start_stats = enhanced_co2_tracker.get_current_stats()
    
    final_tracker = CarbonGPUTracker("final_training")
    final_tracker.start_tracking()
    
    print(f"\n🏅 Training final model with carbon-optimized configuration...")
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
    
    final_metrics = final_tracker.stop_tracking()
    
    # Track final training stage
    final_end_stats = enhanced_co2_tracker.get_current_stats()
    carbon_analyzer.log_stage_emission("final_model_training", (final_end_stats['total_co2_g'] - final_start_stats['total_co2_g']) / 1000)
    
    # Save predictions
    output_path = Path("final_test_preds.npy") if args.dataset == "skin_cancer" else args.output_path
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    print(f"\n✅ FINAL RESULTS:")
    print(f"📊 Predictions saved to: {output_path}")
    if not np.isnan(test_labels).any():
        final_acc = accuracy_score(test_labels, test_preds)
        final_f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"✅ Final Test Performance: Acc={final_acc:.4f}, F1={final_f1:.4f}")
    else:
        print(f"ℹ️ No test labels available for {dataset_class.__name__}")
    
    # Stop enhanced tracking and create comprehensive analysis
    enhanced_summary = enhanced_co2_tracker.stop_tracking()
    
    # Enhanced carbon summary with detailed breakdown
    total_emissions = global_metrics['emissions_kg'] + final_metrics['emissions_kg']
    efficiency_used = get_architecture_efficiency_weight(backbone)
    
    print(f"\n🌱 COMPREHENSIVE SUSTAINABILITY SUMMARY:")
    print("="*60)
    print(f"📊 Total Process Emissions: {enhanced_summary.get('total_co2_g', 0)/1000:.4f} kg CO2eq")
    print(f"   • CodeCarbon Tracked: {total_emissions:.4f} kg CO2eq")
    print(f"   • Enhanced Tracker: {enhanced_summary.get('total_co2_g', 0)/1000:.4f} kg CO2eq")
    
    print(f"\n🔍 Stage-by-Stage Breakdown:")
    for stage, emission in carbon_analyzer.stage_emissions.items():
        percentage = (emission / sum(carbon_analyzer.stage_emissions.values())) * 100
        print(f"   • {stage.replace('_', ' ').title()}: {emission:.4f} kg ({percentage:.1f}%)")
    
    print(f"\n⚡ Resource Utilization:")
    print(f"   • Peak GPU Memory: {max(global_metrics['peak_gpu_memory_gb'], final_metrics['peak_gpu_memory_gb']):.2f} GB")
    print(f"   • Total Training Time: {(global_metrics['training_time'] + final_metrics['training_time'])/60:.1f} minutes")
    print(f"   • Architecture Efficiency Used: {efficiency_used:.1f}/1.0")
    
    print(f"\n🎯 Carbon Efficiency Metrics:")
    if total_emissions > 0 and not np.isnan(test_labels).any():
        carbon_efficiency = final_acc / (total_emissions * 1000)  # Accuracy per gram CO2
        print(f"   • Carbon Efficiency: {carbon_efficiency:.2f} accuracy points per gram CO2")
        print(f"   • Environmental ROI: {(args.baseline_emissions - total_emissions)/args.baseline_emissions*100:.1f}% reduction vs baseline")
    
    # Create comprehensive carbon analysis plots
    if args.save_carbon_analysis:
        print(f"\n📊 Creating comprehensive carbon analysis...")
        carbon_analyzer.create_comprehensive_carbon_analysis(study, f"{args.dataset}_carbon_aware")
        
        # Save detailed carbon report
        report = carbon_analyzer.save_carbon_analysis_report(f"{args.dataset}_carbon_analysis_report.json")
        
        # Create enhanced visualizations
        enhanced_visualizer.create_comprehensive_dashboard(f"{args.dataset}_enhanced_dashboard.png")
        enhanced_visualizer.create_search_space_3d(f"{args.dataset}_enhanced_search_space.png")
        enhanced_co2_tracker.save_data(f"{args.dataset}_enhanced_co2_data.json")
        
        print(f"✅ Comprehensive carbon analysis saved!")
    
    print("✅ Enhanced Carbon-Aware AutoML completed successfully!")
    print(f"🌍 Environmental Impact: Equivalent to {enhanced_summary.get('car_equivalent_km', 0):.3f}km of car driving")
    
    # Save Optuna plots (unchanged)
    save_optuna_visualizations(study)
    save_accuracy_histogram(study)
    save_metric_curves(study)
    print("✅ All visualizations and analysis completed!")