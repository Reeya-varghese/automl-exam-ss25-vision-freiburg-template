# co2_tracker.py
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

class CO2Tracker:
    """
    Real-time CO2 emission tracker for ML training processes.
    Estimates CO2 emissions based on power consumption and carbon intensity.
    """
    
    def __init__(self, region: str = "DE", tracking_interval: float = 1.0):
        """
        Initialize CO2 tracker.
        
        Args:
            region: Region code for carbon intensity (DE=Germany, US=USA, etc.)
            tracking_interval: How often to sample power consumption (seconds)
        """
        self.region = region
        self.tracking_interval = tracking_interval
        self.is_tracking = False
        self.tracking_thread = None
        
        # Carbon intensity factors (gCO2/kWh) by region
        self.carbon_intensity = {
            "DE": 401,  # Germany
            "US": 400,  # USA average
            "FR": 57,   # France (nuclear heavy)
            "CN": 681,  # China
            "GB": 233,  # UK
            "NO": 17,   # Norway (hydro heavy)
            "default": 400
        }
        
        # Initialize tracking data
        self.reset()
        
    def reset(self):
        """Reset all tracking data."""
        self.start_time = None
        self.end_time = None
        self.power_samples = []
        self.timestamps = []
        self.co2_samples = []
        self.cumulative_co2 = []
        self.process_phases = []  # Track different phases (training, validation, etc.)
        self.current_phase = "idle"
        
    def get_power_consumption(self) -> float:
        """
        Estimate power consumption based on CPU and GPU usage.
        Returns power in watts.
        """
        try:
            # CPU power estimation
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_power = (cpu_percent / 100) * 45  # Assume 45W max CPU power
            
            # GPU power estimation (simplified)
            gpu_power = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_power += power_mw / 1000  # Convert to watts
            except:
                # Fallback: estimate based on CUDA availability
                import torch
                if torch.cuda.is_available():
                    gpu_power = 150  # Assume 150W GPU usage during training
                    
            # Memory and other components (rough estimate)
            memory_power = 10  # RAM and other components
            
            total_power = cpu_power + gpu_power + memory_power
            return max(total_power, 20)  # Minimum idle power
            
        except Exception as e:
            print(f"Error measuring power: {e}")
            return 100  # Fallback estimate
    
    def _tracking_loop(self):
        """Main tracking loop that runs in separate thread."""
        while self.is_tracking:
            timestamp = time.time()
            power_w = self.get_power_consumption()
            
            # Calculate instantaneous CO2 (gCO2)
            carbon_factor = self.carbon_intensity.get(self.region, self.carbon_intensity["default"])
            # Power in kW * time interval in hours * carbon intensity
            co2_g = (power_w / 1000) * (self.tracking_interval / 3600) * carbon_factor
            
            self.timestamps.append(timestamp)
            self.power_samples.append(power_w)
            self.co2_samples.append(co2_g)
            
            # Calculate cumulative CO2
            total_co2 = sum(self.co2_samples)
            self.cumulative_co2.append(total_co2)
            
            time.sleep(self.tracking_interval)
    
    def start_tracking(self, phase: str = "training"):
        """Start CO2 tracking."""
        if self.is_tracking:
            return
            
        self.current_phase = phase
        self.start_time = time.time()
        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print(f"🌱 CO2 tracking started for phase: {phase}")
    
    def set_phase(self, phase: str):
        """Change current tracking phase."""
        self.current_phase = phase
        if self.timestamps:
            self.process_phases.append({
                'timestamp': self.timestamps[-1],
                'phase': phase
            })
    
    def stop_tracking(self):
        """Stop CO2 tracking."""
        if not self.is_tracking:
            return
            
        self.is_tracking = False
        self.end_time = time.time()
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2)
        
        total_co2 = sum(self.co2_samples) if self.co2_samples else 0
        duration = (self.end_time - self.start_time) / 60  # minutes
        print(f"🌱 CO2 tracking stopped. Total CO2: {total_co2:.2f}g over {duration:.1f} minutes")
        
        return self.get_summary()
    
    def get_current_stats(self) -> Dict:
        """Get current tracking statistics."""
        if not self.timestamps:
            return {"total_co2_g": 0, "avg_power_w": 0, "duration_min": 0}
            
        total_co2 = sum(self.co2_samples)
        avg_power = np.mean(self.power_samples)
        duration = (time.time() - self.start_time) / 60
        
        return {
            "total_co2_g": total_co2,
            "avg_power_w": avg_power,
            "duration_min": duration,
            "current_phase": self.current_phase
        }
    
    def get_summary(self) -> Dict:
        """Get complete tracking summary."""
        if not self.timestamps:
            return {}
            
        total_co2_g = sum(self.co2_samples)
        total_co2_kg = total_co2_g / 1000
        avg_power = np.mean(self.power_samples)
        max_power = max(self.power_samples)
        duration_hours = (self.end_time - self.start_time) / 3600
        
        # CO2 equivalents for context
        car_km = total_co2_g / 120  # Assume 120g CO2/km for average car
        tree_offset = total_co2_g / 21000  # Assume 21kg CO2/year per tree
        
        return {
            "total_co2_g": total_co2_g,
            "total_co2_kg": total_co2_kg,
            "avg_power_w": avg_power,
            "max_power_w": max_power,
            "duration_hours": duration_hours,
            "carbon_intensity": self.carbon_intensity.get(self.region, 400),
            "region": self.region,
            "car_equivalent_km": car_km,
            "tree_offset_years": tree_offset,
            "samples_count": len(self.timestamps)
        }
    
    def save_data(self, filepath: str):
        """Save tracking data to JSON file."""
        data = {
            "summary": self.get_summary(),
            "timestamps": self.timestamps,
            "power_samples": self.power_samples,
            "co2_samples": self.co2_samples,
            "cumulative_co2": self.cumulative_co2,
            "process_phases": self.process_phases
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 CO2 tracking data saved to {filepath}")


class EnhancedVisualizer:
    """
    Enhanced visualization system for AutoML with CO2 tracking.
    """
    
    def __init__(self, co2_tracker: CO2Tracker):
        self.co2_tracker = co2_tracker
        self.training_history = {}
        self.search_space_data = []
        self.trial_data = []
        
    def log_trial(self, trial_info: Dict):
        """Log trial information for search space visualization."""
        self.trial_data.append({
            'trial_id': trial_info.get('trial_id', len(self.trial_data)),
            'params': trial_info.get('params', {}),
            'values': trial_info.get('values', []),
            'timestamp': time.time(),
            'co2_at_trial': self.co2_tracker.get_current_stats().get('total_co2_g', 0)
        })
    
    def log_training_step(self, epoch: int, metrics: Dict, phase: str = "train"):
        """Log training step for detailed visualization."""
        if phase not in self.training_history:
            self.training_history[phase] = {
                'epochs': [], 'loss': [], 'accuracy': [], 'co2': [], 'power': []
            }
        
        co2_stats = self.co2_tracker.get_current_stats()
        self.training_history[phase]['epochs'].append(epoch)
        self.training_history[phase]['loss'].append(metrics.get('loss', 0))
        self.training_history[phase]['accuracy'].append(metrics.get('accuracy', 0))
        self.training_history[phase]['co2'].append(co2_stats.get('total_co2_g', 0))
        self.training_history[phase]['power'].append(co2_stats.get('avg_power_w', 0))
    
    def create_comprehensive_dashboard(self, save_path: str = "automl_dashboard.png"):
        """Create comprehensive visualization dashboard."""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. CO2 Emissions Over Time
        plt.subplot(3, 4, 1)
        if self.co2_tracker.timestamps:
            times = [(t - self.co2_tracker.start_time) / 60 for t in self.co2_tracker.timestamps]
            plt.plot(times, self.co2_tracker.cumulative_co2, color='green', linewidth=2)
            plt.fill_between(times, self.co2_tracker.cumulative_co2, alpha=0.3, color='green')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Cumulative CO2 (g)')
            plt.title('🌱 CO2 Emissions Over Time', fontweight='bold')
            plt.grid(alpha=0.3)
            
            # Add phase markers
            for phase_info in self.co2_tracker.process_phases:
                phase_time = (phase_info['timestamp'] - self.co2_tracker.start_time) / 60
                plt.axvline(x=phase_time, color='red', linestyle='--', alpha=0.7)
                plt.text(phase_time, max(self.co2_tracker.cumulative_co2) * 0.8, 
                        phase_info['phase'], rotation=90, fontsize=8)
        
        # 2. Power Consumption Pattern
        plt.subplot(3, 4, 2)
        if self.co2_tracker.timestamps:
            times = [(t - self.co2_tracker.start_time) / 60 for t in self.co2_tracker.timestamps]
            plt.plot(times, self.co2_tracker.power_samples, color='orange', linewidth=1.5)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Power (W)')
            plt.title('⚡ Power Consumption Pattern', fontweight='bold')
            plt.grid(alpha=0.3)
            
            # Add moving average
            if len(self.co2_tracker.power_samples) > 10:
                window = min(10, len(self.co2_tracker.power_samples) // 5)
                moving_avg = pd.Series(self.co2_tracker.power_samples).rolling(window=window).mean()
                plt.plot(times, moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
                plt.legend()
        
        # 3. Training Progress with CO2
        plt.subplot(3, 4, 3)
        if 'train' in self.training_history:
            ax1 = plt.gca()
            epochs = self.training_history['train']['epochs']
            accuracy = self.training_history['train']['accuracy']
            
            ax1.plot(epochs, accuracy, color='blue', marker='o', label='Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            co2_vals = self.training_history['train']['co2']
            ax2.plot(epochs, co2_vals, color='green', marker='s', label='CO2')
            ax2.set_ylabel('Cumulative CO2 (g)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            plt.title('📈 Training Progress vs CO2', fontweight='bold')
        
        # 4. Search Space Exploration
        plt.subplot(3, 4, 4)
        if self.trial_data and len(self.trial_data) > 1:
            trials_df = pd.DataFrame(self.trial_data)
            if 'values' in trials_df.columns and trials_df['values'].iloc[0]:
                accuracies = [trial['values'][0] if trial['values'] else 0 for trial in self.trial_data]
                co2_costs = [trial['co2_at_trial'] for trial in self.trial_data]
                
                scatter = plt.scatter(co2_costs, accuracies, c=range(len(accuracies)), 
                                    cmap='viridis', s=60, alpha=0.7)
                plt.xlabel('CO2 Cost (g)')
                plt.ylabel('Validation Accuracy')
                plt.title('🔍 Search Space: Accuracy vs CO2', fontweight='bold')
                plt.colorbar(scatter, label='Trial Order')
        
        # 5. Architecture Selection Heatmap
        plt.subplot(3, 4, 5)
        if self.trial_data:
            # Extract backbone usage
            backbone_usage = {}
            for trial in self.trial_data:
                backbone = trial['params'].get('candidate_id', 'unknown')
                backbone = backbone.split('_')[0] if '_' in backbone else backbone
                backbone_usage[backbone] = backbone_usage.get(backbone, 0) + 1
            
            if backbone_usage:
                backbones = list(backbone_usage.keys())
                counts = list(backbone_usage.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(backbones)))
                
                plt.pie(counts, labels=backbones, autopct='%1.1f%%', colors=colors)
                plt.title('🏗️ Architecture Selection\nFrequency', fontweight='bold')
        
        # 6. CO2 Efficiency Analysis
        plt.subplot(3, 4, 6)
        if self.trial_data and len(self.trial_data) > 3:
            accuracies = [trial['values'][0] if trial['values'] else 0 for trial in self.trial_data]
            co2_costs = [trial['co2_at_trial'] for trial in self.trial_data]
            
            # Calculate efficiency (accuracy per gram CO2)
            efficiency = [acc / (co2 + 0.1) for acc, co2 in zip(accuracies, co2_costs)]
            trial_ids = range(len(efficiency))
            
            bars = plt.bar(trial_ids, efficiency, color='lightgreen', alpha=0.7)
            plt.xlabel('Trial ID')
            plt.ylabel('Accuracy / CO2 (g⁻¹)')
            plt.title('♻️ CO2 Efficiency by Trial', fontweight='bold')
            
            # Highlight best efficiency
            best_idx = np.argmax(efficiency)
            bars[best_idx].set_color('darkgreen')
            plt.text(best_idx, efficiency[best_idx] + 0.001, 'Best', 
                    ha='center', fontweight='bold')
        
        # 7. Hyperparameter Impact on CO2
        plt.subplot(3, 4, 7)
        if self.trial_data:
            # Analyze batch size impact
            batch_sizes = []
            co2_per_trial = []
            
            for trial in self.trial_data:
                if 'batch_size' in trial['params']:
                    batch_sizes.append(trial['params']['batch_size'])
                    co2_per_trial.append(trial['co2_at_trial'])
            
            if batch_sizes and len(set(batch_sizes)) > 1:
                df = pd.DataFrame({'batch_size': batch_sizes, 'co2': co2_per_trial})
                
                for bs in set(batch_sizes):
                    subset = df[df['batch_size'] == bs]['co2']
                    plt.scatter([bs] * len(subset), subset, alpha=0.6, s=50, label=f'Batch {bs}')
                
                plt.xlabel('Batch Size')
                plt.ylabel('CO2 Emissions (g)')
                plt.title('⚙️ Batch Size vs CO2', fontweight='bold')
                plt.legend()
        
        # 8. Carbon Footprint Comparison
        plt.subplot(3, 4, 8)
        summary = self.co2_tracker.get_summary()
        if summary:
            # Create comparison with everyday activities
            comparisons = {
                'This Training': summary.get('total_co2_g', 0),
                'Car (1km)': 120,
                'Email': 4,
                'Google Search': 0.2,
                'Smartphone (1h)': 12
            }
            
            items = list(comparisons.keys())
            values = list(comparisons.values())
            colors = ['red' if item == 'This Training' else 'lightblue' for item in items]
            
            bars = plt.bar(items, values, color=colors, alpha=0.7)
            plt.ylabel('CO2 Emissions (g)')
            plt.title('🌍 Carbon Footprint Context', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            # Highlight our training
            bars[0].set_edgecolor('darkred')
            bars[0].set_linewidth(2)
        
        # 9. Real-time Metrics Dashboard
        plt.subplot(3, 4, 9)
        plt.axis('off')
        current_stats = self.co2_tracker.get_current_stats()
        summary = self.co2_tracker.get_summary()
        
        metrics_text = f"""
🔋 REAL-TIME METRICS
━━━━━━━━━━━━━━━━━━━━━━━
🌱 Total CO2: {current_stats.get('total_co2_g', 0):.1f}g
⚡ Avg Power: {current_stats.get('avg_power_w', 0):.1f}W
⏱️ Duration: {current_stats.get('duration_min', 0):.1f}min
🎯 Phase: {current_stats.get('current_phase', 'N/A')}

🌍 ENVIRONMENTAL IMPACT
━━━━━━━━━━━━━━━━━━━━━━━
🚗 Car equivalent: {summary.get('car_equivalent_km', 0):.2f}km
🌳 Tree offset: {summary.get('tree_offset_years', 0):.3f} tree-years
🏭 Carbon intensity: {summary.get('carbon_intensity', 0)}g/kWh
📍 Region: {summary.get('region', 'N/A')}
        """
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 10. Optimization Progress
        plt.subplot(3, 4, 10)
        if self.trial_data and len(self.trial_data) > 1:
            trial_numbers = range(len(self.trial_data))
            best_so_far = []
            current_best = 0
            
            for trial in self.trial_data:
                acc = trial['values'][0] if trial['values'] else 0
                current_best = max(current_best, acc)
                best_so_far.append(current_best)
            
            plt.plot(trial_numbers, best_so_far, marker='o', linewidth=2, 
                    color='purple', markersize=4)
            plt.fill_between(trial_numbers, best_so_far, alpha=0.3, color='purple')
            plt.xlabel('Trial Number')
            plt.ylabel('Best Accuracy So Far')
            plt.title('📊 Optimization Progress', fontweight='bold')
            plt.grid(alpha=0.3)
        
        # 11. Energy Efficiency Trends
        plt.subplot(3, 4, 11)
        if len(self.trial_data) > 5:
            # Calculate rolling efficiency
            window_size = min(5, len(self.trial_data) // 3)
            efficiencies = [trial['values'][0] / (trial['co2_at_trial'] + 0.1) 
                          if trial['values'] else 0 for trial in self.trial_data]
            
            rolling_eff = pd.Series(efficiencies).rolling(window=window_size).mean()
            
            plt.plot(range(len(rolling_eff)), rolling_eff, color='green', linewidth=2)
            plt.xlabel('Trial Number')
            plt.ylabel('Rolling Efficiency')
            plt.title(f'🔄 Energy Efficiency Trend\n(Window: {window_size})', fontweight='bold')
            plt.grid(alpha=0.3)
        
        # 12. Final Summary Box
        plt.subplot(3, 4, 12)
        plt.axis('off')
        
        if self.trial_data and summary:
            best_trial = max(self.trial_data, key=lambda x: x['values'][0] if x['values'] else 0)
            total_trials = len(self.trial_data)
            
            summary_text = f"""
📋 AUTOML SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━
🏆 Best Accuracy: {best_trial['values'][0]:.4f}
🔢 Total Trials: {total_trials}
🌱 Total CO2: {summary.get('total_co2_g', 0):.1f}g
⚡ Total Energy: {summary.get('avg_power_w', 0) * summary.get('duration_hours', 0) / 1000:.2f}kWh
⏱️ Duration: {summary.get('duration_hours', 0):.1f}h

🎯 EFFICIENCY METRICS
━━━━━━━━━━━━━━━━━━━━━━━
💚 Best Trial CO2: {best_trial.get('co2_at_trial', 0):.1f}g
🔋 Avg Power: {summary.get('avg_power_w', 0):.1f}W
📈 Trials/Hour: {total_trials / max(summary.get('duration_hours', 1), 0.1):.1f}
            """
            
            plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Comprehensive dashboard saved to {save_path}")
    
    def create_search_space_3d(self, save_path: str = "search_space_3d.png"):
        """Create 3D visualization of search space exploration."""
        if len(self.trial_data) < 10:
            print("Not enough trials for 3D visualization")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # Extract parameters for 3D plot
        learning_rates = []
        accuracies = []
        co2_costs = []
        batch_sizes = []
        
        for trial in self.trial_data:
            if 'lr' in trial['params'] and trial['values']:
                learning_rates.append(np.log10(trial['params']['lr']))
                accuracies.append(trial['values'][0])
                co2_costs.append(trial['co2_at_trial'])
                batch_sizes.append(trial['params'].get('batch_size', 32))
        
        if len(learning_rates) > 5:
            ax = fig.add_subplot(121, projection='3d')
            
            scatter = ax.scatter(learning_rates, co2_costs, accuracies, 
                               c=batch_sizes, cmap='viridis', s=60, alpha=0.7)
            
            ax.set_xlabel('Log Learning Rate')
            ax.set_ylabel('CO2 Cost (g)')
            ax.set_zlabel('Accuracy')
            ax.set_title('🔍 3D Search Space Exploration', fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
            cbar.set_label('Batch Size')
            
            # Add Pareto front projection
            ax2 = fig.add_subplot(122)
            ax2.scatter(co2_costs, accuracies, c=batch_sizes, cmap='viridis', s=60, alpha=0.7)
            ax2.set_xlabel('CO2 Cost (g)')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('🎯 Pareto Front: Accuracy vs CO2', fontweight='bold')
            
            # Draw Pareto front
            points = list(zip(co2_costs, accuracies))
            points.sort()
            pareto_front = []
            for point in points:
                if not pareto_front or point[1] > pareto_front[-1][1]:
                    pareto_front.append(point)
            
            if len(pareto_front) > 1:
                pareto_x, pareto_y = zip(*pareto_front)
                ax2.plot(pareto_x, pareto_y, 'r-', linewidth=2, alpha=0.8, label='Pareto Front')
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"🔍 3D search space visualization saved to {save_path}")


# Integration helpers for existing code
def create_tracked_automl(original_automl_class):
    """
    Factory function to create CO2-tracked version of AutoML class.
    """
    class TrackedAutoML(original_automl_class):
        def __init__(self, *args, co2_tracker=None, visualizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.co2_tracker = co2_tracker
            self.visualizer = visualizer
            
        def fit(self, *args, **kwargs):
            if self.co2_tracker:
                self.co2_tracker.set_phase("training")
            
            # Call original fit method
            result = super().fit(*args, **kwargs)
            
            # Log training history if visualizer is available
            if self.visualizer and hasattr(self, '_history'):
                for epoch, (loss, acc) in enumerate(zip(
                    self._history.get('loss', []), 
                    self._history.get('acc', [])
                )):
                    self.visualizer.log_training_step(
                        epoch, {'loss': loss, 'accuracy': acc}, 'train'
                    )
                    
                for epoch, acc in enumerate(self._history.get('val_acc', [])):
                    self.visualizer.log_training_step(
                        epoch, {'accuracy': acc}, 'validation'
                    )
            
            return result
    
    return TrackedAutoML


# Usage example integration
def integrate_co2_tracking():
    """
    Example of how to integrate CO2 tracking into existing AutoML pipeline.
    Add this to your main execution file.
    """
    
    # Initialize tracking
    co2_tracker = CO2Tracker(region="DE", tracking_interval=0.5)
    visualizer = EnhancedVisualizer(co2_tracker)
    
    # Start tracking
    co2_tracker.start_tracking("automl_optimization")
    
    # Your existing AutoML code here...
    # Just add logging calls where appropriate:
    
    # Example for trial logging (add to your optuna objective):
    def enhanced_optuna_objective(trial, dataset_class, seed=42, top_k_candidates=None):
        # Your existing objective code...
        
        # Log trial for visualization
        visualizer.log_trial({
            'trial_id': trial.number,
            'params': trial.params,
            'values': [accuracy, f1, training_time]  # your results
        })
        
        return accuracy, f1, training_time
    
    # At the end of your pipeline:
    co2_tracker.stop_tracking()
    visualizer.create_comprehensive_dashboard("final_dashboard.png")
    visualizer.create_search_space_3d("search_space_3d.png")
    co2_tracker.save_data("co2_tracking_data.json")
    
    return co2_tracker, visualizer


if __name__ == "__main__":
    # Demo usage
    tracker = CO2Tracker()
    visualizer = EnhancedVisualizer(tracker)
    
    # Simulate some tracking data
    tracker.start_tracking("demo")
    
    # Simulate training process
    import time
    for i in range(10):
        time.sleep(0.1)
        visualizer.log_trial({
            'trial_id': i,
            'params': {'lr': 0.001 * (i+1), 'batch_size': 32},
            'values': [0.8 + i*0.01, 0.75 + i*0.01, 10.0]
        })
    
    tracker.stop_tracking()
    visualizer.create_comprehensive_dashboard()
    print("✅ Demo completed!")