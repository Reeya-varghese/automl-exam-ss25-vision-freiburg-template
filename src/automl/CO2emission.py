import torch
import time
import threading
import numpy as np
import warnings
import logging
import os
import sys
from contextlib import contextmanager
import contextlib


warnings.filterwarnings("ignore")
logging.getLogger("codecarbon").setLevel(logging.ERROR)
logging.getLogger("codecarbon").propagate = False

class CarbonGPUTracker:
  
    
    def __init__(self, project_name="automl_carbon_tracking"):
        self.project_name = project_name
        self.emissions_tracker = None
        self.gpu_available = torch.cuda.is_available()
        self.monitoring_active = False
        self.peak_gpu_memory = 0
        self.start_time = None
        
    def start_tracking(self, trial_id=None):
      
        self.start_time = time.time()
        self.peak_gpu_memory = 0
        self.monitoring_active = True

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
        
       
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
    
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


def get_architecture_efficiency_weight(backbone_name):
    """Efficiency weights based on parameter count and energy research"""
    efficiency_weights = {
        'resnet18': 1.0,        
        'efficientnet_b0': 0.9, 
        'vit_base_patch16_224': 0.8, 
    }
    return efficiency_weights.get(backbone_name, 0.5)

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
            'min_batch_size': 64,
            'prefer_efficient_arch': True
        }
    elif progress_ratio < 0.7:  # Middle 40%: balanced approach
        return {
            'max_epochs': 10,
            'min_batch_size': 32,
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
