"""
SMOTE Data Balancer Module for AutoML Vision Pipeline

This module provides SMOTE-based class balancing functionality that integrates
seamlessly with the existing AutoML pipeline without modifying core components.

Features:
- Memory-efficient SMOTE with configurable sample limits
- Automatic class imbalance detection
- Integration with existing dataset classes
- Preserves original data transforms and preprocessing
- Optional fallback to random oversampling for extreme memory constraints
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms
from collections import Counter
from typing import Any, Optional, Tuple, Dict
import warnings
from sklearn.utils import resample

# Try to import SMOTE, with fallback options
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

logger = logging.getLogger(__name__)

class SMOTEDatasetWrapper(Dataset):
    """
    A wrapper dataset that applies SMOTE balancing to the original dataset.
    Preserves original transforms and maintains compatibility with existing pipeline.
    """
    
    def __init__(
        self,
        original_dataset: Dataset,
        max_samples: int = 2000,
        smote_strategy: str = 'auto',
        random_state: int = 42,
        enable_smote: bool = True
    ):
        """
        Initialize SMOTE dataset wrapper.
        
        Args:
            original_dataset: The original dataset to balance
            max_samples: Maximum samples to use for SMOTE (memory constraint)
            smote_strategy: SMOTE sampling strategy ('auto', 'minority', 'not minority', 'all', or dict)
            random_state: Random seed for reproducibility
            enable_smote: Whether to apply SMOTE or use original dataset
        """
        self.original_dataset = original_dataset
        self.max_samples = max_samples
        self.smote_strategy = smote_strategy
        self.random_state = random_state
        self.enable_smote = enable_smote
        
        # Extract transform from original dataset if available
        self.transform = getattr(original_dataset, 'transform', None)
        self.target_transform = getattr(original_dataset, 'target_transform', None)
        
        # Apply SMOTE balancing
        if self.enable_smote and IMBALANCED_LEARN_AVAILABLE:
            self._apply_smote_balancing()
        else:
            logger.warning("SMOTE disabled or unavailable. Using original dataset.")
            self._use_original_data()
    
    def _extract_data_for_smote(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract data from dataset for SMOTE processing.
        Handles memory constraints by limiting sample size.
        """
        logger.info(f"Extracting data for SMOTE (max {self.max_samples} samples)...")
        
        # Determine actual sample size
        dataset_size = len(self.original_dataset)
        actual_samples = min(self.max_samples, dataset_size)
        
        if actual_samples < dataset_size:
            # Create random subset for SMOTE
            indices = np.random.choice(dataset_size, actual_samples, replace=False)
            subset_dataset = Subset(self.original_dataset, indices)
            logger.info(f"Using subset of {actual_samples} samples for SMOTE from {dataset_size} total")
        else:
            subset_dataset = self.original_dataset
        
        # Extract features and labels
        features = []
        labels = []
        
        # Temporarily remove transforms to get raw data
        original_transform = getattr(subset_dataset.dataset if hasattr(subset_dataset, 'dataset') else subset_dataset, 'transform', None)
        if hasattr(subset_dataset, 'dataset'):
            subset_dataset.dataset.transform = transforms.ToTensor()
        else:
            subset_dataset.transform = transforms.ToTensor()
        
        try:
            for i in range(len(subset_dataset)):
                try:
                    data, label = subset_dataset[i]
                    
                    # Convert tensor to numpy and flatten
                    if torch.is_tensor(data):
                        data = data.numpy()
                    
                    # Flatten image data
                    features.append(data.flatten())
                    labels.append(label)
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
        
        finally:
            # Restore original transform
            if hasattr(subset_dataset, 'dataset'):
                subset_dataset.dataset.transform = original_transform
            else:
                subset_dataset.transform = original_transform
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"Extracted {len(features)} samples with shape {features.shape}")
        return features, labels
    
    def _check_class_imbalance(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze class distribution and determine if balancing is needed.
        """
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate imbalance metrics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Class distribution percentages
        class_percentages = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Determine if significant imbalance exists (ratio > 2.0 is often considered imbalanced)
        is_imbalanced = imbalance_ratio > 2.0
        
        analysis = {
            'class_counts': class_counts,
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'class_percentages': class_percentages,
            'is_imbalanced': is_imbalanced,
            'num_classes': len(class_counts)
        }
        
        logger.info(f"Class imbalance analysis:")
        logger.info(f"  - Total samples: {total_samples}")
        logger.info(f"  - Number of classes: {len(class_counts)}")
        logger.info(f"  - Class distribution: {dict(class_counts)}")
        logger.info(f"  - Imbalance ratio: {imbalance_ratio:.2f}")
        logger.info(f"  - Requires balancing: {is_imbalanced}")
        
        return analysis
    
    def _apply_smote_balancing(self):
        """
        Apply SMOTE balancing to the dataset.
        """
        logger.info("Applying SMOTE balancing...")
        
        try:
            # Extract data for SMOTE
            features, labels = self._extract_data_for_smote()
            
            # Check if balancing is needed
            imbalance_analysis = self._check_class_imbalance(labels)
            
            if not imbalance_analysis['is_imbalanced']:
                logger.info("Dataset is already balanced. Skipping SMOTE.")
                self._use_original_data()
                return
            
            # Configure SMOTE based on dataset characteristics
            smote_kwargs = {
                'sampling_strategy': self.smote_strategy,
                'random_state': self.random_state,
                'k_neighbors': min(5, len(labels) // imbalance_analysis['num_classes'] - 1)
            }
            
            # Choose appropriate SMOTE variant
            if imbalance_analysis['num_classes'] > 10:
                # Use BorderlineSMOTE for high-dimensional problems
                smote = BorderlineSMOTE(**smote_kwargs)
                logger.info("Using BorderlineSMOTE for multi-class problem")
            else:
                # Use standard SMOTE
                smote = SMOTE(**smote_kwargs)
                logger.info("Using standard SMOTE")
            
            # Apply SMOTE
            logger.info("Generating synthetic samples...")
            features_resampled, labels_resampled = smote.fit_resample(features, labels)
            
            # Log results
            original_counts = Counter(labels)
            new_counts = Counter(labels_resampled)
            
            logger.info(f"SMOTE balancing completed:")
            logger.info(f"  - Original samples: {len(labels)} -> {len(labels_resampled)}")
            logger.info(f"  - Original distribution: {dict(original_counts)}")
            logger.info(f"  - New distribution: {dict(new_counts)}")
            
            # Store balanced data
            self.balanced_features = features_resampled
            self.balanced_labels = labels_resampled
            self._create_balanced_indices()
            
        except Exception as e:
            logger.error(f"SMOTE balancing failed: {e}")
            logger.info("Falling back to original dataset")
            self._use_original_data()
    
    def _create_balanced_indices(self):
        """
        Create mapping from balanced dataset indices to original samples.
        """
        # For synthetic samples, we'll need to reconstruct them
        # This is a simplified approach - in practice, you might want to store
        # the synthetic samples differently
        original_size = len(self.original_dataset)
        self.index_mapping = []
        
        for i in range(len(self.balanced_labels)):
            if i < original_size:
                # Original sample
                self.index_mapping.append(('original', i))
            else:
                # Synthetic sample - map to a random original sample of same class
                target_class = self.balanced_labels[i]
                # Find original samples of the same class
                original_class_indices = [
                    j for j in range(min(len(self.balanced_labels), original_size))
                    if self.balanced_labels[j] == target_class
                ]
                if original_class_indices:
                    base_idx = np.random.choice(original_class_indices)
                    self.index_mapping.append(('synthetic', base_idx, self.balanced_features[i]))
                else:
                    # Fallback to first sample
                    self.index_mapping.append(('synthetic', 0, self.balanced_features[i]))
    
    def _use_original_data(self):
        """
        Use original dataset without SMOTE balancing.
        """
        self.balanced_features = None
        self.balanced_labels = None
        self.index_mapping = [('original', i) for i in range(len(self.original_dataset))]
        logger.info("Using original dataset without SMOTE balancing")
    
    def __len__(self):
        """Return length of balanced dataset."""
        if self.balanced_labels is not None:
            return len(self.balanced_labels)
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from balanced dataset.
        """
        if self.balanced_labels is None:
            # No SMOTE applied, use original dataset
            return self.original_dataset[idx]
        
        # Handle balanced dataset
        mapping = self.index_mapping[idx]
        
        if mapping[0] == 'original':
            # Original sample
            return self.original_dataset[mapping[1]]
        else:
            # Synthetic sample
            _, base_idx, synthetic_features = mapping
            
            # Get base sample for transform reference
            base_data, _ = self.original_dataset[base_idx]
            
            # Reconstruct synthetic image
            if torch.is_tensor(base_data):
                original_shape = base_data.shape
            else:
                # Assume PIL Image, get shape after tensor conversion
                tensor_transform = transforms.ToTensor()
                temp_tensor = tensor_transform(base_data)
                original_shape = temp_tensor.shape
            
            # Reshape synthetic features back to image shape
            synthetic_image = synthetic_features.reshape(original_shape)
            synthetic_tensor = torch.from_numpy(synthetic_image).float()
            
            # Apply transforms if available
            if self.transform:
                # Convert back to PIL if needed for transforms
                to_pil = transforms.ToPILImage()
                synthetic_pil = to_pil(synthetic_tensor)
                synthetic_data = self.transform(synthetic_pil)
            else:
                synthetic_data = synthetic_tensor
            
            synthetic_label = self.balanced_labels[idx]
            
            if self.target_transform:
                synthetic_label = self.target_transform(synthetic_label)
            
            return synthetic_data, synthetic_label


class SMOTEBalancer:
    """
    Main class for handling SMOTE balancing in the AutoML pipeline.
    """
    
    @staticmethod
    def create_balanced_dataset(
        dataset_class: Any,
        max_samples: int = 2000,
        smote_strategy: str = 'auto',
        random_state: int = 42,
        enable_smote: bool = True,
        **dataset_kwargs
    ) -> SMOTEDatasetWrapper:
        """
        Create a SMOTE-balanced version of the dataset.
        
        Args:
            dataset_class: Dataset class (e.g., FashionDataset, FlowersDataset)
            max_samples: Maximum samples for SMOTE processing
            smote_strategy: SMOTE sampling strategy
            random_state: Random seed
            enable_smote: Whether to apply SMOTE
            **dataset_kwargs: Additional arguments for dataset creation
        
        Returns:
            SMOTEDatasetWrapper: Balanced dataset
        """
        # Create original dataset
        original_dataset = dataset_class(**dataset_kwargs)
        
        # Wrap with SMOTE balancer
        balanced_dataset = SMOTEDatasetWrapper(
            original_dataset=original_dataset,
            max_samples=max_samples,
            smote_strategy=smote_strategy,
            random_state=random_state,
            enable_smote=enable_smote
        )
        
        return balanced_dataset
    
    @staticmethod
    def analyze_dataset_balance(dataset) -> Dict[str, Any]:
        """
        Analyze class balance of any dataset.
        
        Args:
            dataset: Dataset to analyze
        
        Returns:
            Dictionary with balance analysis
        """
        labels = []
        for i in range(min(1000, len(dataset))):  # Sample for analysis
            _, label = dataset[i]
            labels.append(label)
        
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            'class_counts': dict(class_counts),
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 2.0,
            'num_classes': len(class_counts)
        }


# Utility functions for integration with existing code
def check_imbalanced_learn_availability():
    """Check if imbalanced-learn is available."""
    return IMBALANCED_LEARN_AVAILABLE

def install_imbalanced_learn():
    """Print installation instructions for imbalanced-learn."""
    print("To use SMOTE balancing, install imbalanced-learn:")
    print("pip install imbalanced-learn")
    print("or")
    print("conda install -c conda-forge imbalanced-learn")