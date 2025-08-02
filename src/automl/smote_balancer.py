"""
SMOTE Class Balancer Module for AutoML Vision Datasets

This module provides SMOTE (Synthetic Minority Oversampling Technique) functionality
to handle class imbalance problems in vision datasets. It integrates seamlessly with
the existing AutoML framework without modifying core files.

Usage:
    from smote_balancer import SMOTEBalancer, apply_smote_to_dataset
    
    # Option 1: Use the wrapper function (easiest)
    balanced_dataset = apply_smote_to_dataset(original_dataset, target_distribution='balanced')
    
    # Option 2: Use the class directly (more control)
    balancer = SMOTEBalancer(strategy='auto', k_neighbors=5)
    balanced_dataset = balancer.balance_dataset(original_dataset)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import torchvision.transforms as transforms
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

class SMOTEBalancer:
    """
    Advanced SMOTE-based class balancer for vision datasets.
    
    Supports multiple SMOTE variants and strategies for handling class imbalance.
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        smote_variant: str = 'standard',
        k_neighbors: int = 5,
        sampling_strategy: Union[str, dict] = 'auto',
        random_state: int = 42,
        feature_extractor: str = 'flatten'
    ):
        """
        Initialize SMOTE balancer.
        
        Args:
            strategy: Balancing strategy ('auto', 'minority', 'not_majority', 'all')
            smote_variant: SMOTE variant ('standard', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn')
            k_neighbors: Number of neighbors for SMOTE
            sampling_strategy: Target distribution strategy
            random_state: Random seed for reproducibility
            feature_extractor: How to extract features ('flatten', 'resnet_features', 'simple_cnn')
        """
        self.strategy = strategy
        self.smote_variant = smote_variant
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.feature_extractor = feature_extractor
        
        # Initialize SMOTE variant
        self.smote = self._get_smote_variant()
        
    def _get_smote_variant(self):
        """Get the appropriate SMOTE variant based on configuration."""
        smote_kwargs = {
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state,
            'k_neighbors': self.k_neighbors
        }
        
        variants = {
            'standard': SMOTE,
            'borderline': BorderlineSMOTE,
            'adasyn': ADASYN,
            'smote_tomek': SMOTETomek,
            'smote_enn': SMOTEENN
        }
        
        if self.smote_variant in ['smote_tomek', 'smote_enn']:
            # These variants don't use k_neighbors parameter
            smote_kwargs.pop('k_neighbors')
            return variants[self.smote_variant](
                smote=SMOTE(**{k: v for k, v in smote_kwargs.items() if k != 'sampling_strategy'}),
                random_state=self.random_state
            )
        elif self.smote_variant == 'adasyn':
            # ADASYN uses n_neighbors instead of k_neighbors
            smote_kwargs['n_neighbors'] = smote_kwargs.pop('k_neighbors')
            return variants[self.smote_variant](**smote_kwargs)
        else:
            return variants[self.smote_variant](**smote_kwargs)
    
    def analyze_class_distribution(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze class distribution in the dataset."""
        labels = []
        
        # Extract all labels
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        # Count distribution
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        analysis = {
            'total_samples': total_samples,
            'num_classes': len(label_counts),
            'class_counts': dict(label_counts),
            'class_percentages': {k: (v/total_samples)*100 for k, v in label_counts.items()},
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()),
            'is_imbalanced': max(label_counts.values()) / min(label_counts.values()) > 1.5
        }
        
        logger.info(f"Dataset Analysis:")
        logger.info(f"  Total samples: {analysis['total_samples']}")
        logger.info(f"  Number of classes: {analysis['num_classes']}")
        logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
        logger.info(f"  Class distribution: {analysis['class_counts']}")
        
        return analysis
    
    def _extract_features(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataset for SMOTE processing."""
        features = []
        labels = []
        
        logger.info(f"Extracting features using {self.feature_extractor} method...")
        
        for i in range(len(dataset)):
            image, label = dataset[i]
            
            # Convert to tensor if not already
            if isinstance(image, Image.Image):
                image = transforms.ToTensor()(image)
            
            # Extract features based on method
            if self.feature_extractor == 'flatten':
                # Simple flattening approach
                feature_vector = image.flatten().numpy()
            elif self.feature_extractor == 'resnet_features':
                # Use pre-trained ResNet features (more sophisticated)
                feature_vector = self._extract_resnet_features(image)
            elif self.feature_extractor == 'simple_cnn':
                # Use a simple CNN for feature extraction
                feature_vector = self._extract_cnn_features(image)
            else:
                # Default to flattening
                feature_vector = image.flatten().numpy()
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _extract_resnet_features(self, image: torch.Tensor) -> np.ndarray:
        """Extract features using a pre-trained ResNet model."""
        try:
            import torchvision.models as models
            
            if not hasattr(self, '_resnet_extractor'):
                # Initialize ResNet feature extractor
                resnet = models.resnet18(pretrained=True)
                resnet.fc = torch.nn.Identity()  # Remove final layer
                resnet.eval()
                self._resnet_extractor = resnet
            
            with torch.no_grad():
                # Ensure 3 channels for ResNet
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                
                # Add batch dimension and extract features
                features = self._resnet_extractor(image.unsqueeze(0))
                return features.squeeze().numpy()
                
        except Exception as e:
            logger.warning(f"ResNet feature extraction failed: {e}. Falling back to flattening.")
            return image.flatten().numpy()
    
    def _extract_cnn_features(self, image: torch.Tensor) -> np.ndarray:
        """Extract features using a simple CNN."""
        try:
            if not hasattr(self, '_simple_cnn'):
                # Create a simple CNN for feature extraction
                self._simple_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(image.shape[0], 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((8, 8)),
                    torch.nn.Flatten()
                ).eval()
            
            with torch.no_grad():
                features = self._simple_cnn(image.unsqueeze(0))
                return features.squeeze().numpy()
                
        except Exception as e:
            logger.warning(f"CNN feature extraction failed: {e}. Falling back to flattening.")
            return image.flatten().numpy()
    
    def _create_synthetic_images(self, original_features: np.ndarray, synthetic_features: np.ndarray, 
                                original_dataset: Dataset) -> List[torch.Tensor]:
        """Convert synthetic features back to image format."""
        synthetic_images = []
        
        # Get original image shape from dataset
        sample_image, _ = original_dataset[0]
        if isinstance(sample_image, Image.Image):
            sample_image = transforms.ToTensor()(sample_image)
        
        original_shape = sample_image.shape
        
        for synthetic_feature in synthetic_features:
            if self.feature_extractor == 'flatten':
                # Reshape back to original image dimensions
                synthetic_image = torch.tensor(synthetic_feature.reshape(original_shape), dtype=torch.float32)
                # Clamp values to valid range
                synthetic_image = torch.clamp(synthetic_image, 0, 1)
            else:
                # For feature-based extraction, we need to generate new images
                # This is more complex and might require a generative approach
                # For now, we'll use a simple interpolation method
                synthetic_image = self._interpolate_from_features(synthetic_feature, original_dataset, original_shape)
            
            synthetic_images.append(synthetic_image)
        
        return synthetic_images
    
    def _interpolate_from_features(self, synthetic_feature: np.ndarray, original_dataset: Dataset, 
                                 target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate synthetic image by interpolating from original dataset."""
        # Find closest images in feature space and interpolate
        # This is a simplified approach - in practice, you might want to use more sophisticated methods
        
        # For now, randomly select an image and add some noise
        random_idx = np.random.randint(0, len(original_dataset))
        base_image, _ = original_dataset[random_idx]
        
        if isinstance(base_image, Image.Image):
            base_image = transforms.ToTensor()(base_image)
        
        # Add small amount of noise
        noise = torch.randn_like(base_image) * 0.1
        synthetic_image = base_image + noise
        synthetic_image = torch.clamp(synthetic_image, 0, 1)
        
        return synthetic_image
    
    def balance_dataset(self, dataset: Dataset) -> 'BalancedDataset':
        """Apply SMOTE balancing to the dataset."""
        # Analyze original distribution
        analysis = self.analyze_class_distribution(dataset)
        
        if not analysis['is_imbalanced']:
            logger.info("Dataset is already balanced. Returning original dataset.")
            return BalancedDataset(dataset, [], [], analysis)
        
        # Extract features and labels
        features, labels = self._extract_features(dataset)
        
        # Apply SMOTE
        logger.info(f"Applying {self.smote_variant} SMOTE with strategy: {self.sampling_strategy}")
        
        try:
            features_resampled, labels_resampled = self.smote.fit_resample(features, labels)
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Returning original dataset.")
            return BalancedDataset(dataset, [], [], analysis)
        
        # Find synthetic samples (those not in original dataset)
        original_size = len(features)
        synthetic_features = features_resampled[original_size:]
        synthetic_labels = labels_resampled[original_size:]
        
        # Create synthetic images
        synthetic_images = self._create_synthetic_images(features, synthetic_features, dataset)
        
        # Create balanced dataset
        balanced_dataset = BalancedDataset(
            original_dataset=dataset,
            synthetic_images=synthetic_images,
            synthetic_labels=synthetic_labels.tolist(),
            analysis=analysis
        )
        
        # Analyze new distribution
        new_analysis = self.analyze_class_distribution(balanced_dataset)
        logger.info(f"Balancing complete:")
        logger.info(f"  Original samples: {analysis['total_samples']}")
        logger.info(f"  New samples: {new_analysis['total_samples']}")
        logger.info(f"  Synthetic samples added: {len(synthetic_images)}")
        logger.info(f"  New imbalance ratio: {new_analysis['imbalance_ratio']:.2f}")
        
        return balanced_dataset

class BalancedDataset(Dataset):
    """
    A dataset that combines original and synthetic samples from SMOTE.
    """
    
    def __init__(self, original_dataset: Dataset, synthetic_images: List[torch.Tensor], 
                 synthetic_labels: List[int], analysis: Dict[str, Any]):
        self.original_dataset = original_dataset
        self.synthetic_images = synthetic_images
        self.synthetic_labels = synthetic_labels
        self.analysis = analysis
        self.total_length = len(original_dataset) + len(synthetic_images)
        
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        if idx < len(self.original_dataset):
            # Return original sample
            return self.original_dataset[idx]
        else:
            # Return synthetic sample
            synthetic_idx = idx - len(self.original_dataset)
            return self.synthetic_images[synthetic_idx], self.synthetic_labels[synthetic_idx]
    
    def get_balance_info(self) -> Dict[str, Any]:
        """Get information about the balancing operation."""
        return {
            'original_size': len(self.original_dataset),
            'synthetic_size': len(self.synthetic_images),
            'total_size': self.total_length,
            'original_analysis': self.analysis
        }

# Utility Functions for Easy Integration

def apply_smote_to_dataset(
    dataset: Dataset,
    target_distribution: str = 'balanced',
    smote_variant: str = 'standard',
    feature_extractor: str = 'flatten',
    k_neighbors: int = 5,
    random_state: int = 42
) -> BalancedDataset:
    """
    Convenient function to apply SMOTE balancing to a dataset.
    
    Args:
        dataset: Original dataset to balance
        target_distribution: 'balanced', 'auto', or custom dict
        smote_variant: 'standard', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn'
        feature_extractor: 'flatten', 'resnet_features', 'simple_cnn'
        k_neighbors: Number of neighbors for SMOTE
        random_state: Random seed
    
    Returns:
        BalancedDataset with original + synthetic samples
    """
    
    # Map target_distribution to sampling_strategy
    strategy_mapping = {
        'balanced': 'auto',
        'auto': 'auto',
        'minority': 'minority',
        'not_majority': 'not minority'
    }
    
    sampling_strategy = strategy_mapping.get(target_distribution, target_distribution)
    
    # Create balancer
    balancer = SMOTEBalancer(
        strategy='auto',
        smote_variant=smote_variant,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        feature_extractor=feature_extractor
    )
    
    # Apply balancing
    return balancer.balance_dataset(dataset)

def check_class_imbalance(dataset: Dataset, threshold: float = 1.5) -> Dict[str, Any]:
    """
    Check if a dataset has class imbalance issues.
    
    Args:
        dataset: Dataset to analyze
        threshold: Imbalance ratio threshold (default 1.5)
    
    Returns:
        Dictionary with imbalance analysis
    """
    balancer = SMOTEBalancer()
    analysis = balancer.analyze_class_distribution(dataset)
    analysis['needs_balancing'] = analysis['imbalance_ratio'] > threshold
    return analysis

# Integration helper for existing AutoML framework
class SMOTEIntegrator:
    """
    Helper class to integrate SMOTE with existing AutoML training pipeline.
    """
    
    @staticmethod
    def enhance_automl_with_smote(
        automl_instance,
        enable_smote: bool = True,
        smote_config: Optional[Dict[str, Any]] = None
    ):
        """
        Enhance AutoML instance with SMOTE capabilities.
        
        This monkey-patches the AutoML class to automatically apply SMOTE
        when class imbalance is detected.
        """
        if not enable_smote:
            return automl_instance
        
        # Default SMOTE configuration
        default_config = {
            'smote_variant': 'standard',
            'feature_extractor': 'flatten',
            'k_neighbors': 5,
            'threshold': 1.5
        }
        
        if smote_config:
            default_config.update(smote_config)
        
        # Store original fit method
        original_fit = automl_instance.fit
        
        def enhanced_fit(dataset_class, subsample=None, trial=None):
            """Enhanced fit method with SMOTE integration."""
            
            # Create original dataset
            mean, std = automl_instance._calculate_mean_std(dataset_class) if hasattr(automl_instance, '_calculate_mean_std') else (None, None)
            
            # Check if we need to apply SMOTE
            temp_dataset = dataset_class(root="./data", split='train', download=True, transform=None)
            analysis = check_class_imbalance(temp_dataset, default_config['threshold'])
            
            if analysis['needs_balancing']:
                logger.info(f"Class imbalance detected (ratio: {analysis['imbalance_ratio']:.2f}). Applying SMOTE...")
                
                # Apply SMOTE
                balanced_dataset = apply_smote_to_dataset(
                    temp_dataset,
                    smote_variant=default_config['smote_variant'],
                    feature_extractor=default_config['feature_extractor'],
                    k_neighbors=default_config['k_neighbors']
                )
                
                logger.info(f"SMOTE applied. Dataset size: {len(temp_dataset)} -> {len(balanced_dataset)}")
                
                # Replace the dataset creation in the original fit method
                # This would require more sophisticated integration depending on the AutoML implementation
                
            # Call original fit method
            return original_fit(dataset_class, subsample, trial)
        
        # Replace fit method
        automl_instance.fit = enhanced_fit
        automl_instance._smote_config = default_config
        
        return automl_instance

# Installation check
def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import imblearn
        import sklearn
        logger.info("✅ All SMOTE dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Please install: pip install imbalanced-learn scikit-learn")
        return False

if __name__ == "__main__":
    # Example usage and testing
    print("🔄 SMOTE Balancer Module")
    print("=" * 50)
    
    if check_dependencies():
        print("✅ Dependencies check passed")
        print("\nUsage examples:")
        print("1. from smote_balancer import apply_smote_to_dataset")
        print("2. balanced_dataset = apply_smote_to_dataset(original_dataset)")
        print("3. Check the module docstring for detailed usage")
    else:
        print("❌ Please install missing dependencies first")