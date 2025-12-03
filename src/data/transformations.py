# src/data/transforms.py

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

class DataAugmentation:
    """
    Data augmentation for neural signals.
    All transforms are applied to input features, not targets.
    """
    def __init__(self, config):
        """
        Args:
            config: Dict with augmentation parameters
        """
        self.white_noise_std = config.get('white_noise_std', 0.0)
        self.constant_offset_std = config.get('constant_offset_std', 0.0)
        self.random_walk_std = config.get('random_walk_std', 0.0)
        self.random_walk_axis = config.get('random_walk_axis', -1)
        self.static_gain_std = config.get('static_gain_std', 0.0)
        self.random_cut = config.get('random_cut', 0)
        self.smooth_kernel_size = config.get('smooth_kernel_size', 0)
        self.smooth_kernel_std = config.get('smooth_kernel_std', 2.0)
        self.smooth_data = config.get('smooth_data', False)
        
        # Print active transforms
        active_transforms = []
        if self.white_noise_std > 0:
            active_transforms.append(f"White Noise (std={self.white_noise_std})")
        if self.constant_offset_std > 0:
            active_transforms.append(f"Constant Offset (std={self.constant_offset_std})")
        if self.random_walk_std > 0:
            active_transforms.append(f"Random Walk (std={self.random_walk_std})")
        if self.static_gain_std > 0:
            active_transforms.append(f"Static Gain (std={self.static_gain_std})")
        if self.random_cut > 0:
            active_transforms.append(f"Random Cut (n={self.random_cut})")
        if self.smooth_data and self.smooth_kernel_size > 0:
            active_transforms.append(f"Gaussian Smooth (kernel={self.smooth_kernel_size})")
        
        if active_transforms:
            print("\nActive Data Transforms:")
            for t in active_transforms:
                print(f"  - {t}")
        else:
            print("\nNo data transforms enabled")
    
    def __call__(self, features):
        """
        Apply all transforms to neural features.
        
        Args:
            features: torch.Tensor of shape (time_steps, n_features)
        
        Returns:
            Augmented features of same shape
        """
        # Convert to numpy for easier manipulation
        if isinstance(features, torch.Tensor):
            features_np = features.numpy()
            was_tensor = True
        else:
            features_np = features
            was_tensor = False
        
        # Apply transforms in sequence
        if self.smooth_data and self.smooth_kernel_size > 0:
            features_np = self._smooth(features_np)
        
        if self.white_noise_std > 0:
            features_np = self._add_white_noise(features_np)
        
        if self.constant_offset_std > 0:
            features_np = self._add_constant_offset(features_np)
        
        if self.random_walk_std > 0:
            features_np = self._add_random_walk(features_np)
        
        if self.static_gain_std > 0:
            features_np = self._apply_static_gain(features_np)
        
        if self.random_cut > 0:
            features_np = self._random_cut(features_np)
        
        # Convert back to tensor if needed
        if was_tensor:
            return torch.FloatTensor(features_np)
        return features_np
    
    def _smooth(self, features):
        """Apply Gaussian smoothing along time axis"""
        # features: (time, features)
        # Smooth each feature channel independently
        smoothed = gaussian_filter1d(
            features,
            sigma=self.smooth_kernel_std,
            axis=0,  # Time axis
            mode='nearest'
        )
        return smoothed
    
    def _add_white_noise(self, features):
        """Add Gaussian white noise"""
        noise = np.random.normal(0, self.white_noise_std, features.shape)
        return features + noise
    
    def _add_constant_offset(self, features):
        """Add a constant offset to all time steps"""
        # Sample one offset per feature channel
        offset = np.random.normal(0, self.constant_offset_std, (1, features.shape[1]))
        return features + offset
    
    def _add_random_walk(self, features):
        """Add random walk noise"""
        # Generate random walk
        steps = np.random.normal(0, self.random_walk_std, features.shape)
        
        if self.random_walk_axis == -1:
            # Random walk along time axis for each feature
            random_walk = np.cumsum(steps, axis=0)
        elif self.random_walk_axis == 0:
            # Random walk along feature axis for each time step
            random_walk = np.cumsum(steps, axis=1)
        else:
            random_walk = steps
        
        return features + random_walk
    
    def _apply_static_gain(self, features):
        """Apply multiplicative gain"""
        # Sample one gain per feature channel
        gain = np.random.normal(1.0, self.static_gain_std, (1, features.shape[1]))
        # Ensure gains are positive
        gain = np.abs(gain)
        return features * gain
    
    def _random_cut(self, features):
        """Randomly zero out segments of the signal"""
        time_steps = features.shape[0]
        features_out = features.copy()
        
        for _ in range(self.random_cut):
            # Random segment length (5-15% of total length)
            cut_length = np.random.randint(
                int(0.05 * time_steps),
                int(0.15 * time_steps)
            )
            
            # Random start position
            start_idx = np.random.randint(0, time_steps - cut_length)
            
            # Zero out this segment
            features_out[start_idx:start_idx + cut_length, :] = 0
        
        return features_out


class NoTransform:
    """Identity transform - does nothing"""
    def __call__(self, features):
        return features