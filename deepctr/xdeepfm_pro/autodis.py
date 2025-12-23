# -*- coding:utf-8 -*-
"""
AutoDis (Automatic Discretization) Layer for Dense Features

Reference:
    [1] Guo H, Chen B, Tang R, et al. An Embedding Learning Framework for 
        Numerical Features in CTR Prediction[C]. KDD 2021.
        (https://arxiv.org/abs/2012.08986)

AutoDis automatically discretizes dense/numerical features into learnable
embeddings, providing a more expressive representation than raw values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AutoDisLayer(nn.Module):
    """
    Automatic Discretization Layer for Dense Features
    
    Converts numerical features into embeddings by learning soft discretization.
    Each numerical value is mapped to a weighted combination of bucket embeddings.
    
    Args:
        num_features: Number of dense features
        num_buckets: Number of discretization buckets per feature
        embedding_dim: Output embedding dimension
        temperature: Temperature for soft discretization (higher = harder)
        keep_raw: Whether to keep raw dense values in output
        device: Device to run on
    """
    
    def __init__(
        self,
        num_features: int,
        num_buckets: int = 16,
        embedding_dim: int = 8,
        temperature: float = 1.0,
        keep_raw: bool = True,
        device: str = 'cpu'
    ):
        super(AutoDisLayer, self).__init__()
        
        self.num_features = num_features
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.keep_raw = keep_raw
        self.device = device
        
        if num_features > 0:
            # Meta-embeddings: learnable bucket centers for each feature
            # Shape: [num_features, num_buckets, embedding_dim]
            self.meta_embeddings = nn.Parameter(
                torch.randn(num_features, num_buckets, embedding_dim) * 0.01
            )
            
            # Projection layers to compute bucket weights
            # For each feature, project the input value to bucket scores
            self.bucket_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, num_buckets),
                    nn.LeakyReLU(0.2),
                    nn.Linear(num_buckets, num_buckets)
                ) for _ in range(num_features)
            ])
            
            # Optional: learnable temperature per feature
            self.feature_temperatures = nn.Parameter(
                torch.ones(num_features) * temperature
            )
        
        self.to(device)
    
    def forward(self, dense_values: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for AutoDis
        
        Args:
            dense_values: List of dense feature tensors, each [batch, 1]
            
        Returns:
            autodis_embeddings: Combined AutoDis embeddings [batch, num_features * embedding_dim]
            dense_embeddings_list: List of individual embeddings, each [batch, 1, embedding_dim]
        """
        if self.num_features == 0 or len(dense_values) == 0:
            batch_size = dense_values[0].shape[0] if dense_values else 1
            return (
                torch.zeros(batch_size, 0, device=self.device),
                []
            )
        
        batch_size = dense_values[0].shape[0]
        dense_embeddings = []
        
        for i, dense_val in enumerate(dense_values):
            # Ensure correct shape: [batch, 1]
            if len(dense_val.shape) == 1:
                dense_val = dense_val.unsqueeze(-1)
            
            # Compute bucket scores
            bucket_scores = self.bucket_projectors[i](dense_val)  # [batch, num_buckets]
            
            # Apply temperature-scaled softmax for soft discretization
            temp = self.feature_temperatures[i]
            bucket_weights = F.softmax(bucket_scores / temp, dim=-1)  # [batch, num_buckets]
            
            # Weighted sum of meta embeddings
            # meta_embeddings[i]: [num_buckets, embedding_dim]
            # bucket_weights: [batch, num_buckets]
            # Output: [batch, embedding_dim]
            feature_embedding = torch.matmul(
                bucket_weights, 
                self.meta_embeddings[i]
            )  # [batch, embedding_dim]
            
            # Add dimension for consistency: [batch, 1, embedding_dim]
            dense_embeddings.append(feature_embedding.unsqueeze(1))
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(dense_embeddings, dim=1)  # [batch, num_features, embedding_dim]
        flat_embeddings = all_embeddings.view(batch_size, -1)  # [batch, num_features * embedding_dim]
        
        return flat_embeddings, dense_embeddings
    
    def get_bucket_indices(self, dense_values: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Get the dominant bucket index for each feature (for reconstruction)
        
        Args:
            dense_values: List of dense feature tensors
            
        Returns:
            List of bucket indices for each feature
        """
        bucket_indices = []
        
        for i, dense_val in enumerate(dense_values):
            if len(dense_val.shape) == 1:
                dense_val = dense_val.unsqueeze(-1)
            
            bucket_scores = self.bucket_projectors[i](dense_val)
            bucket_idx = bucket_scores.argmax(dim=-1)
            bucket_indices.append(bucket_idx)
        
        return bucket_indices


class DenseFeatureEncoder(nn.Module):
    """
    Combined Dense Feature Encoder with AutoDis and raw value support
    
    This module provides a unified interface for encoding dense features,
    optionally using AutoDis for richer representations.
    
    Args:
        dense_feature_names: List of dense feature names
        embedding_dim: Output embedding dimension for AutoDis
        use_autodis: Whether to use AutoDis (if False, just uses raw values)
        num_buckets: Number of AutoDis buckets
        temperature: AutoDis temperature
        device: Device to run on
    """
    
    def __init__(
        self,
        dense_feature_names: List[str],
        embedding_dim: int = 8,
        use_autodis: bool = True,
        num_buckets: int = 16,
        temperature: float = 1.0,
        device: str = 'cpu'
    ):
        super(DenseFeatureEncoder, self).__init__()
        
        self.dense_feature_names = dense_feature_names
        self.embedding_dim = embedding_dim
        self.use_autodis = use_autodis
        self.num_features = len(dense_feature_names)
        self.device = device
        
        if use_autodis and self.num_features > 0:
            self.autodis = AutoDisLayer(
                num_features=self.num_features,
                num_buckets=num_buckets,
                embedding_dim=embedding_dim,
                temperature=temperature,
                device=device
            )
        else:
            self.autodis = None
        
        self.to(device)
    
    def forward(
        self, 
        dense_values: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Encode dense features
        
        Args:
            dense_values: List of dense feature tensors
            
        Returns:
            encoded_output: Flat encoded representation
            embedding_list: List of embeddings for each feature
            raw_values: Concatenated raw values
        """
        if self.num_features == 0 or len(dense_values) == 0:
            batch_size = dense_values[0].shape[0] if dense_values else 1
            return (
                torch.zeros(batch_size, 0, device=self.device),
                [],
                torch.zeros(batch_size, 0, device=self.device)
            )
        
        # Concatenate raw values
        raw_values = torch.cat(dense_values, dim=-1)  # [batch, num_features]
        
        if self.use_autodis and self.autodis is not None:
            flat_embeddings, embedding_list = self.autodis(dense_values)
            return flat_embeddings, embedding_list, raw_values
        else:
            # Return raw values as embeddings
            embedding_list = [dv.unsqueeze(-1) for dv in dense_values]
            return raw_values, embedding_list, raw_values
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder"""
        if self.use_autodis:
            return self.num_features * self.embedding_dim
        else:
            return self.num_features

