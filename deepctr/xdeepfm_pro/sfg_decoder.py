# -*- coding:utf-8 -*-
"""
Supervised Feature Generation (SFG) Decoder Module

This module implements the SFG decoder for reconstructing original features
from embeddings, using label-aware attention to focus on click samples (Y=1).

Reference:
    The SFG approach helps solve embedding collapse by forcing the model to
    remember feature patterns that lead to clicks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class SFGDecoder(nn.Module):
    """
    Supervised Feature Generation Decoder
    
    A lightweight MLP decoder that reconstructs original features from embeddings.
    Uses label-aware loss to focus on positive samples.
    
    Args:
        embedding_dim: Dimension of input embeddings
        sparse_feature_dims: Dict mapping sparse feature names to their vocabulary sizes
        dense_feature_names: List of dense feature names
        hidden_units: Tuple of hidden layer sizes for the decoder MLP
        dropout_rate: Dropout rate for regularization
        use_label_aware_attention: Whether to use label-aware attention mechanism
        device: Device to run on ('cpu' or 'cuda:X')
    """
    
    def __init__(
        self,
        embedding_dim: int,
        sparse_feature_dims: Dict[str, int],
        dense_feature_names: List[str],
        hidden_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.1,
        use_label_aware_attention: bool = True,
        device: str = 'cpu'
    ):
        super(SFGDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.sparse_feature_dims = sparse_feature_dims
        self.dense_feature_names = dense_feature_names
        self.use_label_aware_attention = use_label_aware_attention
        self.device = device
        
        self.num_sparse_features = len(sparse_feature_dims)
        self.num_dense_features = len(dense_feature_names)
        
        # Input dimension: concatenated embeddings for all sparse features + dense features
        # For sparse: num_sparse_features * embedding_dim
        # For dense: num_dense_features (each dense feature has dimension 1)
        input_dim = self.num_sparse_features * embedding_dim + self.num_dense_features
        
        # Build decoder MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate output heads for sparse and dense features
        # Sparse features: use softmax (cross-entropy loss)
        self.sparse_heads = nn.ModuleDict()
        for feat_name, vocab_size in sparse_feature_dims.items():
            self.sparse_heads[feat_name] = nn.Linear(prev_dim, vocab_size)
        
        # Dense features: direct regression (MSE loss)
        if self.num_dense_features > 0:
            self.dense_head = nn.Linear(prev_dim, self.num_dense_features)
        else:
            self.dense_head = None
        
        # Label-aware attention for weighting reconstruction importance
        if use_label_aware_attention:
            self.label_attention = LabelAwareAttention(
                input_dim=input_dim,
                hidden_dim=hidden_units[0] if hidden_units else 64,
                device=device
            )
        
        self.to(device)
    
    def forward(
        self,
        sparse_embeddings: List[torch.Tensor],
        dense_values: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the SFG decoder
        
        Args:
            sparse_embeddings: List of sparse feature embeddings, each [batch, 1, embed_dim]
            dense_values: List of dense feature values, each [batch, 1]
            labels: Optional labels for label-aware attention [batch, 1]
            
        Returns:
            sparse_logits: Dict mapping feature names to reconstruction logits
            dense_preds: Tensor of dense feature predictions [batch, num_dense]
        """
        # Concatenate sparse embeddings
        if len(sparse_embeddings) > 0:
            # Each embedding: [batch, 1, embed_dim] -> squeeze -> [batch, embed_dim]
            sparse_concat = torch.cat([
                emb.squeeze(1) if len(emb.shape) == 3 else emb 
                for emb in sparse_embeddings
            ], dim=-1)  # [batch, num_sparse * embed_dim]
        else:
            sparse_concat = torch.zeros(
                sparse_embeddings[0].shape[0], 0, 
                device=self.device
            )
        
        # Concatenate dense values
        if len(dense_values) > 0:
            dense_concat = torch.cat(dense_values, dim=-1)  # [batch, num_dense]
        else:
            dense_concat = torch.zeros(
                sparse_concat.shape[0], 0,
                device=self.device
            )
        
        # Full input
        decoder_input = torch.cat([sparse_concat, dense_concat], dim=-1)
        
        # Apply label-aware attention if enabled and labels provided
        if self.use_label_aware_attention and labels is not None:
            attention_weights = self.label_attention(decoder_input, labels)
            decoder_input = decoder_input * attention_weights
        
        # Shared MLP layers
        hidden = self.shared_layers(decoder_input)
        
        # Sparse feature reconstruction (logits for cross-entropy)
        sparse_logits = {}
        for feat_name in self.sparse_feature_dims.keys():
            sparse_logits[feat_name] = self.sparse_heads[feat_name](hidden)
        
        # Dense feature reconstruction
        if self.dense_head is not None:
            dense_preds = self.dense_head(hidden)
        else:
            dense_preds = torch.zeros(hidden.shape[0], 0, device=self.device)
        
        return sparse_logits, dense_preds


class LabelAwareAttention(nn.Module):
    """
    Label-Aware Attention Module
    
    Computes attention weights based on the input and label information.
    This allows the model to focus more on patterns from positive samples.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, device: str = 'cpu'):
        super(LabelAwareAttention, self).__init__()
        
        # Label embedding (binary: 0 or 1)
        self.label_embedding = nn.Embedding(2, hidden_dim)
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label-aware attention weights
        
        Args:
            x: Input features [batch, input_dim]
            labels: Labels [batch, 1] or [batch]
            
        Returns:
            Attention weights [batch, input_dim]
        """
        # Get label embeddings
        if len(labels.shape) > 1:
            labels = labels.squeeze(-1)
        label_emb = self.label_embedding(labels.long())  # [batch, hidden_dim]
        
        # Concatenate input with label embedding
        combined = torch.cat([x, label_emb], dim=-1)
        
        # Compute attention weights
        attention_weights = self.attention_net(combined)
        
        return attention_weights


class SFGLoss(nn.Module):
    """
    Supervised Feature Generation Loss
    
    Computes the reconstruction loss for SFG:
    - Cross-entropy for sparse features
    - MSE for dense features
    
    Optionally only computes loss on positive samples (Y=1).
    """
    
    def __init__(
        self,
        sparse_feature_names: List[str],
        dense_feature_names: List[str],
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        positive_only: bool = True,
        label_smooth: float = 0.0,
        device: str = 'cpu'
    ):
        super(SFGLoss, self).__init__()
        
        self.sparse_feature_names = sparse_feature_names
        self.dense_feature_names = dense_feature_names
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.positive_only = positive_only
        self.label_smooth = label_smooth
        self.device = device
        
    def forward(
        self,
        sparse_logits: Dict[str, torch.Tensor],
        dense_preds: torch.Tensor,
        sparse_targets: Dict[str, torch.Tensor],
        dense_targets: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute SFG reconstruction loss
        
        Args:
            sparse_logits: Dict of predicted logits for each sparse feature
            dense_preds: Predicted dense feature values [batch, num_dense]
            sparse_targets: Dict of target indices for each sparse feature
            dense_targets: Target dense feature values [batch, num_dense]
            labels: Binary labels [batch, 1] or [batch]
            
        Returns:
            total_loss: Total SFG loss
            loss_dict: Dictionary of individual losses for logging
        """
        if len(labels.shape) > 1:
            labels = labels.squeeze(-1)
        
        # Create mask for positive samples
        if self.positive_only:
            positive_mask = (labels == 1).float()
            num_positive = positive_mask.sum() + 1e-8
        else:
            positive_mask = torch.ones_like(labels).float()
            num_positive = labels.shape[0]
        
        loss_dict = {}
        total_sparse_loss = torch.tensor(0.0, device=self.device)
        total_dense_loss = torch.tensor(0.0, device=self.device)
        
        # Sparse feature loss (Cross-Entropy)
        for feat_name in self.sparse_feature_names:
            if feat_name in sparse_logits and feat_name in sparse_targets:
                logits = sparse_logits[feat_name]  # [batch, vocab_size]
                targets = sparse_targets[feat_name].long()  # [batch] or [batch, 1]
                
                if len(targets.shape) > 1:
                    targets = targets.squeeze(-1)
                
                # Per-sample cross entropy
                ce_loss = F.cross_entropy(logits, targets, reduction='none')
                
                # Apply positive mask
                masked_loss = (ce_loss * positive_mask).sum() / num_positive
                
                total_sparse_loss = total_sparse_loss + masked_loss
                loss_dict[f'sfg_sparse_{feat_name}'] = masked_loss.item()
        
        # Dense feature loss (MSE)
        if len(self.dense_feature_names) > 0 and dense_preds.shape[1] > 0:
            # MSE per sample
            mse_loss = F.mse_loss(dense_preds, dense_targets, reduction='none')
            mse_loss = mse_loss.mean(dim=-1)  # Average over features
            
            # Apply positive mask
            masked_mse = (mse_loss * positive_mask).sum() / num_positive
            
            total_dense_loss = masked_mse
            loss_dict['sfg_dense'] = masked_mse.item()
        
        # Combine losses
        total_loss = self.sparse_weight * total_sparse_loss + self.dense_weight * total_dense_loss
        loss_dict['sfg_total'] = total_loss.item()
        
        return total_loss, loss_dict

