# -*- coding:utf-8 -*-
"""
xDeepFM Pro - xDeepFM with Supervised Feature Generation (SFG)

This module implements an enhanced version of xDeepFM that addresses embedding
collapse through Supervised Feature Generation (SFG).

Key improvements:
1. SFG Decoder: Auxiliary decoder network to reconstruct original features
2. Label-aware loss: Focus on positive samples (Y=1) for reconstruction
3. AutoDis (optional): Automatic discretization for dense features

Reference:
    [1] Original xDeepFM: Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining 
        Explicit and Implicit Feature Interactions for Recommender Systems[J]. 
        arXiv preprint arXiv:1803.05170, 2018.
    [2] SFG concept: Using auxiliary reconstruction to prevent embedding collapse
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .basemodel_sfg import BaseModelSFG
from ..inputs import combined_dnn_input
from ..layers import DNN, CIN

from .autodis import AutoDisLayer, DenseFeatureEncoder


class xDeepFMPro(BaseModelSFG):
    """
    xDeepFM with Supervised Feature Generation (SFG)
    
    This model extends xDeepFM with:
    1. SFG decoder for feature reconstruction
    2. Label-aware reconstruction loss (focus on click samples)
    3. Optional AutoDis for dense feature encoding
    
    Args:
        linear_feature_columns: Features for linear part
        dnn_feature_columns: Features for DNN part
        dnn_hidden_units: Hidden units for DNN layers
        cin_layer_size: Layer sizes for CIN
        cin_split_half: Whether to split CIN output
        cin_activation: Activation for CIN
        l2_reg_linear: L2 reg for linear part
        l2_reg_embedding: L2 reg for embeddings
        l2_reg_dnn: L2 reg for DNN
        l2_reg_cin: L2 reg for CIN
        init_std: Weight initialization std
        seed: Random seed
        dnn_dropout: Dropout rate for DNN
        dnn_activation: Activation for DNN
        dnn_use_bn: Whether to use batch norm in DNN
        task: 'binary' or 'regression'
        device: Device to run on
        gpus: List of GPUs
        
        # SFG parameters
        use_sfg: Whether to enable SFG
        sfg_weight: Weight for SFG loss (lambda in the paper)
        sfg_hidden_units: Hidden units for SFG decoder
        sfg_dropout: Dropout for SFG decoder
        sfg_positive_only: Only compute SFG loss on positive samples
        sfg_use_label_attention: Use label-aware attention in decoder
        
        # AutoDis parameters
        use_autodis: Whether to use AutoDis for dense features
        autodis_buckets: Number of AutoDis buckets
        autodis_temperature: Temperature for soft discretization
    """

    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        dnn_hidden_units=(256, 256),
        cin_layer_size=(256, 128),
        cin_split_half=True,
        cin_activation='relu',
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0,
        l2_reg_cin=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation='relu',
        dnn_use_bn=False,
        task='binary',
        device='cpu',
        gpus=None,
        # SFG parameters
        use_sfg=True,
        sfg_weight=0.1,
        sfg_hidden_units=(128, 64),
        sfg_dropout=0.1,
        sfg_positive_only=True,
        sfg_use_label_attention=True,
        # AutoDis parameters
        use_autodis=False,
        autodis_buckets=16,
        autodis_temperature=1.0
    ):
        super(xDeepFMPro, self).__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            task=task,
            device=device,
            gpus=gpus,
            use_sfg=use_sfg,
            sfg_weight=sfg_weight,
            sfg_hidden_units=sfg_hidden_units,
            sfg_dropout=sfg_dropout,
            sfg_positive_only=sfg_positive_only,
            sfg_use_label_attention=sfg_use_label_attention
        )
        
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.use_autodis = use_autodis
        
        # AutoDis for dense features
        if use_autodis and len(self.dense_feature_columns) > 0:
            self.autodis_encoder = DenseFeatureEncoder(
                dense_feature_names=[feat.name for feat in self.dense_feature_columns],
                embedding_dim=self.embedding_dim,
                use_autodis=True,
                num_buckets=autodis_buckets,
                temperature=autodis_temperature,
                device=device
            )
            # Adjust DNN input dimension if using AutoDis
            autodis_output_dim = self.autodis_encoder.get_output_dim()
        else:
            self.autodis_encoder = None
            autodis_output_dim = 0
        
        # DNN component
        if self.use_dnn:
            # Calculate DNN input dimension
            base_dnn_input_dim = self.compute_input_dim(dnn_feature_columns)
            
            if use_autodis and self.autodis_encoder is not None:
                # Replace raw dense dim with AutoDis dim
                dense_dim = sum(feat.dimension for feat in self.dense_feature_columns)
                dnn_input_dim = base_dnn_input_dim - dense_dim + autodis_output_dim
            else:
                dnn_input_dim = base_dnn_input_dim
            
            self.dnn = DNN(
                dnn_input_dim,
                dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=dnn_use_bn,
                init_std=init_std,
                device=device
            )
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()),
                l2=l2_reg_dnn
            )
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        # CIN component
        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        
        if self.use_cin:
            field_num = len(self.embedding_dict)
            if cin_split_half:
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
                
            self.cin = CIN(
                field_num,
                cin_layer_size,
                cin_activation,
                cin_split_half,
                l2_reg_cin,
                seed,
                device=device
            )
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                l2=l2_reg_cin
            )

        self.to(device)

    def forward_with_sfg(
        self, 
        X: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with SFG loss computation
        
        Args:
            X: Input tensor
            y: Labels (optional, used for SFG)
            
        Returns:
            y_pred: Prediction tensor
            sfg_info: Dictionary containing SFG loss and other info
        """
        # Get sparse embeddings and dense values
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        # Linear part
        linear_logit = self.linear_model(X)
        
        # CIN part
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        
        # DNN part
        if self.use_dnn:
            # Apply AutoDis to dense features if enabled
            if self.use_autodis and self.autodis_encoder is not None and len(dense_value_list) > 0:
                autodis_output, autodis_embeddings, raw_dense = self.autodis_encoder(dense_value_list)
                # Use AutoDis embeddings for DNN input
                sparse_flat = torch.flatten(
                    torch.cat(sparse_embedding_list, dim=-1), start_dim=1
                )
                dnn_input = torch.cat([sparse_flat, autodis_output], dim=-1)
            else:
                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        # Combine logits
        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        # Compute SFG loss if enabled and labels provided
        sfg_info = None
        if self.use_sfg and y is not None and self.training:
            sfg_loss, sfg_info = self.compute_sfg_loss(
                X, 
                sparse_embedding_list, 
                dense_value_list, 
                y
            )
            sfg_info['sfg_loss'] = sfg_loss

        return y_pred, sfg_info

    def forward(self, X):
        """Standard forward pass"""
        y_pred, _ = self.forward_with_sfg(X, None)
        return y_pred

    def get_embedding_analysis(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze embeddings for the given input
        
        Useful for understanding embedding quality and detecting collapse.
        
        Args:
            X: Input tensor
            
        Returns:
            Dictionary with embedding statistics
        """
        with torch.no_grad():
            sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
                X, self.dnn_feature_columns, self.embedding_dict
            )
            
            # Stack all sparse embeddings
            all_embeddings = torch.cat(sparse_embedding_list, dim=1)  # [batch, num_fields, embed_dim]
            
            # Compute statistics
            mean_embedding = all_embeddings.mean(dim=0)  # [num_fields, embed_dim]
            std_embedding = all_embeddings.std(dim=0)
            
            # Embedding collapse indicator: low variance indicates collapse
            embedding_variance = all_embeddings.var(dim=0).mean()
            
            # Cosine similarity between different samples (should be diverse, not all similar)
            flat_embeddings = all_embeddings.view(all_embeddings.shape[0], -1)
            normalized = flat_embeddings / (flat_embeddings.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim_matrix = torch.mm(normalized, normalized.t())
            avg_cosine_sim = (cosine_sim_matrix.sum() - cosine_sim_matrix.trace()) / \
                            (cosine_sim_matrix.numel() - cosine_sim_matrix.shape[0])
            
            return {
                'mean_embedding': mean_embedding,
                'std_embedding': std_embedding,
                'embedding_variance': embedding_variance,
                'avg_sample_cosine_similarity': avg_cosine_sim,
                'num_fields': all_embeddings.shape[1],
                'embedding_dim': all_embeddings.shape[2]
            }


class xDeepFMProLight(xDeepFMPro):
    """
    Lightweight version of xDeepFM Pro
    
    Same as xDeepFMPro but with smaller default configurations
    for faster training and lower memory usage.
    """
    
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        dnn_hidden_units=(128, 64),
        cin_layer_size=(128, 64),
        cin_split_half=True,
        cin_activation='relu',
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0,
        l2_reg_cin=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation='relu',
        dnn_use_bn=False,
        task='binary',
        device='cpu',
        gpus=None,
        # Lighter SFG defaults
        use_sfg=True,
        sfg_weight=0.05,
        sfg_hidden_units=(64, 32),
        sfg_dropout=0.1,
        sfg_positive_only=True,
        sfg_use_label_attention=True,
        # No AutoDis by default for lighter version
        use_autodis=False,
        autodis_buckets=8,
        autodis_temperature=1.0
    ):
        super(xDeepFMProLight, self).__init__(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            dnn_hidden_units=dnn_hidden_units,
            cin_layer_size=cin_layer_size,
            cin_split_half=cin_split_half,
            cin_activation=cin_activation,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            l2_reg_dnn=l2_reg_dnn,
            l2_reg_cin=l2_reg_cin,
            init_std=init_std,
            seed=seed,
            dnn_dropout=dnn_dropout,
            dnn_activation=dnn_activation,
            dnn_use_bn=dnn_use_bn,
            task=task,
            device=device,
            gpus=gpus,
            use_sfg=use_sfg,
            sfg_weight=sfg_weight,
            sfg_hidden_units=sfg_hidden_units,
            sfg_dropout=sfg_dropout,
            sfg_positive_only=sfg_positive_only,
            sfg_use_label_attention=sfg_use_label_attention,
            use_autodis=use_autodis,
            autodis_buckets=autodis_buckets,
            autodis_temperature=autodis_temperature
        )

