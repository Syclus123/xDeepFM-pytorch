# -*- coding:utf-8 -*-
"""
xDeepFM with Multi-Head Self-Attention CIN
解决原始xDeepFM中CIN层Sum Pooling导致的信息丢失问题

Author:
    Based on xDeepFM by Wutong Zhang
    Attention enhancement added

Reference:
    [1] Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature 
        Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.
    [2] Vaswani A, et al. Attention is all you need[C]. NeurIPS 2017.
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN
from ..layers.cin_attention import CINAttention, CINAttentionV2


class xDeepFMAttention(BaseModel):
    """xDeepFM with Multi-Head Self-Attention enhanced CIN.
    
    使用多头自注意力机制替代CIN中的Sum Pooling，解决信息丢失问题。
    
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of deep net.
    :param cin_layer_size: list, list of positive integer or empty list, the feature maps in each hidden layer of CIN.
    :param cin_split_half: bool. If set to True, half of the feature maps in each hidden will connect to output unit.
    :param cin_activation: activation function used on feature maps.
    :param cin_num_heads: int. Number of attention heads in CIN's MHSA.
    :param cin_attn_dropout: float. Dropout rate for CIN attention.
    :param cin_use_layer_norm: bool. Whether to use layer normalization in CIN attention.
    :param cin_use_residual: bool. Whether to use residual connection in CIN attention.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: L2 regularizer strength applied to deep net.
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
    :param task: str, ``"binary"`` for binary logloss or ``"regression"`` for regression loss.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, 
                 dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu',
                 cin_num_heads=4, cin_attn_dropout=0.0, cin_use_layer_norm=True, cin_use_residual=True,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, 
                 init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(xDeepFMAttention, self).__init__(
            linear_feature_columns, dnn_feature_columns, 
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding, 
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if self.use_dnn:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns), 
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

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        
        if self.use_cin:
            field_num = len(self.embedding_dict)
            
            # 获取embedding维度
            embedding_size = self._get_embedding_size(dnn_feature_columns)
            
            if cin_split_half:
                self.featuremap_num = sum(cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            
            # 使用带注意力机制的CIN
            self.cin = CINAttention(
                field_size=field_num,
                embedding_size=embedding_size,
                layer_size=cin_layer_size,
                activation=cin_activation,
                split_half=cin_split_half,
                num_heads=cin_num_heads,
                attn_dropout=cin_attn_dropout,
                use_layer_norm=cin_use_layer_norm,
                use_residual=cin_use_residual,
                l2_reg=l2_reg_cin,
                seed=seed,
                device=device
            )
            
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                l2=l2_reg_cin
            )

        self.to(device)
    
    def _get_embedding_size(self, feature_columns):
        """获取embedding维度"""
        from ..inputs import SparseFeat, VarLenSparseFeat, DenseFeat
        
        for feat in feature_columns:
            if isinstance(feat, SparseFeat):
                return feat.embedding_dim
            elif isinstance(feat, VarLenSparseFeat):
                return feat.sparsefeat.embedding_dim
        
        # 如果没有稀疏特征，返回默认值
        return 4

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        linear_logit = self.linear_model(X)
        
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:  # linear + CIN
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:  # linear + Deep
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + CIN + Deep
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred


class xDeepFMAttentionV2(BaseModel):
    """xDeepFM with Multi-Head Self-Attention enhanced CIN (V2版本)
    
    V2版本：CIN输出维度为embedding_size而非featuremap_num，
    保留更多的信息。
    
    参数与xDeepFMAttention相同。
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, 
                 dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu',
                 cin_num_heads=4, cin_attn_dropout=0.0, cin_use_layer_norm=True, 
                 cin_use_residual=True, cin_num_attn_layers=1,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, 
                 init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(xDeepFMAttentionV2, self).__init__(
            linear_feature_columns, dnn_feature_columns, 
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding, 
            init_std=init_std, seed=seed, task=task,
            device=device, gpus=gpus
        )
        
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        
        if self.use_dnn:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns), 
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

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        
        if self.use_cin:
            field_num = len(self.embedding_dict)
            
            # 获取embedding维度
            embedding_size = self._get_embedding_size(dnn_feature_columns)
            self.cin_output_dim = embedding_size
            
            # 使用带注意力机制的CIN V2
            self.cin = CINAttentionV2(
                field_size=field_num,
                embedding_size=embedding_size,
                layer_size=cin_layer_size,
                activation=cin_activation,
                split_half=cin_split_half,
                num_heads=cin_num_heads,
                attn_dropout=cin_attn_dropout,
                use_layer_norm=cin_use_layer_norm,
                use_residual=cin_use_residual,
                num_attn_layers=cin_num_attn_layers,
                l2_reg=l2_reg_cin,
                seed=seed,
                device=device
            )
            
            # CIN输出为embedding_size维度
            self.cin_linear = nn.Linear(embedding_size, 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                l2=l2_reg_cin
            )

        self.to(device)
    
    def _get_embedding_size(self, feature_columns):
        """获取embedding维度"""
        from ..inputs import SparseFeat, VarLenSparseFeat, DenseFeat
        
        for feat in feature_columns:
            if isinstance(feat, SparseFeat):
                return feat.embedding_dim
            elif isinstance(feat, VarLenSparseFeat):
                return feat.sparsefeat.embedding_dim
        
        return 4

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )

        linear_logit = self.linear_model(X)
        
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

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

        return y_pred

