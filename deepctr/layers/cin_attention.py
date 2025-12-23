# -*- coding:utf-8 -*-
"""
CIN with Multi-Head Self-Attention Pooling
解决原始CIN中Sum Pooling导致的信息丢失问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..layers.activation import activation_layer


def _get_valid_num_heads(embed_dim, num_heads):
    """自动调整num_heads使其能整除embed_dim"""
    if embed_dim % num_heads == 0:
        return num_heads
    # 找到能整除embed_dim的最大头数（不超过原num_heads）
    for h in range(num_heads, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1  # fallback to single head


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块
    
    Input shape:
        - 3D tensor with shape: ``(batch_size, seq_len, embed_dim)``
    Output shape:
        - 3D tensor with shape: ``(batch_size, seq_len, embed_dim)``
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.0, device='cpu'):
        super(MultiHeadSelfAttention, self).__init__()
        
        # 自动调整num_heads使其能整除embed_dim
        num_heads = _get_valid_num_heads(embed_dim, num_heads)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        self.to(device)
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output


class AttentionPooling(nn.Module):
    """注意力池化层，用于将序列聚合为单个向量
    
    Input shape:
        - 3D tensor with shape: ``(batch_size, seq_len, embed_dim)``
    Output shape:
        - 2D tensor with shape: ``(batch_size, embed_dim)``
    """
    
    def __init__(self, embed_dim, hidden_dim=None, device='cpu'):
        super(AttentionPooling, self).__init__()
        
        hidden_dim = hidden_dim or embed_dim
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self._init_weights()
        self.to(device)
    
    def _init_weights(self):
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, embed_dim)
        """
        # Compute attention weights
        attn_scores = self.attention(x)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        output = torch.sum(attn_weights * x, dim=1)  # (batch_size, embed_dim)
        
        return output


class CINAttention(nn.Module):
    """CIN with Multi-Head Self-Attention Pooling
    
    解决原始CIN中Sum Pooling导致的信息丢失问题，
    使用多头自注意力机制对特征图进行融合，保留更多信息。
    
    Input shape:
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``
    Output shape:
        - 2D tensor with shape: ``(batch_size, output_dim)``
        - output_dim = featuremap_num (和原始CIN兼容)
    
    Arguments:
        - **field_size**: Positive integer, number of feature groups.
        - **layer_size**: list of int. Feature maps in each layer.
        - **activation**: activation function name used on feature maps.
        - **split_half**: bool. If set to True, half of the feature maps in each hidden will connect to output unit.
        - **num_heads**: int. Number of attention heads in MHSA.
        - **attn_dropout**: float. Dropout rate for attention.
        - **use_layer_norm**: bool. Whether to use layer normalization.
        - **use_residual**: bool. Whether to use residual connection in attention.
        - **l2_reg**: float. L2 regularizer strength.
        - **seed**: A Python integer to use as random seed.
        - **device**: str, ``"cpu"`` or ``"cuda:0"``
    """
    
    def __init__(self, field_size, embedding_size, layer_size=(128, 128), activation='relu', 
                 split_half=True, num_heads=4, attn_dropout=0.0, use_layer_norm=True,
                 use_residual=True, l2_reg=1e-5, seed=1024, device='cpu'):
        super(CINAttention, self).__init__()
        
        if len(layer_size) == 0:
            raise ValueError("layer_size must be a list(tuple) of length greater than 1")
        
        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed
        self.embedding_size = embedding_size
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # CIN卷积层
        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1)
            )
            
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True"
                    )
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        
        # 计算feature map总数
        if split_half:
            self.featuremap_num = sum(layer_size[:-1]) // 2 + layer_size[-1]
        else:
            self.featuremap_num = sum(layer_size)
        
        # 多头自注意力层
        self.mhsa = MultiHeadSelfAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=attn_dropout,
            device=device
        )
        
        # Layer Normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_size)
        
        # 注意力池化层
        self.attn_pooling = AttentionPooling(
            embed_dim=embedding_size,
            hidden_dim=embedding_size,
            device=device
        )
        
        # 输出投影层，将embedding_size映射到featuremap_num
        # 保持与原始CIN输出维度兼容
        self.output_proj = nn.Linear(embedding_size, self.featuremap_num, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        self.to(device)
    
    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, field_size, embedding_size)
        Returns:
            result: (batch_size, featuremap_num)
        """
        if len(inputs.shape) != 3:
            raise ValueError(
                f"Unexpected inputs dimensions {len(inputs.shape)}, expect to be 3 dimensions"
            )
        
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]  # embedding_size
        hidden_nn_layers = [inputs]
        final_result = []
        
        # CIN交叉层
        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0: 外积
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0]
            )
            # x.shape = (batch_size, hi * m, dim)
            x = x.reshape(
                batch_size, 
                hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], 
                dim
            )
            # 卷积压缩
            x = self.conv1ds[i](x)
            
            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)
            
            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1
                    )
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out
            
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        
        # 拼接所有特征图: (batch_size, featuremap_num, embedding_size)
        result = torch.cat(final_result, dim=1)
        
        # ============ 替代Sum Pooling的部分 ============
        # 原始CIN: result = torch.sum(result, -1)  # 简单求和，丢失信息
        
        # 新方法: 使用多头自注意力进行特征融合
        # result形状: (batch_size, featuremap_num, embedding_size)
        # 将featuremap_num看作序列长度，embedding_size看作特征维度
        
        # 1. 多头自注意力
        attn_output = self.mhsa(result)  # (batch_size, featuremap_num, embedding_size)
        
        # 2. 残差连接
        if self.use_residual:
            attn_output = attn_output + result
        
        # 3. Layer Normalization
        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output)
        
        # 4. 注意力池化：将序列聚合为单个向量
        pooled = self.attn_pooling(attn_output)  # (batch_size, embedding_size)
        
        # 5. 投影到与原始CIN相同的输出维度
        output = self.output_proj(pooled)  # (batch_size, featuremap_num)
        
        return output


class CINAttentionV2(nn.Module):
    """CIN with Multi-Head Self-Attention Pooling (V2版本)
    
    不使用投影层，直接保持更高维度的输出。
    需要配合修改后的xDeepFM使用。
    
    Input shape:
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``
    Output shape:
        - 2D tensor with shape: ``(batch_size, embedding_size)``
    """
    
    def __init__(self, field_size, embedding_size, layer_size=(128, 128), activation='relu', 
                 split_half=True, num_heads=4, attn_dropout=0.0, use_layer_norm=True,
                 use_residual=True, num_attn_layers=1, l2_reg=1e-5, seed=1024, device='cpu'):
        super(CINAttentionV2, self).__init__()
        
        if len(layer_size) == 0:
            raise ValueError("layer_size must be a list(tuple) of length greater than 1")
        
        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed
        self.embedding_size = embedding_size
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.num_attn_layers = num_attn_layers
        
        # CIN卷积层
        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1)
            )
            
            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True"
                    )
                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)
        
        # 计算feature map总数
        if split_half:
            self.featuremap_num = sum(layer_size[:-1]) // 2 + layer_size[-1]
        else:
            self.featuremap_num = sum(layer_size)
        
        # 多层多头自注意力
        self.mhsa_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        for _ in range(num_attn_layers):
            self.mhsa_layers.append(
                MultiHeadSelfAttention(
                    embed_dim=embedding_size,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    device=device
                )
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(embedding_size))
        
        # 注意力池化层
        self.attn_pooling = AttentionPooling(
            embed_dim=embedding_size,
            hidden_dim=embedding_size,
            device=device
        )
        
        self.to(device)
    
    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, field_size, embedding_size)
        Returns:
            result: (batch_size, embedding_size)
        """
        if len(inputs.shape) != 3:
            raise ValueError(
                f"Unexpected inputs dimensions {len(inputs.shape)}, expect to be 3 dimensions"
            )
        
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []
        
        # CIN交叉层
        for i, size in enumerate(self.layer_size):
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0]
            )
            x = x.reshape(
                batch_size, 
                hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], 
                dim
            )
            x = self.conv1ds[i](x)
            
            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)
            
            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1
                    )
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out
            
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        
        # 拼接所有特征图: (batch_size, featuremap_num, embedding_size)
        result = torch.cat(final_result, dim=1)
        
        # 多层自注意力
        for i in range(self.num_attn_layers):
            attn_output = self.mhsa_layers[i](result)
            
            if self.use_residual:
                attn_output = attn_output + result
            
            if self.use_layer_norm:
                attn_output = self.layer_norms[i](attn_output)
            
            result = attn_output
        
        # 注意力池化
        output = self.attn_pooling(result)  # (batch_size, embedding_size)
        
        return output

