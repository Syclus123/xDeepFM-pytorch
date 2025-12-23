# -*- coding:utf-8 -*-
"""
xDeepFM Pro - Enhanced xDeepFM with Supervised Feature Generation

This package provides an improved version of xDeepFM that addresses
embedding collapse through Supervised Feature Generation (SFG).

Main Components:
- xDeepFMPro: Main model with SFG support
- xDeepFMProLight: Lightweight version with smaller defaults
- SFGDecoder: Auxiliary decoder for feature reconstruction
- AutoDisLayer: Automatic discretization for dense features

Usage:
    from deepctr.xdeepfm_pro import xDeepFMPro, xDeepFMProLight
    
    model = xDeepFMPro(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        use_sfg=True,
        sfg_weight=0.1,
        sfg_positive_only=True,
        device='cuda:0'
    )
"""

from .xdeepfm_pro import xDeepFMPro, xDeepFMProLight
from .sfg_decoder import SFGDecoder, SFGLoss, LabelAwareAttention
from .autodis import AutoDisLayer, DenseFeatureEncoder
from .basemodel_sfg import BaseModelSFG

__all__ = [
    'xDeepFMPro',
    'xDeepFMProLight',
    'SFGDecoder',
    'SFGLoss',
    'LabelAwareAttention',
    'AutoDisLayer',
    'DenseFeatureEncoder',
    'BaseModelSFG'
]

