# -*- coding:utf-8 -*-
"""
BaseModel with Supervised Feature Generation (SFG) Support

This module extends the original BaseModel to support SFG loss computation
during training, enabling the model to learn better embeddings by 
reconstructing original features with label-aware guidance.
"""

from __future__ import print_function

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import (
    build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat,
    get_varlen_pooling_list, create_embedding_matrix, varlen_embedding_lookup
)
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History

from .sfg_decoder import SFGDecoder, SFGLoss


class Linear(nn.Module):
    """Linear part of the model (same as original)"""
    
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModelSFG(nn.Module):
    """
    Base Model with Supervised Feature Generation Support
    
    This extends the original BaseModel with:
    1. SFG Decoder for feature reconstruction
    2. Label-aware loss computation
    3. Combined loss (prediction + reconstruction)
    
    Args:
        linear_feature_columns: Features for linear part
        dnn_feature_columns: Features for DNN part
        l2_reg_linear: L2 regularization for linear part
        l2_reg_embedding: L2 regularization for embeddings
        init_std: Standard deviation for weight initialization
        seed: Random seed
        task: 'binary' or 'regression'
        device: Device to run on
        gpus: List of GPUs for parallel training
        use_sfg: Whether to enable SFG
        sfg_weight: Weight for SFG loss
        sfg_hidden_units: Hidden units for SFG decoder
        sfg_dropout: Dropout rate for SFG decoder
        sfg_positive_only: Whether to compute SFG loss only on positive samples
    """

    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        init_std=0.0001,
        seed=1024,
        task='binary',
        device='cpu',
        gpus=None,
        use_sfg=True,
        sfg_weight=0.1,
        sfg_hidden_units=(128, 64),
        sfg_dropout=0.1,
        sfg_positive_only=True,
        sfg_use_label_attention=True
    ):
        super(BaseModelSFG, self).__init__()
        
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.use_sfg = use_sfg
        self.sfg_weight = sfg_weight
        self.sfg_positive_only = sfg_positive_only

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.sfg_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task)
        
        # Extract feature information for SFG
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []
        
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)
        ) if dnn_feature_columns else []
        
        # Get embedding dimension
        if self.sparse_feature_columns:
            self.embedding_dim = self.sparse_feature_columns[0].embedding_dim
        else:
            self.embedding_dim = 8
        
        # Initialize SFG components
        if use_sfg:
            # Sparse feature vocab sizes
            sparse_feature_dims = {
                feat.name: feat.vocabulary_size 
                for feat in self.sparse_feature_columns
            }
            dense_feature_names = [feat.name for feat in self.dense_feature_columns]
            
            self.sfg_decoder = SFGDecoder(
                embedding_dim=self.embedding_dim,
                sparse_feature_dims=sparse_feature_dims,
                dense_feature_names=dense_feature_names,
                hidden_units=sfg_hidden_units,
                dropout_rate=sfg_dropout,
                use_label_aware_attention=sfg_use_label_attention,
                device=device
            )
            
            self.sfg_loss_fn = SFGLoss(
                sparse_feature_names=[feat.name for feat in self.sparse_feature_columns],
                dense_feature_names=dense_feature_names,
                positive_only=sfg_positive_only,
                device=device
            )
        else:
            self.sfg_decoder = None
            self.sfg_loss_fn = None
        
        self.to(device)

        # Parameters for callbacks
        self._is_graph_network = True
        self._ckpt_saved_epoch = False
        self.history = History()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        """
        Training with SFG loss support
        """
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights). '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
            
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # Configure callbacks
        callbacks = (callbacks or []) + [self.history]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            sfg_loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        # Forward pass with SFG
                        y_pred, sfg_info = self.forward_with_sfg(x, y)
                        y_pred = y_pred.squeeze()

                        optim.zero_grad()
                        
                        # Main prediction loss
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks
                            loss = sum([loss_func[i](y_pred[:, i], y[:, i], reduction='sum') 
                                       for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        
                        reg_loss = self.get_regularization_loss()

                        # SFG loss
                        if self.use_sfg and sfg_info is not None:
                            sfg_loss = sfg_info['sfg_loss']
                            sfg_loss_epoch += sfg_loss.item()
                        else:
                            sfg_loss = torch.tensor(0.0, device=self.device)

                        total_loss = loss + reg_loss + self.aux_loss + self.sfg_weight * sfg_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), 
                                    y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            if self.use_sfg:
                epoch_logs["sfg_loss"] = sfg_loss_epoch / sample_num
                
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                    
            # Verbose output
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])
                
                if self.use_sfg:
                    eval_str += " - sfg_loss: {0: .4f}".format(epoch_logs["sfg_loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
                
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def forward_with_sfg(
        self, 
        X: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass that returns both predictions and SFG information
        
        This should be overridden by child classes to provide actual forward logic.
        """
        raise NotImplementedError("Subclass must implement forward_with_sfg")

    def forward(self, X):
        """Standard forward pass (calls forward_with_sfg)"""
        y_pred, _ = self.forward_with_sfg(X, None)
        return y_pred

    def compute_sfg_loss(
        self,
        X: torch.Tensor,
        sparse_embedding_list: List[torch.Tensor],
        dense_value_list: List[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute SFG reconstruction loss
        
        Args:
            X: Input tensor
            sparse_embedding_list: List of sparse embeddings
            dense_value_list: List of dense values
            labels: Target labels
            
        Returns:
            sfg_loss: The SFG loss tensor
            sfg_info: Dictionary with additional info
        """
        if not self.use_sfg or self.sfg_decoder is None:
            return torch.tensor(0.0, device=self.device), {}
        
        # Get reconstruction from decoder
        sparse_logits, dense_preds = self.sfg_decoder(
            sparse_embedding_list, 
            dense_value_list,
            labels
        )
        
        # Get original feature values as targets
        sparse_targets = {}
        for feat in self.sparse_feature_columns:
            feat_idx = self.feature_index[feat.name]
            sparse_targets[feat.name] = X[:, feat_idx[0]:feat_idx[1]].squeeze(-1)
        
        # Get dense targets
        dense_targets_list = []
        for feat in self.dense_feature_columns:
            feat_idx = self.feature_index[feat.name]
            dense_targets_list.append(X[:, feat_idx[0]:feat_idx[1]])
        
        if dense_targets_list:
            dense_targets = torch.cat(dense_targets_list, dim=-1)
        else:
            dense_targets = torch.zeros(X.shape[0], 0, device=self.device)
        
        # Compute loss
        sfg_loss, loss_dict = self.sfg_loss_fn(
            sparse_logits,
            dense_preds,
            sparse_targets,
            dense_targets,
            labels
        )
        
        return sfg_loss, {'sfg_loss': sfg_loss, 'sfg_loss_dict': loss_dict}

    def evaluate(self, x, y, batch_size=256):
        """Evaluate the model"""
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """Make predictions"""
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        """Extract embeddings and dense values from feature columns"""
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        """Compute the input dimension"""
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        """Add regularization weight"""
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        """Get regularization loss"""
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)
        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        """Add auxiliary loss"""
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None):
        """Compile the model"""
        self.metrics_names = ["loss"]
        if self.use_sfg:
            self.metrics_names.append("sfg_loss")
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        """Get optimizer"""
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        """Get loss function"""
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_single) for loss_single in loss]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        """Get single loss function"""
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        """Log loss computation"""
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        """Accuracy score"""
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        """Get metrics"""
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        """For EarlyStopping in tf1.15"""
        return None

    @property
    def embedding_size(self):
        """Get embedding size"""
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

