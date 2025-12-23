# train_xdeepfm_pro.py
# -*- coding: utf-8 -*-
"""
xDeepFM Pro 训练脚本

使用监督特征生成（SFG）增强的xDeepFM模型进行训练。
SFG通过辅助解码器重建原始特征来解决嵌入坍塌问题。
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime

from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr.xdeepfm_pro import xDeepFMPro, xDeepFMProLight
from deepctr.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------
# Utils (复用原有工具函数)
# ---------------------------

class TensorBoardCallback:
    """
    tensorboard callback compatible with Keras callback interface
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_step = 0
        self.epoch = 0
        self.model = None
        self.params = None
    
    def set_model(self, model):
        self.model = model
    
    def set_params(self, params):
        self.params = params
    
    def _implements_train_batch_hooks(self):
        return False
    
    def _implements_test_batch_hooks(self):
        return False
    
    def _implements_predict_batch_hooks(self):
        return False
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        
        for key, value in logs.items():
            if value is not None:
                try:
                    if key.startswith('val_'):
                        metric_name = key.replace('val_', '')
                        self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
                    elif key == 'sfg_loss':
                        self.writer.add_scalar('Training/sfg_loss', value, epoch)
                    else:
                        metric_name = key
                        self.writer.add_scalar(f'Training/{metric_name}', value, epoch)
                except Exception as e:
                    print(f"[WARN] TensorBoard logging error for {key}: {e}")
        
        self.writer.flush()
    
    def on_train_begin(self, logs=None):
        pass
    
    def on_train_end(self, logs=None):
        try:
            self.writer.close()
        except Exception as e:
            print(f"[WARN] Error closing TensorBoard writer: {e}")


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
        data_line = f.readline()
    
    if data_line:
        if "\t" in data_line:
            return "\t"
        elif "," in data_line:
            return ","
    
    return "\t" if ("\t" in header_line and "," not in header_line) else ","


def read_criteo_like(path: str, require_label: bool = True) -> pd.DataFrame:
    """
    Robust reader for Criteo-like data files.
    Auto-detects whether the file has a label column based on number of columns.
    
    Args:
        path: Path to the data file
        require_label: If True and no label found, will raise an error
    """
    expected_with_label = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    expected_no_label = [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        data_line = f.readline().strip()
    
    header_sep = "\t" if ("\t" in header_line and "," not in header_line) else ","
    data_sep = "\t" if "\t" in data_line else ","
    
    # 检测数据列数
    num_cols = len(data_line.split(data_sep))
    print(f"[DEBUG] Detected {num_cols} columns in data file: {path}")
    
    # 根据列数判断是否有label
    if num_cols == 40:  # 有label: 1 + 13 + 26
        expected = expected_with_label
        has_label = True
    elif num_cols == 39:  # 无label: 13 + 26
        expected = expected_no_label
        has_label = False
        if require_label:
            print(f"[ERROR] File {path} has 39 columns (no label). Use --test_path for test data without labels.")
            raise ValueError(f"Eval data must have label column. File {path} has only 39 columns.")
    else:
        # 尝试从header判断
        header_cols = header_line.split(header_sep)
        if "label" in header_cols:
            expected = expected_with_label
            has_label = True
        else:
            expected = expected_no_label
            has_label = False
        print(f"[WARN] Unexpected column count: {num_cols}, has_label={has_label}")
    
    if header_sep != data_sep:
        print(f"[WARN] Header uses '{repr(header_sep)}' but data uses '{repr(data_sep)}'. Fixing...")
        header_cols = header_line.split(header_sep)
        if all(c in header_cols for c in expected):
            df = pd.read_csv(path, sep=data_sep, skiprows=1, header=None, 
                           names=expected, engine="python")
            if not has_label:
                df.insert(0, "label", 0)  # 添加dummy label
            return df
    
    sep = data_sep
    df = pd.read_csv(path, sep=sep, engine="python")
    
    if all(c in df.columns for c in expected):
        if not has_label and "label" not in df.columns:
            df.insert(0, "label", 0)
        return df

    df = pd.read_csv(path, sep=sep, header=None, names=expected, engine="python")
    if not has_label:
        df.insert(0, "label", 0)  # 添加dummy label
    return df


@dataclass
class SafeLabelEncoder:
    mapping: Dict[str, int]
    unk: int = 0

    @staticmethod
    def fit(series: pd.Series) -> "SafeLabelEncoder":
        uniq = pd.Series(series.astype(str).unique())
        mapping = {v: i + 1 for i, v in enumerate(uniq.tolist())}
        return SafeLabelEncoder(mapping=mapping, unk=0)

    def transform(self, series: pd.Series) -> np.ndarray:
        s = series.astype(str)
        return s.map(self.mapping).fillna(self.unk).astype("int64").values


def build_model_input(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, np.ndarray]:
    return {name: df[name].values for name in feature_names}


def prepare_features(
    df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    fit_df: Optional[pd.DataFrame] = None,
    encoders: Optional[Dict[str, SafeLabelEncoder]] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, Dict[str, SafeLabelEncoder], MinMaxScaler]:
    df = df.copy()

    df[sparse_features] = df[sparse_features].fillna("-1")
    
    # 对数值特征进行清洗：将非数值转换为0
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
    
    if fit_df is not None:
        fit_df = fit_df.copy()
        fit_df[sparse_features] = fit_df[sparse_features].fillna("-1")
        
        # 对数值特征进行清洗：将非数值转换为0
        for feat in dense_features:
            fit_df[feat] = pd.to_numeric(fit_df[feat], errors='coerce').fillna(0)

        encoders = {}
        for feat in sparse_features:
            le = SafeLabelEncoder.fit(fit_df[feat])
            encoders[feat] = le

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(fit_df[dense_features].astype("float32"))

    assert encoders is not None and scaler is not None

    for feat in sparse_features:
        df[feat] = encoders[feat].transform(df[feat])

    # 确保数值特征已经清洗过
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
    
    df[dense_features] = scaler.transform(df[dense_features].astype("float32"))

    for feat in sparse_features:
        df[feat] = df[feat].astype("int64")
    for feat in dense_features:
        df[feat] = df[feat].astype("float32")

    return df, encoders, scaler


def build_feature_columns(
    df_for_vocab: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    embedding_dim: int = 4,
):
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=int(df_for_vocab[feat].max()) + 1, embedding_dim=embedding_dim)
        for feat in sparse_features
    ] + [
        DenseFeat(feat, 1) for feat in dense_features
    ]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return linear_feature_columns, dnn_feature_columns, feature_names


def build_model_pro(
    linear_feature_columns,
    dnn_feature_columns,
    device: str,
    l2_reg_embedding: float,
    l2_reg_dnn: float = 0.0,
    dnn_dropout: float = 0.0,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    # SFG parameters
    use_sfg: bool = True,
    sfg_weight: float = 0.1,
    sfg_hidden_units: Tuple[int, ...] = (128, 64),
    sfg_dropout: float = 0.1,
    sfg_positive_only: bool = True,
    sfg_use_label_attention: bool = True,
    # AutoDis parameters
    use_autodis: bool = False,
    autodis_buckets: int = 16,
    # Model variant
    use_light_version: bool = False,
):
    """
    Build xDeepFM Pro model with SFG support
    """
    ModelClass = xDeepFMProLight if use_light_version else xDeepFMPro
    
    model = ModelClass(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task="binary",
        l2_reg_embedding=l2_reg_embedding,
        l2_reg_dnn=l2_reg_dnn,
        dnn_dropout=dnn_dropout,
        device=device,
        # SFG parameters
        use_sfg=use_sfg,
        sfg_weight=sfg_weight,
        sfg_hidden_units=sfg_hidden_units,
        sfg_dropout=sfg_dropout,
        sfg_positive_only=sfg_positive_only,
        sfg_use_label_attention=sfg_use_label_attention,
        # AutoDis parameters
        use_autodis=use_autodis,
        autodis_buckets=autodis_buckets,
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["binary_crossentropy", "auc"],
    )
    
    for param_group in model.optim.param_groups:
        param_group['lr'] = learning_rate
    
    return model


# ---------------------------
# Train modes
# ---------------------------

def read_criteo_test(path: str, sparse_features: List[str], dense_features: List[str]) -> pd.DataFrame:
    feature_cols = dense_features + sparse_features
    df = pd.read_csv(path, sep="\t", header=None, names=feature_cols, engine="python")
    return df


def run_eval(args):
    """
    Eval mode with SFG enhanced xDeepFM
    """
    set_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.out_dir, f"tensorboard_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] command: tensorboard --logdir={tb_log_dir}")
    print(f"[INFO] Using xDeepFM Pro with SFG (Supervised Feature Generation)")
    
    if args.use_sfg:
        print(f"[INFO] SFG enabled: weight={args.sfg_weight}, positive_only={args.sfg_positive_only}")
    else:
        print(f"[INFO] SFG disabled - running standard xDeepFM")
    
    start_time = time.time()

    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"

    print(f"[INFO] Loading train data from: {args.data_path}")
    train_df = read_criteo_like(args.data_path)
    print(f"[DEBUG] Train data shape: {train_df.shape}")
    
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce")
    nan_count = train_df["label"].isna().sum()
    if nan_count > 0:
        print(f"[WARN] Found {nan_count} NaN labels, filling with 0")
    train_df["label"] = train_df["label"].fillna(0).astype("float32")
    
    label_counts = train_df["label"].value_counts().sort_index()
    print(f"[DEBUG] Train label distribution:\n{label_counts}")
    train_pos_ratio = (train_df["label"] == 1).sum() / len(train_df)
    print(f"[DEBUG] Train positive ratio: {train_pos_ratio:.4f}")
    
    if train_pos_ratio == 0.0:
        print("[ERROR] All labels are 0! SFG needs positive samples to work.")
    elif train_pos_ratio == 1.0:
        print("[ERROR] All labels are 1! Check data file format!")

    eval_df = None
    if args.eval_path:
        print(f"[INFO] Loading eval data from: {args.eval_path}")
        eval_df = read_criteo_like(args.eval_path)
        print(f"[DEBUG] Eval data shape: {eval_df.shape}")
        
        eval_df["label"] = pd.to_numeric(eval_df["label"], errors="coerce")
        eval_df["label"] = eval_df["label"].fillna(0).astype("float32")
        
        eval_pos_ratio = (eval_df["label"] == 1).sum() / len(eval_df)
        print(f"[DEBUG] Eval positive ratio: {eval_pos_ratio:.4f}")
    else:
        print(f"[INFO] No eval_path provided, splitting {args.val_size*100:.0f}% from train data")
        train_df, eval_df = train_test_split(
            train_df, test_size=args.val_size, random_state=args.seed,
            stratify=train_df[target] if args.stratify else None
        )
        print(f"[DEBUG] After split - Train: {len(train_df)}, Eval: {len(eval_df)}")

    test_df = None
    if args.test_path:
        print(f"[INFO] Loading test data from: {args.test_path}")
        test_df = read_criteo_test(args.test_path, sparse_features, dense_features)
        print(f"[DEBUG] Test data shape (no label): {test_df.shape}")

    # Combine for encoder fitting
    all_labeled_df = pd.concat([train_df, eval_df], axis=0, ignore_index=True)
    print(f"[INFO] Total labeled samples for fitting encoders: {len(all_labeled_df)}")

    all_labeled_df, encoders, scaler = prepare_features(
        all_labeled_df, sparse_features, dense_features, fit_df=all_labeled_df
    )
    
    train_df_processed = all_labeled_df.iloc[:len(train_df)].copy()
    eval_df_processed = all_labeled_df.iloc[len(train_df):].copy()

    test_df_processed = None
    if test_df is not None:
        test_df_processed, _, _ = prepare_features(
            test_df, sparse_features, dense_features, fit_df=None, encoders=encoders, scaler=scaler
        )

    linear_cols, dnn_cols, feature_names = build_feature_columns(
        all_labeled_df, sparse_features, dense_features, embedding_dim=args.embedding_dim
    )

    train_x = build_model_input(train_df_processed, feature_names)
    eval_x = build_model_input(eval_df_processed, feature_names)

    y_train = train_df_processed[[target]].values
    y_eval = eval_df_processed[[target]].values

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to cpu")
        device = "cpu"

    # Build xDeepFM Pro model
    model = build_model_pro(
        linear_cols, dnn_cols, 
        device=device, 
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_dnn=args.l2_reg_dnn,
        dnn_dropout=args.dnn_dropout,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        # SFG parameters
        use_sfg=args.use_sfg,
        sfg_weight=args.sfg_weight,
        sfg_hidden_units=tuple(args.sfg_hidden_units),
        sfg_dropout=args.sfg_dropout,
        sfg_positive_only=args.sfg_positive_only,
        sfg_use_label_attention=args.sfg_use_label_attention,
        # AutoDis parameters
        use_autodis=args.use_autodis,
        autodis_buckets=args.autodis_buckets,
        # Light version
        use_light_version=args.use_light_version,
    )

    ckpt_path = os.path.join(args.out_dir, "xdeepfm_pro_best.pth")

    callbacks = [
        TensorBoardCallback(log_dir=tb_log_dir),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    
    if args.use_early_stopping:
        print(f"[INFO] Early stopping enabled with patience={args.patience}")
        callbacks.insert(1, EarlyStopping(monitor="val_auc", patience=args.patience, mode="max", verbose=1))
    else:
        print(f"[INFO] Early stopping disabled - will train for full {args.epochs} epochs")

    print(f"\n[INFO] Starting training with xDeepFM Pro...")
    print(f"  - Train samples: {len(train_df_processed)}")
    print(f"  - Eval samples: {len(eval_df_processed)}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Device: {device}")
    print(f"  - SFG enabled: {args.use_sfg}")
    if args.use_sfg:
        print(f"  - SFG weight: {args.sfg_weight}")
        print(f"  - SFG positive only: {args.sfg_positive_only}")

    history = model.fit(
        train_x, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=(eval_x, y_eval),
        shuffle=True,
        callbacks=callbacks,
    )

    # Load best weights and evaluate
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    eval_pred = model.predict(eval_x, batch_size=args.pred_batch_size)
    eval_logloss = log_loss(y_eval, eval_pred)
    eval_auc = roc_auc_score(y_eval, eval_pred)
    
    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n[Eval] eval LogLoss = {eval_logloss:.6f}")
    print(f"[Eval] eval AUC     = {eval_auc:.6f}")
    print(f"[Eval] Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Test predictions
    test_predictions = None
    if test_df_processed is not None:
        print(f"\n[INFO] Running inference on test data ({len(test_df_processed)} samples)...")
        test_predictions = model.predict(
            build_model_input(test_df_processed, feature_names), 
            batch_size=args.pred_batch_size
        )
        test_pred_path = os.path.join(args.out_dir, "test_predictions.csv")
        pd.DataFrame({"predicted_ctr": test_predictions.flatten()}).to_csv(test_pred_path, index=False)
        print(f"[INFO] Test predictions saved to: {test_pred_path}")

    # Additional TensorBoard logging
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    tb_writer.add_scalar('Eval/LogLoss', eval_logloss, 0)
    tb_writer.add_scalar('Eval/AUC', eval_auc, 0)
    tb_writer.add_text('Model/Config', str(vars(args)), 0)
    tb_writer.close()

    # Save artifacts
    joblib.dump({
        "encoders": encoders, 
        "scaler": scaler,
        "sparse_features": sparse_features, 
        "dense_features": dense_features,
        "feature_names": feature_names
    }, os.path.join(args.out_dir, "preprocess.joblib"))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "xdeepfm_pro_weights.pth"))

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)
    
    training_log = {
        "mode": "eval",
        "model": "xDeepFM_Pro",
        "timestamp": timestamp,
        "training_time_seconds": training_time,
        "data_info": {
            "train_path": args.data_path,
            "eval_path": args.eval_path,
            "test_path": args.test_path,
            "train_samples": len(train_df_processed),
            "eval_samples": len(eval_df_processed),
            "test_samples": len(test_df_processed) if test_df_processed is not None else 0,
            "train_positive_ratio": float(train_pos_ratio),
        },
        "model_config": {
            "embedding_dim": args.embedding_dim,
            "l2_reg_embedding": args.l2_reg_embedding,
            "l2_reg_dnn": args.l2_reg_dnn,
            "dnn_dropout": args.dnn_dropout,
            "device": device,
        },
        "sfg_config": {
            "use_sfg": args.use_sfg,
            "sfg_weight": args.sfg_weight,
            "sfg_hidden_units": args.sfg_hidden_units,
            "sfg_dropout": args.sfg_dropout,
            "sfg_positive_only": args.sfg_positive_only,
            "sfg_use_label_attention": args.sfg_use_label_attention,
        },
        "autodis_config": {
            "use_autodis": args.use_autodis,
            "autodis_buckets": args.autodis_buckets,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "use_early_stopping": args.use_early_stopping,
            "patience": args.patience,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
        },
        "results": {
            "eval_logloss": float(eval_logloss),
            "eval_auc": float(eval_auc),
            "best_val_auc": float(max(history.history.get('val_auc', [0]))) if history.history else None,
        },
        "history": history.history,
        "tensorboard_log_dir": tb_log_dir,
    }
    
    log_path = os.path.join(args.out_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 训练完成！结果已保存:")
    print(f"  - 模型权重: {os.path.join(args.out_dir, 'xdeepfm_pro_weights.pth')}")
    print(f"  - 预处理器: {os.path.join(args.out_dir, 'preprocess.joblib')}")
    print(f"  - 训练历史: {os.path.join(args.out_dir, 'history.json')}")
    print(f"  - 训练日志: {log_path}")
    print(f"  - TensorBoard日志: {tb_log_dir}")
    if test_df_processed is not None:
        print(f"  - 测试预测: {os.path.join(args.out_dir, 'test_predictions.csv')}")
    print(f"\n[INFO] 查看TensorBoard: tensorboard --logdir={tb_log_dir}")


def run_final(args):
    """
    Final training mode - train on all data
    """
    set_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.out_dir, f"tensorboard_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] Using xDeepFM Pro with SFG (Final training mode)")
    
    start_time = time.time()

    df = read_criteo_like(args.data_path)

    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"

    print(f"[DEBUG] Data shape: {df.shape}")
    
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    nan_count = df["label"].isna().sum()
    if nan_count > 0:
        print(f"[WARN] Found {nan_count} NaN labels, filling with 0")
    df["label"] = df["label"].fillna(0).astype("float32")
    
    pos_ratio = (df["label"] == 1).sum() / len(df)
    print(f"[DEBUG] Positive ratio: {pos_ratio:.4f}")

    df, encoders, scaler = prepare_features(
        df, sparse_features, dense_features, fit_df=df
    )

    linear_cols, dnn_cols, feature_names = build_feature_columns(
        df, sparse_features, dense_features, embedding_dim=args.embedding_dim
    )

    x_full = build_model_input(df, feature_names)
    y_full = df[["label"]].values

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to cpu")
        device = "cpu"

    # Build model
    model = build_model_pro(
        linear_cols, dnn_cols, 
        device=device, 
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_dnn=args.l2_reg_dnn,
        dnn_dropout=args.dnn_dropout,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        use_sfg=args.use_sfg,
        sfg_weight=args.sfg_weight,
        sfg_hidden_units=tuple(args.sfg_hidden_units),
        sfg_dropout=args.sfg_dropout,
        sfg_positive_only=args.sfg_positive_only,
        sfg_use_label_attention=args.sfg_use_label_attention,
        use_autodis=args.use_autodis,
        autodis_buckets=args.autodis_buckets,
        use_light_version=args.use_light_version,
    )
    
    # Override metrics for final training (no validation)
    model.compile(
        optimizer=args.optimizer,
        loss="binary_crossentropy",
        metrics=[],
    )
    for param_group in model.optim.param_groups:
        param_group['lr'] = args.learning_rate

    callbacks = [TensorBoardCallback(log_dir=tb_log_dir)]

    history = model.fit(
        x_full, y_full,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_split=0.0,
        shuffle=True,
        callbacks=callbacks,
    )

    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n[Final] Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Save artifacts
    joblib.dump({
        "encoders": encoders, 
        "scaler": scaler,
        "sparse_features": sparse_features, 
        "dense_features": dense_features,
        "feature_names": feature_names
    }, os.path.join(args.out_dir, "preprocess.joblib"))
    
    torch.save(model.state_dict(), os.path.join(args.out_dir, "xdeepfm_pro_full_weights.pth"))

    with open(os.path.join(args.out_dir, "history_full.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)
    
    training_log = {
        "mode": "final",
        "model": "xDeepFM_Pro",
        "timestamp": timestamp,
        "training_time_seconds": training_time,
        "data_info": {
            "data_path": args.data_path,
            "total_samples": len(df),
            "positive_ratio": float(pos_ratio),
        },
        "sfg_config": {
            "use_sfg": args.use_sfg,
            "sfg_weight": args.sfg_weight,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "history": history.history,
        "tensorboard_log_dir": tb_log_dir,
    }
    
    log_path = os.path.join(args.out_dir, "training_log_full.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 训练完成！结果已保存:")
    print(f"  - 模型权重: {os.path.join(args.out_dir, 'xdeepfm_pro_full_weights.pth')}")
    print(f"  - 预处理器: {os.path.join(args.out_dir, 'preprocess.joblib')}")
    print(f"  - TensorBoard日志: {tb_log_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train xDeepFM Pro with SFG")
    
    # Data paths
    p.add_argument("--data_path", type=str, required=True, help="Path to train data")
    p.add_argument("--eval_path", type=str, default=None, help="Path to eval data")
    p.add_argument("--test_path", type=str, default=None, help="Path to test data (no label)")
    p.add_argument("--out_dir", type=str, default="./outputs_xdeepfm_pro", help="Output directory")
    p.add_argument("--mode", type=str, choices=["eval", "final"], default="eval")

    # Device and seed
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    # Model hyperparameters
    p.add_argument("--embedding_dim", type=int, default=10)
    p.add_argument("--l2_reg_embedding", type=float, default=1e-5)
    p.add_argument("--l2_reg_dnn", type=float, default=1e-5)
    p.add_argument("--dnn_dropout", type=float, default=0.0)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adagrad", "sgd"])

    # Training parameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--pred_batch_size", type=int, default=4096)

    # Validation parameters
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--use_early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    # SFG parameters
    p.add_argument("--use_sfg", action="store_true", default=True, 
                   help="Enable Supervised Feature Generation (default: True)")
    p.add_argument("--no_sfg", action="store_false", dest="use_sfg",
                   help="Disable SFG")
    p.add_argument("--sfg_weight", type=float, default=0.1, 
                   help="Weight for SFG reconstruction loss")
    p.add_argument("--sfg_hidden_units", type=int, nargs='+', default=[128, 64],
                   help="Hidden units for SFG decoder")
    p.add_argument("--sfg_dropout", type=float, default=0.1,
                   help="Dropout rate for SFG decoder")
    p.add_argument("--sfg_positive_only", action="store_true", default=True,
                   help="Only compute SFG loss on positive samples (Y=1)")
    p.add_argument("--sfg_all_samples", action="store_false", dest="sfg_positive_only",
                   help="Compute SFG loss on all samples")
    p.add_argument("--sfg_use_label_attention", action="store_true", default=True,
                   help="Use label-aware attention in SFG decoder")

    # AutoDis parameters  
    p.add_argument("--use_autodis", action="store_true", default=False,
                   help="Enable AutoDis for dense features")
    p.add_argument("--autodis_buckets", type=int, default=16,
                   help="Number of AutoDis buckets")

    # Model variant
    p.add_argument("--use_light_version", action="store_true",
                   help="Use lighter xDeepFM Pro variant")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "eval":
        run_eval(args)
    else:
        run_final(args)

