# train_xdeepfm_full.py
# -*- coding: utf-8 -*-

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

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------
# Utils
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
        """Set the model reference"""
        self.model = model
    
    def set_params(self, params):
        """Set training parameters"""
        self.params = params
    
    def _implements_train_batch_hooks(self):
        """Required by Keras callback interface"""
        return False
    
    def _implements_test_batch_hooks(self):
        """Required by Keras callback interface"""
        return False
    
    def _implements_predict_batch_hooks(self):
        """Required by Keras callback interface"""
        return False
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
        
    def on_epoch_end(self, epoch, logs=None):
        """epoch end"""
        logs = logs or {}
        self.epoch = epoch
        
        for key, value in logs.items():
            if value is not None:
                try:
                    if key.startswith('val_'):
                        metric_name = key.replace('val_', '')
                        self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
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
    """
    attention: header and data may use different separators!
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline()
        data_line = f.readline()  # data 2 line
    
    if data_line:
        if "\t" in data_line:
            return "\t"
        elif "," in data_line:
            return ","
    
    # fallback: header line
    return "\t" if ("\t" in header_line and "," not in header_line) else ","


def read_criteo_like(path: str) -> pd.DataFrame:
    """
    Robust reader for:
      - DeepCTR-Torch demo file criteo_sample.txt (tab-separated, with header)
      - Criteo-like train.txt (tab-separated, often no header)
      - your train-labeled.txt (header comma-separated, data tab-separated)
    Expected columns: label, I1..I13, C1..C26
    """
    expected = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        data_line = f.readline().strip()
    
    # header separator
    header_sep = "\t" if ("\t" in header_line and "," not in header_line) else ","
    # data separator
    data_sep = "\t" if "\t" in data_line else ","
    
    # if header and data separator are different, need special treatment
    if header_sep != data_sep:
        print(f"[WARN] Header uses '{repr(header_sep)}' but data uses '{repr(data_sep)}'. Fixing...")
        header_cols = header_line.split(header_sep)
        if all(c in header_cols for c in expected):
            df = pd.read_csv(path, sep=data_sep, skiprows=1, header=None, 
                           names=expected, engine="python")
            return df
    
    sep = data_sep
    
    df = pd.read_csv(path, sep=sep, engine="python")
    
    if all(c in df.columns for c in expected):
        return df

    # If header missing or wrong, re-read without header and assign names
    df = pd.read_csv(path, sep=sep, header=None, names=expected, engine="python")
    return df


@dataclass
class SafeLabelEncoder:
    """
    Fit on train only.
    Unknown categories -> 0.
    Known categories mapped to 1..N.
    """
    mapping: Dict[str, int]
    unk: int = 0

    @staticmethod
    def fit(series: pd.Series) -> "SafeLabelEncoder":
        # convert to string to be safe (Criteo categorical often strings/hashed ints)
        uniq = pd.Series(series.astype(str).unique())
        # reserve 0 for unknown
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
    """
    If fit_df is provided -> fit encoders/scaler on fit_df, then transform df.
    Else -> use given encoders/scaler to transform df.
    """
    df = df.copy()

    # fill missing
    df[sparse_features] = df[sparse_features].fillna("-1")
    df[dense_features] = df[dense_features].fillna(0)

    if fit_df is not None:
        fit_df = fit_df.copy()
        fit_df[sparse_features] = fit_df[sparse_features].fillna("-1")
        fit_df[dense_features] = fit_df[dense_features].fillna(0)

        encoders = {}
        for feat in sparse_features:
            le = SafeLabelEncoder.fit(fit_df[feat])
            encoders[feat] = le

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(fit_df[dense_features].astype("float32"))

    assert encoders is not None and scaler is not None

    # transform
    for feat in sparse_features:
        df[feat] = encoders[feat].transform(df[feat])

    df[dense_features] = scaler.transform(df[dense_features].astype("float32"))

    # ensure dtypes
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
    # vocabulary_size should be max_id + 1 (because we map unknown->0)
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


def build_model(
    linear_feature_columns,
    dnn_feature_columns,
    device: str,
    l2_reg_embedding: float,
    l2_reg_dnn: float = 0.0,
    dnn_dropout: float = 0.0,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
):
    model = xDeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task="binary",
        l2_reg_embedding=l2_reg_embedding,
        l2_reg_dnn=l2_reg_dnn,
        dnn_dropout=dnn_dropout,
        device=device,
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["binary_crossentropy", "auc"],
    )
    # 手动设置学习率
    for param_group in model.optim.param_groups:
        param_group['lr'] = learning_rate
    return model


# ---------------------------
# Train modes
# ---------------------------

def read_criteo_test(path: str, sparse_features: List[str], dense_features: List[str]) -> pd.DataFrame:
    """
    Read Criteo test.txt (no header, no label column, 39 features only).
    Format: tab-separated, columns are I1..I13, C1..C26 (no label)
    """
    feature_cols = dense_features + sparse_features  # I1-I13, then C1-C26
    df = pd.read_csv(path, sep="\t", header=None, names=feature_cols, engine="python")
    return df


def run_eval(args):
    """
    Eval mode: 使用完整数据集训练，类似官方main.py的方式
    - train_file (data_path): 用于训练
    - eval_file (eval_path): 用于验证（每个epoch后评估）
    - test_file (test_path): 可选，用于推理预测（无标签）
    """
    set_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.out_dir, f"tensorboard_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] command: tensorboard --logdir={tb_log_dir}")
    
    start_time = time.time()

    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"

    # 读取训练数据
    print(f"[INFO] Loading train data from: {args.data_path}")
    train_df = read_criteo_like(args.data_path)
    print(f"[DEBUG] Train data shape: {train_df.shape}")
    print(f"[DEBUG] Columns: {list(train_df.columns[:5])} ... (total {len(train_df.columns)} cols)")
    
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce")
    nan_count = train_df["label"].isna().sum()
    if nan_count > 0:
        print(f"[WARN] Found {nan_count} NaN labels, filling with 0")
    train_df["label"] = train_df["label"].fillna(0).astype("float32")
    
    label_counts = train_df["label"].value_counts().sort_index()
    print(f"[DEBUG] Train label distribution:\n{label_counts}")
    train_pos_ratio = (train_df["label"] == 1).sum() / len(train_df)
    print(f"[DEBUG] Train positive ratio: {train_pos_ratio:.4f} ({(train_df['label'] == 1).sum()} / {len(train_df)})")
    
    if train_pos_ratio == 0.0:
        print("[ERROR] All labels are 0! This will cause loss=0. Check data file format!")
    elif train_pos_ratio == 1.0:
        print("[ERROR] All labels are 1! Check data file format!")

    # 读取验证数据
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
        # 如果没有提供eval_path，从train_df中划分
        print(f"[INFO] No eval_path provided, splitting {args.val_size*100:.0f}% from train data for validation")
        train_df, eval_df = train_test_split(
            train_df, test_size=args.val_size, random_state=args.seed,
            stratify=train_df[target] if args.stratify else None
        )
        print(f"[DEBUG] After split - Train: {len(train_df)}, Eval: {len(eval_df)}")

    # 读取测试数据（无标签，用于推理）
    test_df = None
    if args.test_path:
        print(f"[INFO] Loading test data from: {args.test_path}")
        test_df = read_criteo_test(args.test_path, sparse_features, dense_features)
        print(f"[DEBUG] Test data shape (no label): {test_df.shape}")

    # 合并train和eval用于fit encoders/scaler（保持和官方一致）
    all_labeled_df = pd.concat([train_df, eval_df], axis=0, ignore_index=True)
    print(f"[INFO] Total labeled samples for fitting encoders: {len(all_labeled_df)}")

    # fit on all labeled data
    all_labeled_df, encoders, scaler = prepare_features(
        all_labeled_df, sparse_features, dense_features, fit_df=all_labeled_df
    )
    
    # 分割回train和eval
    train_df_processed = all_labeled_df.iloc[:len(train_df)].copy()
    eval_df_processed = all_labeled_df.iloc[len(train_df):].copy()

    # 处理测试数据（如果有）
    test_df_processed = None
    if test_df is not None:
        test_df_processed, _, _ = prepare_features(
            test_df, sparse_features, dense_features, fit_df=None, encoders=encoders, scaler=scaler
        )

    # feature columns (vocab from all labeled data)
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

    model = build_model(
        linear_cols, dnn_cols, 
        device=device, 
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_dnn=args.l2_reg_dnn,
        dnn_dropout=args.dnn_dropout,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
    )

    ckpt_path = os.path.join(args.out_dir, "xdeepfm_best.pth")

    callbacks = [
        TensorBoardCallback(log_dir=tb_log_dir),
        EarlyStopping(monitor="val_auc", patience=args.patience, mode="max", verbose=1),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print(f"\n[INFO] Starting training...")
    print(f"  - Train samples: {len(train_df_processed)}")
    print(f"  - Eval samples: {len(eval_df_processed)}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Device: {device}")

    history = model.fit(
        train_x, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        validation_data=(eval_x, y_eval),
        shuffle=True,
        callbacks=callbacks,
    )

    # load best weights and evaluate on eval set
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

    # 对测试数据进行推理（如果有）
    test_predictions = None
    if test_df_processed is not None:
        print(f"\n[INFO] Running inference on test data ({len(test_df_processed)} samples)...")
        test_predictions = model.predict(
            build_model_input(test_df_processed, feature_names), 
            batch_size=args.pred_batch_size
        )
        # 保存预测结果
        test_pred_path = os.path.join(args.out_dir, "test_predictions.csv")
        pd.DataFrame({"predicted_ctr": test_predictions.flatten()}).to_csv(test_pred_path, index=False)
        print(f"[INFO] Test predictions saved to: {test_pred_path}")

    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    tb_writer.add_scalar('Eval/LogLoss', eval_logloss, 0)
    tb_writer.add_scalar('Eval/AUC', eval_auc, 0)
    tb_writer.add_text('Model/Config', str(vars(args)), 0)
    tb_writer.close()

    # save preprocessors for inference
    joblib.dump({"encoders": encoders, "scaler": scaler,
                 "sparse_features": sparse_features, "dense_features": dense_features,
                 "feature_names": feature_names},
                os.path.join(args.out_dir, "preprocess.joblib"))

    # save final(best) weights
    torch.save(model.state_dict(), os.path.join(args.out_dir, "xdeepfm_weights.pth"))

    # save training history
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)
    
    training_log = {
        "mode": "eval",
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
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
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
    
    print(f"\n[INFO] training completed! results saved:")
    print(f"  - model weights: {os.path.join(args.out_dir, 'xdeepfm_weights.pth')}")
    print(f"  - preprocessor: {os.path.join(args.out_dir, 'preprocess.joblib')}")
    print(f"  - training history: {os.path.join(args.out_dir, 'history.json')}")
    print(f"  - training log: {log_path}")
    print(f"  - TensorBoard log: {tb_log_dir}")
    if test_df_processed is not None:
        print(f"  - test predictions: {os.path.join(args.out_dir, 'test_predictions.csv')}")
    print(f"\n[INFO] view TensorBoard: tensorboard --logdir={tb_log_dir}")


def run_final(args):
    set_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(args.out_dir, f"tensorboard_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] command: tensorboard --logdir={tb_log_dir}")
    
    start_time = time.time()

    df = read_criteo_like(args.data_path)

    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"

    print(f"[DEBUG] Data shape: {df.shape}")
    print(f"[DEBUG] Columns: {list(df.columns[:5])} ... (total {len(df.columns)} cols)")
    
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    nan_count = df["label"].isna().sum()
    if nan_count > 0:
        print(f"[WARN] Found {nan_count} NaN labels, filling with 0")
    df["label"] = df["label"].fillna(0).astype("float32")
    
    label_counts = df["label"].value_counts().sort_index()
    print(f"[DEBUG] Label distribution:\n{label_counts}")
    pos_ratio = (df["label"] == 1).sum() / len(df)
    print(f"[DEBUG] Positive ratio: {pos_ratio:.4f} ({(df['label'] == 1).sum()} / {len(df)})")
    
    if pos_ratio == 0.0:
        print("[ERROR] All labels are 0! This will cause loss=0. Check data file format!")
    elif pos_ratio == 1.0:
        print("[ERROR] All labels are 1! This will cause training issues. Check data file format!")

    # fit preprocessors on FULL data
    df, encoders, scaler = prepare_features(
        df, sparse_features, dense_features, fit_df=df
    )

    # feature columns (vocab from FULL data)
    linear_cols, dnn_cols, feature_names = build_feature_columns(
        df, sparse_features, dense_features, embedding_dim=args.embedding_dim
    )

    x_full = build_model_input(df, feature_names)
    y_full = df[["label"]].values  # shape (N,1)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to cpu")
        device = "cpu"

    # attention: final mode does not use metrics, to avoid warnings/errors due to single-class batches
    model = xDeepFM(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task="binary",
        l2_reg_embedding=args.l2_reg_embedding,
        l2_reg_dnn=args.l2_reg_dnn,
        dnn_dropout=args.dnn_dropout,
        device=device,
    )
    model.compile(
        optimizer=args.optimizer,
        loss="binary_crossentropy",
        metrics=[],  # final mode does not use metrics, to avoid warnings/errors due to single-class batches
    )
    # 手动设置学习率
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

    # history = model.fit(
    #     x_full, y_full,
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,
    #     verbose=2,
    #     validation_split=0.0,
    #     shuffle=True,
    # )

    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n[Final] Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # 将配置信息记录到 TensorBoard
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    tb_writer.add_text('Model/Config', str(vars(args)), 0)
    tb_writer.close()

    # save preprocessors + weights
    joblib.dump({"encoders": encoders, "scaler": scaler,
                 "sparse_features": sparse_features, "dense_features": dense_features,
                 "feature_names": feature_names},
                os.path.join(args.out_dir, "preprocess.joblib"))
    torch.save(model.state_dict(), os.path.join(args.out_dir, "xdeepfm_full_weights.pth"))

    with open(os.path.join(args.out_dir, "history_full.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)
    
    # 保存完整的训练日志
    training_log = {
        "mode": "final",
        "timestamp": timestamp,
        "training_time_seconds": training_time,
        "data_info": {
            "data_path": args.data_path,
            "total_samples": len(df),
            "positive_ratio": float(pos_ratio),
        },
        "model_config": {
            "embedding_dim": args.embedding_dim,
            "l2_reg_embedding": args.l2_reg_embedding,
            "l2_reg_dnn": args.l2_reg_dnn,
            "dnn_dropout": args.dnn_dropout,
            "device": device,
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
        },
        "history": history.history,
        "tensorboard_log_dir": tb_log_dir,
    }
    
    log_path = os.path.join(args.out_dir, "training_log_full.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 训练完成！结果已保存:")
    print(f"  - 模型权重: {os.path.join(args.out_dir, 'xdeepfm_full_weights.pth')}")
    print(f"  - 预处理器: {os.path.join(args.out_dir, 'preprocess.joblib')}")
    print(f"  - 训练历史: {os.path.join(args.out_dir, 'history_full.json')}")
    print(f"  - 训练日志: {log_path}")
    print(f"  - TensorBoard日志: {tb_log_dir}")
    print(f"\n[INFO] 查看TensorBoard: tensorboard --logdir={tb_log_dir}")


# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to train data (train-labeled.txt)")
    p.add_argument("--eval_path", type=str, default=None, help="Path to eval data (optional, if not provided will split from train)")
    p.add_argument("--test_path", type=str, default=None, help="Path to test data for inference (test.txt, no label)")
    p.add_argument("--out_dir", type=str, default="./outputs_xdeepfm", help="Output directory")
    p.add_argument("--mode", type=str, choices=["eval", "final"], default="eval")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)

    p.add_argument("--embedding_dim", type=int, default=10)
    p.add_argument("--l2_reg_embedding", type=float, default=1e-5)
    p.add_argument("--l2_reg_dnn", type=float, default=1e-5, help="L2 regularization for DNN layers")
    p.add_argument("--dnn_dropout", type=float, default=0.0, help="Dropout rate for DNN layers")
    p.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adagrad", "sgd"], help="Optimizer")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--pred_batch_size", type=int, default=8192)

    # eval-only
    p.add_argument("--val_size", type=float, default=0.1, help="Validation split ratio (used when --eval_path not provided)")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--stratify", action="store_true", help="Stratified split by label")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
               help="0=silent, 1=progress bar(tqdm), 2=one line per epoch")


    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "eval":
        run_eval(args)
    else:
        run_final(args)
