# evaluate.py
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 用于对比不同xDeepFM变体的性能
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import xDeepFM, xDeepFMAttention, xDeepFMAttentionV2
from deepctr.xdeepfm_pro import xDeepFMPro, xDeepFMProLight


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_criteo_like(path: str) -> pd.DataFrame:
    """读取Criteo格式的数据文件"""
    expected = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip()
        data_line = f.readline().strip()
    
    header_sep = "\t" if ("\t" in header_line and "," not in header_line) else ","
    data_sep = "\t" if "\t" in data_line else ","
    
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

    df = pd.read_csv(path, sep=sep, header=None, names=expected, engine="python")
    return df


@dataclass
class SafeLabelEncoder:
    """安全的标签编码器，未知类别映射为0"""
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


def prepare_features(
    df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    fit_df: Optional[pd.DataFrame] = None,
    encoders: Optional[Dict[str, SafeLabelEncoder]] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, Dict[str, SafeLabelEncoder], MinMaxScaler]:
    """特征预处理"""
    df = df.copy()

    df[sparse_features] = df[sparse_features].fillna("-1")
    
    # 对数值特征进行清洗
    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)

    if fit_df is not None:
        fit_df = fit_df.copy()
        fit_df[sparse_features] = fit_df[sparse_features].fillna("-1")
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

    for feat in dense_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
    
    df[dense_features] = scaler.transform(df[dense_features].astype("float32"))

    for feat in sparse_features:
        df[feat] = df[feat].astype("int64")
    for feat in dense_features:
        df[feat] = df[feat].astype("float32")

    return df, encoders, scaler


def build_model_input(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, np.ndarray]:
    return {name: df[name].values for name in feature_names}


def build_feature_columns(
    df_for_vocab: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    embedding_dim: int = 10,
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


# ---------------------------
# 模型构建函数
# ---------------------------

def build_xdeepfm(linear_cols, dnn_cols, device):
    """构建原始xDeepFM模型"""
    model = xDeepFM(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task="binary",
        device=device,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
    return model


def build_xdeepfm_pro(linear_cols, dnn_cols, device, use_sfg=True, sfg_weight=0.1):
    """构建xDeepFM Pro模型"""
    model = xDeepFMPro(
        linear_feature_columns=linear_cols,
        dnn_feature_columns=dnn_cols,
        task="binary",
        device=device,
        use_sfg=use_sfg,
        sfg_weight=sfg_weight,
        sfg_positive_only=True,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
    return model


def build_xdeepfm_attn(linear_cols, dnn_cols, device, version="v1", 
                       cin_num_heads=4, cin_num_attn_layers=1):
    """构建xDeepFM Attention模型"""
    if version == "v2":
        model = xDeepFMAttentionV2(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task="binary",
            device=device,
            cin_num_heads=cin_num_heads,
            cin_num_attn_layers=cin_num_attn_layers,
        )
    else:
        model = xDeepFMAttention(
            linear_feature_columns=linear_cols,
            dnn_feature_columns=dnn_cols,
            task="binary",
            device=device,
            cin_num_heads=cin_num_heads,
        )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])
    return model


# ---------------------------
# 评估函数
# ---------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """计算各种评估指标"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {
        "AUC": roc_auc_score(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "Recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "F1": f1_score(y_true, y_pred_binary, zero_division=0),
    }
    
    return metrics


def evaluate_model(
    model,
    model_name: str,
    eval_x: Dict[str, np.ndarray],
    y_eval: np.ndarray,
    batch_size: int = 4096,
) -> Dict:
    """评估单个模型"""
    print(f"\n[INFO] 评估模型: {model_name}")
    
    # 预测
    y_pred = model.predict(eval_x, batch_size=batch_size)
    y_pred = y_pred.flatten()
    y_true = y_eval.flatten()
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    
    print(f"  - AUC: {metrics['AUC']:.6f}")
    print(f"  - LogLoss: {metrics['LogLoss']:.6f}")
    print(f"  - Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  - Precision: {metrics['Precision']:.4f}")
    print(f"  - Recall: {metrics['Recall']:.4f}")
    print(f"  - F1: {metrics['F1']:.4f}")
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "predictions": y_pred,
    }


def run_comparison(args):
    """运行模型对比实验"""
    set_seed(args.seed)
    
    # 定义特征
    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"
    
    # 加载数据
    print(f"\n[INFO] 加载评估数据: {args.eval_path}")
    eval_df = read_criteo_like(args.eval_path)
    print(f"[INFO] 评估数据形状: {eval_df.shape}")
    
    # 处理标签
    eval_df["label"] = pd.to_numeric(eval_df["label"], errors="coerce")
    eval_df["label"] = eval_df["label"].fillna(0).astype("float32")
    
    pos_ratio = (eval_df["label"] == 1).sum() / len(eval_df)
    print(f"[INFO] 正样本比例: {pos_ratio:.4f}")
    
    # 重要：先用全部数据构建特征列和预处理器（保持词表一致）
    print(f"[INFO] 使用全部数据构建特征列...")
    full_df_processed, encoders, scaler = prepare_features(
        eval_df, sparse_features, dense_features, fit_df=eval_df
    )
    
    # 构建特征列（使用全部数据的词表大小）
    linear_cols, dnn_cols, feature_names = build_feature_columns(
        full_df_processed, sparse_features, dense_features, 
        embedding_dim=args.embedding_dim
    )
    
    # 采样（在预处理之后进行，确保词表大小一致）
    if args.sample_ratio < 1.0:
        sample_size = int(len(full_df_processed) * args.sample_ratio)
        sample_indices = full_df_processed.sample(n=sample_size, random_state=args.seed).index
        eval_df_processed = full_df_processed.loc[sample_indices].copy()
        print(f"[INFO] 采样后评估数据量: {len(eval_df_processed)}")
    else:
        eval_df_processed = full_df_processed
    
    # 构建输入
    eval_x = build_model_input(eval_df_processed, feature_names)
    y_eval = eval_df_processed[[target]].values
    
    # 设备
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，使用CPU")
        device = "cpu"
    
    # 存储结果
    results = []
    
    # 解析模型配置
    model_configs = parse_model_configs(args.models)
    
    for config in model_configs:
        model_type = config["type"]
        model_path = config["path"]
        model_name = config.get("name", os.path.basename(model_path))
        
        print(f"\n{'='*60}")
        print(f"[INFO] 加载模型: {model_name}")
        print(f"[INFO] 模型类型: {model_type}")
        print(f"[INFO] 模型路径: {model_path}")
        
        try:
            # 构建模型
            if model_type == "xdeepfm":
                model = build_xdeepfm(linear_cols, dnn_cols, device)
            elif model_type == "xdeepfm_pro":
                model = build_xdeepfm_pro(
                    linear_cols, dnn_cols, device,
                    use_sfg=config.get("use_sfg", True),
                    sfg_weight=config.get("sfg_weight", 0.1)
                )
            elif model_type == "xdeepfm_attn":
                model = build_xdeepfm_attn(
                    linear_cols, dnn_cols, device,
                    version=config.get("version", "v1"),
                    cin_num_heads=config.get("cin_num_heads", 4),
                    cin_num_attn_layers=config.get("cin_num_attn_layers", 1)
                )
            else:
                print(f"[ERROR] 未知模型类型: {model_type}")
                continue
            
            # 加载权重
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"[INFO] 成功加载权重")
            else:
                print(f"[WARN] 模型文件不存在: {model_path}")
                continue
            
            # 评估
            result = evaluate_model(model, model_name, eval_x, y_eval, args.batch_size)
            results.append(result)
            
        except Exception as e:
            print(f"[ERROR] 评估模型失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if results:
        generate_comparison_report(results, args.out_dir)
    
    return results


def parse_model_configs(models_str: str) -> List[Dict]:
    """
    解析模型配置字符串
    格式: "type:path:name,type:path:name,..."
    或者: "type:path,type:path,..."
    
    示例:
    "xdeepfm_attn:/path/to/model.pth:Attention_v1"
    "xdeepfm_pro:/path/to/model.pth"
    """
    configs = []
    for model_str in models_str.split(","):
        parts = model_str.strip().split(":")
        if len(parts) >= 2:
            config = {
                "type": parts[0].strip(),
                "path": parts[1].strip(),
            }
            if len(parts) >= 3:
                config["name"] = parts[2].strip()
            else:
                config["name"] = f"{parts[0]}_{os.path.basename(parts[1])}"
            configs.append(config)
    return configs


def generate_comparison_report(results: List[Dict], out_dir: str):
    """生成对比报告"""
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("模型对比结果")
    print('='*60)
    
    # 准备表格数据
    table_data = []
    for r in results:
        row = [r["model_name"]]
        for metric in ["AUC", "LogLoss", "Accuracy", "Precision", "Recall", "F1"]:
            row.append(f"{r['metrics'][metric]:.6f}")
        table_data.append(row)
    
    headers = ["Model", "AUC", "LogLoss", "Accuracy", "Precision", "Recall", "F1"]
    
    # 打印表格
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 找出最佳模型
    best_auc_idx = np.argmax([r["metrics"]["AUC"] for r in results])
    best_logloss_idx = np.argmin([r["metrics"]["LogLoss"] for r in results])
    
    print(f"\n[最佳AUC] {results[best_auc_idx]['model_name']}: {results[best_auc_idx]['metrics']['AUC']:.6f}")
    print(f"[最佳LogLoss] {results[best_logloss_idx]['model_name']}: {results[best_logloss_idx]['metrics']['LogLoss']:.6f}")
    
    # 保存结果到JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "num_models": len(results),
        "results": [
            {
                "model_name": r["model_name"],
                "metrics": r["metrics"]
            }
            for r in results
        ],
        "best_auc_model": results[best_auc_idx]["model_name"],
        "best_logloss_model": results[best_logloss_idx]["model_name"],
    }
    
    report_path = os.path.join(out_dir, f"comparison_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 对比报告已保存: {report_path}")
    
    # 保存CSV格式
    csv_path = os.path.join(out_dir, f"comparison_results_{timestamp}.csv")
    df = pd.DataFrame(table_data, columns=headers)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV结果已保存: {csv_path}")


def run_single_eval(args):
    """评估单个模型"""
    set_seed(args.seed)
    
    sparse_features = [f"C{i}" for i in range(1, 27)]
    dense_features = [f"I{i}" for i in range(1, 14)]
    target = "label"
    
    # 加载数据
    print(f"\n[INFO] 加载评估数据: {args.eval_path}")
    eval_df = read_criteo_like(args.eval_path)
    print(f"[INFO] 评估数据形状: {eval_df.shape}")
    
    eval_df["label"] = pd.to_numeric(eval_df["label"], errors="coerce")
    eval_df["label"] = eval_df["label"].fillna(0).astype("float32")
    
    pos_ratio = (eval_df["label"] == 1).sum() / len(eval_df)
    print(f"[INFO] 正样本比例: {pos_ratio:.4f}")
    
    # 重要：先用全部数据构建特征列和预处理器（保持词表一致）
    print(f"[INFO] 使用全部数据构建特征列...")
    full_df_processed, encoders, scaler = prepare_features(
        eval_df, sparse_features, dense_features, fit_df=eval_df
    )
    
    # 构建特征列（使用全部数据的词表大小）
    linear_cols, dnn_cols, feature_names = build_feature_columns(
        full_df_processed, sparse_features, dense_features,
        embedding_dim=args.embedding_dim
    )
    
    # 采样（在预处理之后进行，确保词表大小一致）
    if args.sample_ratio < 1.0:
        sample_size = int(len(full_df_processed) * args.sample_ratio)
        sample_indices = full_df_processed.sample(n=sample_size, random_state=args.seed).index
        eval_df_processed = full_df_processed.loc[sample_indices].copy()
        print(f"[INFO] 采样后评估数据量: {len(eval_df_processed)}")
    else:
        eval_df_processed = full_df_processed
    
    eval_x = build_model_input(eval_df_processed, feature_names)
    y_eval = eval_df_processed[[target]].values
    
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，使用CPU")
        device = "cpu"
    
    # 构建模型
    print(f"\n[INFO] 模型类型: {args.model_type}")
    print(f"[INFO] 模型路径: {args.model_path}")
    
    if args.model_type == "xdeepfm":
        model = build_xdeepfm(linear_cols, dnn_cols, device)
    elif args.model_type == "xdeepfm_pro":
        model = build_xdeepfm_pro(linear_cols, dnn_cols, device)
    elif args.model_type == "xdeepfm_attn":
        model = build_xdeepfm_attn(
            linear_cols, dnn_cols, device,
            version=args.attn_version,
            cin_num_heads=args.cin_num_heads,
            cin_num_attn_layers=args.cin_num_attn_layers
        )
    else:
        raise ValueError(f"未知模型类型: {args.model_type}")
    
    # 加载权重
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] 成功加载权重")
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 评估
    result = evaluate_model(model, args.model_type, eval_x, y_eval, args.batch_size)
    
    # 保存结果
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(args.out_dir, f"eval_result_{timestamp}.json")
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_type": args.model_type,
                "model_path": args.model_path,
                "eval_path": args.eval_path,
                "sample_ratio": args.sample_ratio,
                "num_samples": len(eval_df),
                "metrics": result["metrics"],
                "timestamp": timestamp,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] 评估结果已保存: {result_path}")
    
    return result


def parse_args():
    p = argparse.ArgumentParser(description="模型评估与对比实验")
    
    subparsers = p.add_subparsers(dest="command", help="可用命令")
    
    # 单模型评估
    single_parser = subparsers.add_parser("single", help="评估单个模型")
    single_parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    single_parser.add_argument("--model_type", type=str, required=True,
                               choices=["xdeepfm", "xdeepfm_pro", "xdeepfm_attn"],
                               help="模型类型")
    single_parser.add_argument("--eval_path", type=str, required=True, help="评估数据路径")
    single_parser.add_argument("--out_dir", type=str, default="./eval_results", help="输出目录")
    single_parser.add_argument("--device", type=str, default="cuda:0")
    single_parser.add_argument("--batch_size", type=int, default=4096)
    single_parser.add_argument("--embedding_dim", type=int, default=10)
    single_parser.add_argument("--sample_ratio", type=float, default=1.0, help="采样比例 (0-1)")
    single_parser.add_argument("--seed", type=int, default=2025)
    # Attention模型参数
    single_parser.add_argument("--attn_version", type=str, default="v1", choices=["v1", "v2"])
    single_parser.add_argument("--cin_num_heads", type=int, default=4)
    single_parser.add_argument("--cin_num_attn_layers", type=int, default=1)
    
    # 多模型对比
    compare_parser = subparsers.add_parser("compare", help="对比多个模型")
    compare_parser.add_argument("--models", type=str, required=True,
                                help="模型配置，格式: type:path:name,type:path:name,...")
    compare_parser.add_argument("--eval_path", type=str, required=True, help="评估数据路径")
    compare_parser.add_argument("--out_dir", type=str, default="./eval_results", help="输出目录")
    compare_parser.add_argument("--device", type=str, default="cuda:0")
    compare_parser.add_argument("--batch_size", type=int, default=4096)
    compare_parser.add_argument("--embedding_dim", type=int, default=10)
    compare_parser.add_argument("--sample_ratio", type=float, default=1.0, help="采样比例 (0-1)")
    compare_parser.add_argument("--seed", type=int, default=2025)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.command == "single":
        run_single_eval(args)
    elif args.command == "compare":
        run_comparison(args)
    else:
        print("请指定命令: single 或 compare")
        print("使用 --help 查看帮助")

