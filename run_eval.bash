#!/bin/bash

# ============================================
# 模型评估与对比实验脚本
# ============================================

EVAL_DATA="/root/xDeepFM/exdeepfm/datasets/train-labeled.txt"

ATTN_MODEL="/root/xDeepFM-pytorch/outputs_xdeepfm_attn/xdeepfm_attn_best.pth"
PRO_MODEL="/root/xDeepFM-pytorch/outputs_xdeepfm_pro/xdeepfm_pro_best.pth"

OUT_DIR="/root/xDeepFM-pytorch/result"

# ============================================
# 1. 单模型评估示例
# ============================================

echo "=========================================="
echo "评估 xDeepFM Attention 模型"
echo "=========================================="

python evaluate.py single \
  --model_path ${ATTN_MODEL} \
  --model_type xdeepfm_attn \
  --eval_path ${EVAL_DATA} \
  --out_dir ${OUT_DIR} \
  --device cuda:0 \
  --batch_size 4096 \
  --embedding_dim 10 \
  --sample_ratio 0.1 \
  --attn_version v1 \
  --cin_num_heads 4

# ============================================
# 2. 多模型对比实验
# ============================================

echo ""
echo "=========================================="
echo "对比多个模型"
echo "=========================================="


python evaluate.py compare \
  --models "xdeepfm_attn:${ATTN_MODEL}:xDeepFM_Attention,xdeepfm_pro:${PRO_MODEL}:xDeepFM_Pro" \
  --eval_path ${EVAL_DATA} \
  --out_dir ${OUT_DIR} \
  --device cuda:0 \
  --batch_size 4096 \
  --embedding_dim 10 \
  --sample_ratio 0.1

echo ""
echo "=========================================="
echo "评估完成！结果保存在: ${OUT_DIR}"
echo "=========================================="

