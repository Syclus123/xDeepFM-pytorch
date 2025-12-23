#!/bin/bash

# ============================================
# Eval mode - 完整数据集
# ============================================
#   - optimizer: adam 
#   - learning_rate: 0.001 (0.0001)
#   - l2_reg_dnn: 1e-5 (DNN层正则化)
#   - dnn_dropout: 0.1~0.3
#   --use_early_stopping \

python xdftrain.py \
  --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
  --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
  --mode eval \
  --out_dir ./outputs_eval_no_early_stopping \
  --device cuda:0 \
  --epochs 50 \
  --batch_size 4096 \
  --optimizer adam \
  --learning_rate 0.001 \
  --l2_reg_embedding 1e-5 \
  --l2_reg_dnn 1e-5 \
  --dnn_dropout 0.1 \
  --verbose 1

# ============================================
# 过拟合tuned
# ============================================
# python xdftrain.py \
#   --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
#   --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
#   --mode eval \
#   --out_dir ./outputs_eval_tuned \
#   --device cuda:0 \
#   --epochs 20 \
#   --batch_size 4096 \
#   --optimizer adam \
#   --learning_rate 0.0005 \        
#   --l2_reg_embedding 1e-4 \       
#   --l2_reg_dnn 1e-4 \             
#   --dnn_dropout 0.2 \             
#   --use_early_stopping \          
#   --patience 5 \
#   --verbose 1

# ============================================
# Final mode
# ============================================
# python xdftrain.py \
#   --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
#   --mode final \
#   --out_dir ./outputs_final \
#   --device cuda:0 \
#   --epochs 20 \
#   --batch_size 4096 \
#   --optimizer adam \
#   --learning_rate 0.001 \
#   --verbose 1
