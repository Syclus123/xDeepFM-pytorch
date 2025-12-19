#!/bin/bash

# ============================================
# Full mode
# ============================================
# Recommended configuration:
#   - optimizer: adam (better than adagrad)
#   - learning_rate: 0.001 (default, if the first epoch is the best, can be reduced to 0.0001)
#   - l2_reg_dnn: 1e-5 (L2 regularization for DNN layers)
#   - dnn_dropout: 0.1~0.3 (to prevent overfitting)

python xdftrain.py \
  --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
  --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
  --mode eval \
  --out_dir ./outputs_eval \
  --device cuda:0 \
  --epochs 50 \
  --batch_size 4096 \
  --optimizer adam \
  --learning_rate 0.001 \
  --l2_reg_embedding 1e-5 \
  --l2_reg_dnn 1e-5 \
  --dnn_dropout 0.1 \
  --patience 3 \
  --verbose 1

# ============================================
# If the first epoch is the best, and the performance decreases afterwards (overfitting), try:
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
#   --learning_rate 0.0005 \        # reduce learning rate
#   --l2_reg_embedding 1e-4 \       # increase regularization
#   --l2_reg_dnn 1e-4 \             # increase regularization
#   --dnn_dropout 0.2 \             # increase dropout
#   --patience 5 \
#   --verbose 1

# ============================================
# Final mode - Train with all data (no validation, for final model)
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
