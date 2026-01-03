# v1（与原始CIN输出维度兼容）
python xdftrain_attn.py \
  --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
  --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
  --mode eval \
  --model_version v1 \
  --cin_num_heads 2 \
  --dnn_dropout 0.5 \
  --l2_reg_embedding 1e-3 \
  --l2_reg_dnn 1e-3
#  v2版本（保留更多信息）
# python xdftrain_attn.py \
#   --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
#   --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
#   --mode eval \
#   --model_version v2 \
#   --cin_num_heads 4 \
#   --cin_num_attn_layers 2