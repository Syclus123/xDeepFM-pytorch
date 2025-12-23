# SFG训练
python xdftrain_pro.py \
    --data_path /root/xDeepFM/exdeepfm/datasets/train-labeled.txt \
    --test_path /root/xDeepFM/exdeepfm/datasets/test.txt \
    --val_size 0.1 \
    --batch_size 64 \
    --pred_batch_size 128 \
    --use_sfg \
    --sfg_weight 0.1 \
    --sfg_positive_only \
    --epochs 30 \
    --device cuda:1


# 禁用SFG
# python xdftrain_pro.py \
#     --data_path ./data/train.txt \
#     --no_sfg