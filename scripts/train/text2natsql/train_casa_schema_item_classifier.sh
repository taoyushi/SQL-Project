#!/bin/bash

set -e

echo "=== 优化的CASA多GPU训练脚本 ==="
echo "针对2x RTX 4090D (24GB each) 优化"

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')"

# 显存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,roundup_power2_divisions:4

# 检查GPU状态
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

echo ""
echo "开始训练..."

# 针对显存优化的参数
python -u casa_schema_item_classifier.py \
    --batch_size 8 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 30 \
    --patience 8 \
    --seed 42 \
    --save_path "./models/casa_multi_gpu_fixed" \
    --tensorboard_save_path "./tensorboard_log/casa_multi_gpu_fixed" \
    --train_filepath "./data/preprocessed_data/preprocessed_train_spider_natsql.json" \
    --dev_filepath "./data/preprocessed_data/preprocessed_dev_natsql.json" \
    --model_name_or_path "/root/bayes-gpfs-b83d8de711584bc9a9a79f29860df406/tys/RESDSQL/models/roberta-large-local" \
    --use_contents \
    --add_fk_info \
    --mode "train" \
    --fp16 \
    --use_multi_gpu \
    --debug

echo ""
echo "训练完成! 检查tensorboard日志: ./tensorboard_log/casa_multi_gpu_fixed"

# 显示最终GPU状态
echo ""
echo "最终GPU状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv