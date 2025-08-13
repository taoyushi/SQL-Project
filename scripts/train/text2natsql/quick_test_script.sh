#!/bin/bash

echo "=== 快速测试CASA多GPU AUC修复 ==="

# 只训练2个epoch进行快速测试
python -u casa_schema_item_classifier.py \
    --batch_size 4 \
    --gradient_descent_step 4 \
    --device "0" \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 2 \
    --patience 5 \
    --seed 42 \
    --save_path "./models/casa_test_auc" \
    --train_filepath "./data/preprocessed_data/preprocessed_train_spider_natsql.json" \
    --dev_filepath "./data/preprocessed_data/preprocessed_dev_natsql.json" \
    --model_name_or_path "/root/bayes-gpfs-b83d8de711584bc9a9a79f29860df406/tys/RESDSQL/models/roberta-large-local" \
    --use_contents \
    --add_fk_info \
    --mode "train" \
    --fp16 \
    --use_multi_gpu

echo "测试完成! 检查是否有非零AUC值"