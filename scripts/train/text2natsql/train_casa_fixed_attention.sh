#!/bin/bash
set -e

# 回到 RESDSQL 根目录
cd ../../..

echo "Starting training for CASA with fixed attention schema item classifier..."

# 定义训练参数 (与完整CASA基本一致)
BATCH_SIZE=16
GRADIENT_DESCENT_STEP=4
DEVICE="0" # 根据你的GPU设备ID修改
LEARNING_RATE=5e-6
GAMMA=2.0
ALPHA=0.75
EPOCHS=50
PATIENCE=5
SEED=42
SAVE_PATH="models/casa_fixed_attention" # 保存路径区分
TENSORBOARD_SAVE_PATH="tensorboard_log/casa_fixed_attention" # Tensorboard路径区分
TRAIN_FILEPATH="data/preprocessed_data/preprocessed_train_spider_natsql.json" # <--- 修改这里
DEV_FILEPATH="data/preprocessed_data/preprocessed_dev_natsql.json"   
MODEL_NAME_OR_PATH="/root/bayes-gpfs-b83d8de711584bc9a9a79f29860df406/tys/RESDSQL/models/roberta-large-local"  # Cross-encoder 基础模型

# 确保保存目录存在
mkdir -p ${SAVE_PATH}
mkdir -p ${TENSORBOARD_SAVE_PATH}

# 运行 CASA 训练脚本，添加 --fixed_attention_heads 参数
python casa_schema_item_classifier.py \
    --batch_size ${BATCH_SIZE} \
    --gradient_descent_step ${GRADIENT_DESCENT_STEP} \
    --device ${DEVICE} \
    --learning_rate ${LEARNING_RATE} \
    --gamma ${GAMMA} \
    --alpha ${ALPHA} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --seed ${SEED} \
    --save_path ${SAVE_PATH} \
    --tensorboard_save_path ${TENSORBOARD_SAVE_PATH} \
    --train_filepath ${TRAIN_FILEPATH} \
    --dev_filepath ${DEV_FILEPATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --use_contents \
    --add_fk_info \
    --fixed_attention_heads \ # <--- 启用固定注意力消融
    --mode train

echo "CASA with fixed attention schema item classifier training finished."