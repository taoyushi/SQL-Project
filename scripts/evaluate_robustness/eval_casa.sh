#!/bin/bash
set -e

# 回到 RESDSQL 根目录
cd ../..

echo "Evaluating CASA schema item classifiers on Schema Item Classification (AUC)..."

# 定义评估参数
BATCH_SIZE=32
DEVICE="0" # 根据你的GPU设备ID修改
DEV_FILEPATH="data/preprocessed_data/preprocessed_dev_natsql.json" # <--- 修改这里，评估数据

# 评估完整 CASA 模型
CASA_FULL_MODEL_PATH="models/casa_schema_item_classifier" # 完整CASA模型保存路径
OUTPUT_FULL_PATH="data/preprocessed_data/casa_dataset_with_pred_probs_natsql.json" # 输出文件路径 (添加 _natsql 后缀)
echo "Evaluating full CASA model from ${CASA_FULL_MODEL_PATH}..."
python casa_schema_item_classifier.py \ # <--- 调用根目录下的 casa_schema_item_classifier.py
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --save_path ${CASA_FULL_MODEL_PATH} \ # <--- 加载训练好的模型
    --dev_filepath ${DEV_FILEPATH} \
    --output_filepath ${OUTPUT_FULL_PATH} \
    # --model_name_or_path roberta-large \ # <--- 在 eval 模式下，这个参数会被 save_path 中的配置覆盖，可以注释掉或移除
    --use_contents \
    --add_fk_info \
    --mode eval

# 评估无上下文门控的CASA
CASA_NOGATE_MODEL_PATH="models/casa_without_gate"
OUTPUT_NOGATE_PATH="data/preprocessed_data/casa_without_gate_dataset_with_pred_probs_natsql.json"
echo "Evaluating CASA without gate from ${CASA_NOGATE_MODEL_PATH}..."
python casa_schema_item_classifier.py \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --save_path ${CASA_NOGATE_MODEL_PATH} \
    --dev_filepath ${DEV_FILEPATH} \
    --output_filepath ${OUTPUT_NOGATE_PATH} \
    # --model_name_or_path roberta-large \
    --use_contents \
    --add_fk_info \
    --disable_context_gate \
    --mode eval

# 评估固定注意力头的CASA
CASA_FIXED_ATTN_MODEL_PATH="models/casa_fixed_attention"
OUTPUT_FIXED_ATTN_PATH="data/preprocessed_data/casa_fixed_attention_dataset_with_pred_probs_natsql.json"
echo "Evaluating CASA with fixed attention from ${CASA_FIXED_ATTN_MODEL_PATH}..."
python casa_schema_item_classifier.py \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --save_path ${CASA_FIXED_ATTN_MODEL_PATH} \
    --dev_filepath ${DEV_FILEPATH} \
    --output_filepath ${OUTPUT_FIXED_ATTN_PATH} \
    # --model_name_or_path roberta-large \
    --use_contents \
    --add_fk_info \
    --fixed_attention_heads \
    --mode eval

echo "CASA schema item classifiers AUC evaluation finished."