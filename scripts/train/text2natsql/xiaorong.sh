#!/bin/bash

# 使用无门控CASA变体
CASA_MODEL="models/casa_without_gate"
DATASET="spider"

echo "Predicting schema item probabilities with CASA without gate..."
python predict_with_casa.py \
    --dataset_filepath data/${DATASET}/dev.json \
    --model_path ${CASA_MODEL} \
    --output_filepath data/${DATASET}/casa_without_gate_predictions.json \
    --device 0 \
    --disable_context_gate

echo "Running inference with CASA without gate predictions..."
python scripts/inference/original_infer_script.py \
    --casa_predictions_path data/${DATASET}/casa_without_gate_predictions.json \
    --output_dir outputs/${DATASET}_casa_without_gate_output \
    # 其他参数...