#!/bin/bash
set -e

# 在脚本开头回到 RESDSQL 根目录，如果当前不在根目录
# (假设脚本从 scripts/inference/ 执行)
cd "$(dirname "$0")/../.." 

echo "Starting Text-to-SQL inference with CASA schema item probabilities..."

# --- 1. 定义参数 ---
# 模型规模和数据集
MODEL_SCALE=${1:-"3b"} # 默认使用 3b 如果没有参数传入
DATASET=${2:-"spider"} # 默认使用 spider 如果没有参数传入

# 根据模型规模设置T5模型路径和批次大小
if [ "${MODEL_SCALE}" = "base" ]
then
    T5_MODEL_SAVE_PATH="./models/text2natsql-t5-base/checkpoint-14352"
    T5_MODEL_BS=16
elif [ "${MODEL_SCALE}" = "large" ]
then
    T5_MODEL_SAVE_PATH="./models/text2natsql-t5-large/checkpoint-21216"
    T5_MODEL_BS=8
elif [ "${MODEL_SCALE}" = "3b" ]
then
    T5_MODEL_SAVE_PATH="./models/text2natsql-t5-3b/checkpoint-78302"
    T5_MODEL_BS=6
else
    echo "Error: Invalid model scale '${MODEL_SCALE}'. Must be in [base, large, 3b]."
    exit 1
fi

# 定义你的 CASA Cross-encoder 模型路径
CASA_CROSS_ENCODER_PATH="./models/casa_schema_item_classifier" # <--- 根据你的训练脚本保存路径修改

# 根据数据集设置输入文件路径、数据库路径、输出目录等
# 这里的路径需要与 RESDSQL 原有脚本和数据解压位置一致
if [ "${DATASET}" = "spider" ]
then
    # spider's dev set
    TABLE_PATH="./data/spider/tables.json"
    INPUT_DATASET_PATH="./data/spider/dev.json" # 原始dev文件
    PREPROCESSED_INPUT_PATH="./data/preprocessed_data/preprocessed_test_natsql.json" # 预处理文件
    DB_PATH="./database" # 数据库根目录
    GOLD_SQL_PATH="./data/spider/dev_gold.sql" # gold sql文件
    OUTPUT_DIR="./predictions/Spider-dev/resdsql_${MODEL_SCALE}_natsql_casa" # 最终SQL输出目录
elif [ "${DATASET}" = "spider-realistic" ]
then
    # spider-realistic
    TABLE_PATH="./data/spider/tables.json" # 通常与spider使用相同的tables.json
    INPUT_DATASET_PATH="./data/spider-realistic/spider-realistic.json"
    PREPROCESSED_INPUT_PATH="./data/preprocessed_data/preprocessed_test_natsql_${DATASET}.json" # 预处理文件 (可能需要根据实际生成的文件名修改)
    DB_PATH="./database"
    GOLD_SQL_PATH="./data/spider-realistic/spider-realistic_gold.sql" # gold sql文件 (需要确认实际文件名)
    OUTPUT_DIR="./predictions/${DATASET}/resdsql_${MODEL_SCALE}_natsql_casa"
    # RESDSQL 3B 在 spider-realistic 使用了不同的 checkpoint
    if [ "${MODEL_SCALE}" = "3b" ]
    then
        T5_MODEL_SAVE_PATH="./models/text2natsql-t5-3b/checkpoint-61642" # <--- 根据实际下载的 checkpoint 修改
    fi
elif [ "${DATASET}" = "spider-syn" ]
then
    # spider-syn
    TABLE_PATH="./data/spider/tables.json"
    INPUT_DATASET_PATH="./data/spider-syn/dev_syn.json"
    PREPROCESSED_INPUT_PATH="./data/preprocessed_data/preprocessed_test_natsql_${DATASET}.json" # 预处理文件 (可能需要根据实际生成的文件名修改)
    DB_PATH="./database"
    GOLD_SQL_PATH="./data/spider-syn/dev_syn_gold.sql" # gold sql文件 (需要确认实际文件名)
    OUTPUT_DIR="./predictions/${DATASET}/resdsql_${MODEL_SCALE}_natsql_casa"
elif [ "${DATASET}" = "spider-dk" ]
then
    # spider-dk
    TABLE_PATH="./data/spider-dk/tables.json"
    INPUT_DATASET_PATH="./data/spider-dk/Spider-DK.json"
    PREPROCESSED_INPUT_PATH="./data/preprocessed_data/preprocessed_test_natsql_${DATASET}.json" # 预处理文件 (可能需要根据实际生成的文件名修改)
    DB_PATH="./database"
    GOLD_SQL_PATH="./data/spider-dk/Spider-DK_gold.sql" # gold sql文件 (需要确认实际文件名)
    OUTPUT_DIR="./predictions/${DATASET}/resdsql_${MODEL_SCALE}_natsql_casa"
# --- 添加 Dr.Spider 各变体的路径，需要根据实际文件位置和 RESDSQL 原有脚本进行设置 ---
# 这些路径设置需要与 RESDSQL 原有脚本 scripts/evaluate_robustness/evaluate_on_spider_*.sh 一致
# 例如：
# elif [ "${DATASET}" = "DB_DBcontent_equivalence" ]
# then
#     TABLE_PATH="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/tables_post_perturbation.json"
#     INPUT_DATASET_PATH="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/questions_post_perturbation.json"
#     PREPROCESSED_INPUT_PATH="./data/preprocessed_data/preprocessed_test_natsql_${DATASET}.json" # 预处理文件
#     DB_PATH="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/database_post_perturbation"
#     GOLD_SQL_PATH="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/questions_post_perturbation_gold.sql" # gold sql
#     OUTPUT_DIR="./predictions/${DATASET}/resdsql_${MODEL_SCALE}_natsql_casa"
# --- 其他 Dr.Spider 变体省略，请参考 RESDSQL 原有脚本进行补全 ---
else
    echo "Error: Invalid dataset name '${DATASET}'."
    exit 1
fi

# 定义生成器的 Schema 过滤参数
TOPK_TABLE_NUM=4
TOPK_COLUMN_NUM=5

# 定义中间文件路径
TABLES_FOR_NATSQL="./data/preprocessed_data/test_tables_for_natsql_${DATASET}.json" # NatSQL tables 文件 (命名区分数据集)
CASA_PREDICTIONS_FILE="./data/preprocessed_data/casa_test_with_probs_natsql_${DATASET}.json" # CASA预测概率输出文件 (命名区分数据集)
T5_INPUT_FILE="./data/preprocessed_data/resdsql_test_natsql_casa_${DATASET}.json" # 生成的T5输入文件 (命名区分数据集)

# --- 2. 确保输出目录存在 ---
mkdir -p ${OUTPUT_DIR}

# --- 3. 准备 table file for natsql ---
# 这个脚本需要运行，因为它生成 NatSQL 相关的 tables 文件，T5 评估时需要
echo "Preparing tables file for NatSQL..."
python NatSQL/table_transform.py \
    --in_file "${TABLE_PATH}" \
    --out_file "${TABLES_FOR_NATSQL}" \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path "${DB_PATH}"

# --- 4. 预处理测试集 ---
# 这个脚本需要运行，因为它是 predict_with_casa.py 的输入
echo "Preprocessing test set..."
python preprocessing.py \
    --mode "test" \
    --table_path "${TABLE_PATH}" \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_dataset_path "${PREPROCESSED_INPUT_PATH}" \
    --db_path "${DB_PATH}" \
    --target_type "natsql"

# --- 5. 预测 Schema Item 概率 (使用你的 CASA Cross-encoder) ---
# 这个脚本会生成一个包含 CASA 预测概率的文件
echo "Predicting schema item probabilities with CASA model from ${CASA_CROSS_ENCODER_PATH}..."
python predict_with_casa.py \
    --batch_size 32 \ # 推理批次大小可以大一些
    --device ${DEVICE} \
    --model_path "${CASA_CROSS_ENCODER_PATH}" \ # <--- 指定你训练好的 CASA 模型路径
    --dataset_filepath "${PREPROCESSED_INPUT_PATH}" \ # <--- 输入预处理数据
    --output_filepath "${CASA_PREDICTIONS_FILE}" \ # <--- 输出带CASA概率的文件
    --use_contents \ # <--- 确保参数与 CASA 训练时一致
    --add_fk_info \ # <--- 确保参数与 CASA 训练时一致
    --mode "test"

# --- 6. 生成 T5 输入数据 (使用带 CASA 概率的文件) ---
# RESDSQL 的 text2sql_data_generator.py 会加载带概率的文件，并进行 Top-K 过滤和构建 T5 输入序列
echo "Generating T5 dataset with CASA filtered schema items (Top-K=${TOPK_TABLE_NUM}/${TOPK_COLUMN_NUM})..."
python text2sql_data_generator.py \
    --input_dataset_path "${CASA_PREDICTIONS_FILE}" \ # <--- 加载你上一步生成的带 CASA 概率的文件
    --output_dataset_path "${T5_INPUT_FILE}" \ # <--- 生成新的T5输入文件
    --topk_table_num ${TOPK_TABLE_NUM} \
    --topk_column_num ${TOPK_COLUMN_NUM} \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# --- 7. T5 推理生成 NatSQL (使用生成的 T5 输入文件和原始 T5 模型) ---
echo "Running T5 inference with CASA filtered schema items..."
# RESDSQL 的 text2sql.py 需要加载 T5 输入文件，并用 T5 模型生成 NatSQL
# 你需要在 text2sql.py 中添加逻辑来接收 --casa_predictions_path 参数 (尽管这里不需要传递了)
# 但最重要的是，text2sql.py 的 Text2SQLDataset 类要能正确加载你的 ${T5_INPUT_FILE} 文件
python text2sql.py \
    --batch_size ${T5_MODEL_BS} \ # T5 推理批次大小
    --device ${DEVICE} \
    --seed 42 \ # 通常推理也设置种子
    --save_path "${T5_MODEL_SAVE_PATH}" \ # <-- 指定原始 RESDSQL 的 T5 模型检查点路径
    --mode "eval" \ # T5 推理模式
    --dev_filepath "${T5_INPUT_FILE}" \ # <-- 加载上一步生成的 T5 输入文件
    --original_dev_filepath "${INPUT_DATASET_PATH}" \ # <-- 原始dev/test文件路径 (用于评估器)
    --db_path "${DB_PATH}" \ # <-- 数据库路径 (用于评估器)
    --tables_for_natsql "${TABLES_FOR_NATSQL}" \ # <-- NatSQL tables文件路径 (评估器需要)
    --num_beams 8 \ # Beam search 参数
    --num_return_sequences 8 \ # Beam search 参数
    --target_type "natsql" \ # 目标类型
    --output "${OUTPUT_DIR}/predictions.sql" # <--- 最终预测文件输出路径

# --- 8. 评估生成的 SQL ---
# 使用 Spider 官方评估脚本计算 EM/EX
echo "Evaluating generated SQL..."
# 假设评估脚本在 third_party/spider/evaluation.py
python third_party/spider/evaluation.py \
    --gold "${GOLD_SQL_PATH}" \ # gold 文件路径
    --pred "${OUTPUT_DIR}/predictions.sql" \
    --db "${DB_PATH}" \ # 数据库路径 (评估脚本参数名是 --db)
    --table "${TABLE_PATH}" # tables 文件路径 (评估脚本参数名是 --table)

echo "Inference and evaluation finished. Check ${OUTPUT_DIR}/predictions.sql and evaluation output."