#!/bin/bash
set -e

# 设置PYTHONPATH以支持创新模型导入
export PYTHONPATH=$(pwd)/bart_nl2sql
echo "PYTHONPATH set to: $PYTHONPATH"

device="0"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

# 第一阶段模型路径（不变）
schema_classifier_path="./models/text2natsql_schema_item_classifier"

# 第二阶段模型路径（使用您的BART+CopyNet创新模型）
text2sql_model_save_path="./models/model_best"
text2sql_model_bs=2  # 进一步减少batch size以节省显存
bart_model_name="./bart_nl2sql/facebook-bart-base"  # 使用本地英文BART基础模型

# spider dev集路径
table_path="./data/spider/tables.json"
input_dataset_path="./data/spider/dev.json"
db_path="./database"
output="./predictions/Spider-dev/english_sql/pred.sql"

# 创建输出目录
mkdir -p "./predictions/Spider-dev/english_sql"

# 1. 生成NatSQL表结构
python NatSQL/table_transform.py \
    --in_file $table_path \
    --out_file $tables_for_natsql \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path $db_path

# 2. 预处理dev集（生成SQL格式）
python preprocessing.py \
    --mode "test" \
    --table_path $table_path \
    --input_dataset_path $input_dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_test_sql.json" \
    --db_path $db_path \
    --target_type "sql"

# 3. 预测schema item概率（第一阶段，保持原有路径）
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path $schema_classifier_path \
    --dev_filepath "./data/preprocessed_data/preprocessed_test_sql.json" \
    --output_filepath "./data/preprocessed_data/test_with_probs_sql.json" \
    --use_contents \
    --mode "test"

# 4. 生成T5输入文件
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/test_with_probs_sql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_test_sql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "sql"

# 5. 用您的BART+CopyNet创新模型做推理（第二阶段）
python text2sql.py \
    --batch_size $text2sql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $text2sql_model_save_path \
    --model_name_or_path $bart_model_name \
    --mode "eval" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_sql.json" \
    --original_dev_filepath $input_dataset_path \
    --db_path $db_path \
    --tables_for_natsql $tables_for_natsql \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql" \
    --output $output

echo "BART+CopyNet创新模型英文SQL推理完成！结果保存在: $output" 