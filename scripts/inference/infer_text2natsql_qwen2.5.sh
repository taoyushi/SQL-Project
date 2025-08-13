#!/bin/bash
# infer_text2natsql_qwen25.sh
# 使用qwen2.5系列模型的RESDSQL推理脚本

set -e

device="0"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

if [ "$1" = "base" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-base/checkpoint-14352"
    text2natsql_model_bs=16
elif [ "$1" = "large" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-large/checkpoint-21216"
    text2natsql_model_bs=8
elif [ "$1" = "3b" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-78302"
    text2natsql_model_bs=6
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

model_name="resdsql_$1_natsql_qwen25"

if [ "$2" = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider/dev.json"
    db_path="./database"
    output="./predictions/Spider-dev/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/Spider-dev/$model_name/pred_qwen25_detailed.json"
elif [ "$2" = "spider-realistic" ]
then
    # spider-realistic
    table_path="./data/spider-realistic/tables.json"
    input_dataset_path="./data/spider-realistic/spider-realistic.json"
    db_path="./database"
    output="./predictions/spider-realistic/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/spider-realistic/$model_name/pred_qwen25_detailed.json"
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
elif [ "$2" = "spider-dk" ]
then
    # spider-dk
    table_path="./data/spider-dk/tables.json"
    input_dataset_path="./data/spider-dk/Spider-DK.json"
    db_path="./database"
    output="./predictions/spider-dk/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/spider-dk/$model_name/pred_qwen25_detailed.json"
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
elif [ "$2" = "spider-syn" ]
then
    # spider-syn
    table_path="./data/spider-syn/tables.json"
    input_dataset_path="./data/spider-syn/dev_syn.json"
    db_path="./database"
    output="./predictions/spider-syn/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/spider-syn/$model_name/pred_qwen25_detailed.json"
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
else
    echo "The second arg must be a valid dataset name."
    exit
fi

# 创建输出目录
mkdir -p "$(dirname "$output")"

echo "🚀 Starting RESDSQL inference with qwen2.5 self-correction..."
echo "📊 Model: $model_name"
echo "📁 Dataset: $2"
echo "💾 Output: $output"
echo ""

# 测试qwen2.5 API连接
echo "🔍 Testing qwen2.5 API connection..."
python test_qwen25_api.py
if [ $? -ne 0 ]; then
    echo "❌ qwen2.5 API test failed. Please check your configuration."
    echo "💡 Edit configs/self_correction_config_qwen25.yaml with correct endpoint and model"
    exit 1
fi
echo "✅ qwen2.5 API connection verified"
echo ""

# prepare table file for natsql
echo "📋 Preparing tables for NatSQL..."
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

# preprocess test set
echo "🔧 Preprocessing test set..."
python preprocessing.py \
    --mode "test" \
    --table_path $table_path \
    --input_dataset_path $input_dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --db_path $db_path \
    --target_type "natsql"

# predict probability for each schema item in the test set
echo "🎯 Predicting schema item probabilities..."
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2natsql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --output_filepath "./data/preprocessed_data/test_with_probs_natsql.json" \
    --use_contents \
    --mode "test"

# generate text2natsql test set
echo "📝 Generating text2natsql test set..."
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/test_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_test_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# inference using the best text2natsql ckpt with qwen2.5 self-correction
echo "🧠 Running inference with qwen2.5 self-correction module..."
echo "⏱️  This may take a while due to API calls..."
export PYTHONPATH=$(pwd)
python text2sql.py \
    --batch_size $text2natsql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $text2natsql_model_save_path \
    --mode "eval" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath $input_dataset_path \
    --db_path $db_path \
    --tables_for_natsql $tables_for_natsql \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output $output \
    --enable_self_correction \
    --correction_config_path "configs/self_correction_config_optimized.yaml" #修改部分

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Inference completed successfully!"
    echo "📊 Results saved to: $output"
    echo "📝 Detailed results saved to: ${output%.*}_detailed.json"
    echo "📁 Correction logs saved to: ./correction_logs_qwen25/"
    echo ""
    echo "📈 To compare with baseline:"
    echo "   1. Run original: bash infer_text2natsql_cspider.sh $1 $2"
    echo "   2. Compare the exact_match and exec scores"
    echo ""
    echo "🔍 To analyze correction effectiveness:"
    echo "   - Check correction logs in ./correction_logs_qwen25/"
    echo "   - Review detailed JSON output for correction statistics"
else
    echo ""
    echo "❌ Inference failed. Please check the logs above."
    exit 1
fi