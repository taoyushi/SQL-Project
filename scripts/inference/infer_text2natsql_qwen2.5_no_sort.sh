 #!/bin/bash
# infer_text2natsql_qwen2.5_no_sort.sh
# 消融实验：不经过schema排序，直接生成SQL并自修正

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

model_name="resdsql_${1}_natsql_qwen25_no_sort"

if [ "$2" = "spider" ]
then
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider/dev.json"
    db_path="./database"
    output="./predictions/Spider-dev/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/Spider-dev/$model_name/pred_qwen25_detailed.json"
elif [ "$2" = "spider-realistic" ]
then
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider-realistic/spider-realistic.json"
    db_path="./database"
    output="./predictions/spider-realistic/$model_name/pred_qwen25.sql"
    detailed_output="./predictions/spider-realistic/$model_name/pred_qwen25_detailed.json"
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

echo "🚀 Starting RESDSQL inference (no sort) with qwen2.5 self-correction..."
echo "📊 Model: $model_name"
echo "📁 Dataset: $2"
echo "💾 Output: $output"
echo ""

# prepare table file for natsql
# (假设已准备好 test_tables_for_natsql.json，无需重复)

# preprocess test set (假设已准备好 preprocessed_test_natsql.json，无需重复)

# generate text2natsql test set (不经过排序，直接用preprocessed_test_natsql.json)
echo "📝 Generating text2natsql test set (no sort)..."
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_test_natsql_no_sort.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# inference using the best text2natsql ckpt with qwen2.5 self-correction
echo "🧠 Running inference with qwen2.5 self-correction module (no sort)..."
echo "⏱️  This may take a while due to API calls..."
export PYTHONPATH=$(pwd)
python text2sql.py \
    --batch_size $text2natsql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $text2natsql_model_save_path \
    --mode "eval" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql_no_sort.json" \
    --original_dev_filepath $input_dataset_path \
    --db_path $db_path \
    --tables_for_natsql $tables_for_natsql \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output $output \
    --enable_self_correction \
    --correction_config_path "configs/self_correction_config_optimized.yaml"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Inference (no sort) completed successfully!"
    echo "📊 Results saved to: $output"
    echo "📝 Detailed results saved to: ${output%.*}_detailed.json"
    echo "📁 Correction logs saved to: ./correction_logs_qwen25/"
    echo ""
    echo "📈 To compare with baseline:"
    echo "   1. Run original: bash infer_text2natsql_qwen2.5.sh $1 $2"
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
