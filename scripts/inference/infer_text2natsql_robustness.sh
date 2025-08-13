#!/bin/bash
# infer_text2natsql_robustness.sh
# 使用qwen2.5系列模型在Spider-DK、Spider-Syn、Spider-Realistic数据集上进行鲁棒性测试

set -e

device="0"

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <model_size> <dataset>"
    echo "  model_size: base, large, 3b"
    echo "  dataset: spider-dk, spider-syn, spider-realistic"
    echo ""
    echo "示例:"
    echo "  $0 large spider-dk"
    echo "  $0 3b spider-syn"
    echo "  $0 base spider-realistic"
    exit 1
fi

# 设置模型参数
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
    echo "❌ 错误: 第一个参数必须是 [base, large, 3b] 之一"
    exit 1
fi

model_name="resdsql_$1_natsql_qwen25_robustness"

# 设置数据集参数
if [ "$2" = "spider-dk" ]
then
    dataset_name="spider-dk"
    table_path="./data/spider-dk/tables-hybrid.json"  # 使用混合版本的tables.json
    input_dataset_path="./data/spider-dk/Spider-DK-hybrid.json"  # 使用混合版本的数据集
    db_path="./data/spider-dk/database_hybrid"  # 使用混合数据库目录
    output="./predictions/robustness/$model_name/spider-dk/pred_qwen25.sql"
    detailed_output="./predictions/robustness/$model_name/spider-dk/pred_qwen25_detailed.json"
    tables_for_natsql="./data/preprocessed_data/spider_dk_tables_for_natsql.json"
    preprocessed_test="./data/preprocessed_data/preprocessed_spider_dk_natsql.json"
    test_with_probs="./data/preprocessed_data/spider_dk_with_probs_natsql.json"
    resdsql_test="./data/preprocessed_data/resdsql_spider_dk_natsql.json"
    
    # 3b模型使用不同的checkpoint
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
    
elif [ "$2" = "spider-syn" ]
then
    dataset_name="spider-syn"
    table_path="./data/spider/tables.json"  # 使用原始Spider的tables.json
    input_dataset_path="./data/spider-syn/dev_syn.json"  # 使用同义词版本
    db_path="./database"  # 使用原始Spider数据库
    output="./predictions/robustness/$model_name/spider-syn/pred_qwen25.sql"
    detailed_output="./predictions/robustness/$model_name/spider-syn/pred_qwen25_detailed.json"
    tables_for_natsql="./data/preprocessed_data/spider_syn_tables_for_natsql.json"
    preprocessed_test="./data/preprocessed_data/preprocessed_spider_syn_natsql.json"
    test_with_probs="./data/preprocessed_data/spider_syn_with_probs_natsql.json"
    resdsql_test="./data/preprocessed_data/resdsql_spider_syn_natsql.json"
    
    # 3b模型使用不同的checkpoint
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
    
elif [ "$2" = "spider-realistic" ]
then
    dataset_name="spider-realistic"
    table_path="./data/spider/tables.json"  # 使用原始Spider的tables.json
    input_dataset_path="./data/spider-realistic/spider-realistic.json"
    db_path="./database"  # 使用原始Spider数据库
    output="./predictions/robustness/$model_name/spider-realistic/pred_qwen25.sql"
    detailed_output="./predictions/robustness/$model_name/spider-realistic/pred_qwen25_detailed.json"
    tables_for_natsql="./data/preprocessed_data/spider_realistic_tables_for_natsql.json"
    preprocessed_test="./data/preprocessed_data/preprocessed_spider_realistic_natsql.json"
    test_with_probs="./data/preprocessed_data/spider_realistic_with_probs_natsql.json"
    resdsql_test="./data/preprocessed_data/resdsql_spider_realistic_natsql.json"
    
    # 3b模型使用不同的checkpoint
    if [ "$1" = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
    
else
    echo "❌ 错误: 第二个参数必须是 [spider-dk, spider-syn, spider-realistic] 之一"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$output")"

echo "🚀 Starting RESDSQL robustness test with qwen2.5 self-correction..."
echo "📊 Model: $model_name"
echo "📁 Dataset: $dataset_name"
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

# 检查预处理文件是否存在，如果不存在则进行预处理
if [ ! -f "$resdsql_test" ]; then
    echo "📋 Preprocessing files not found. Starting preprocessing for $dataset_name..."
    
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
        --output_dataset_path $preprocessed_test \
        --db_path $db_path \
        --target_type "natsql"

    # predict probability for each schema item in the test set
    echo "🎯 Predicting schema item probabilities..."
    python schema_item_classifier.py \
        --batch_size 2 \
        --device $device \
        --seed 42 \
        --save_path "./models/text2natsql_schema_item_classifier" \
        --dev_filepath $preprocessed_test \
        --output_filepath $test_with_probs \
        --use_contents \
        --mode "test"

    # generate text2natsql test set
    echo "📝 Generating text2natsql test set..."
    python text2sql_data_generator.py \
        --input_dataset_path $test_with_probs \
        --output_dataset_path $resdsql_test \
        --topk_table_num 4 \
        --topk_column_num 5 \
        --mode "test" \
        --use_contents \
        --output_skeleton \
        --target_type "natsql"
    
    echo "✅ Preprocessing completed for $dataset_name"
else
    echo "✅ Preprocessing files already exist for $dataset_name"
fi

echo ""

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
    --dev_filepath $resdsql_test \
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
    echo "🎉 Robustness test completed successfully!"
    echo "📊 Results saved to: $output"
    echo "📝 Detailed results saved to: $detailed_output"
    echo "📁 Correction logs saved to: ./correction_logs_qwen25/"
    echo ""
    echo "📈 To evaluate the results:"
    echo "   python -m utils.spider_metric.evaluator --gold $input_dataset_path --pred $output --db $db_path --table $table_path"
    echo ""
    echo "🔍 To analyze correction effectiveness:"
    echo "   - Check correction logs in ./correction_logs_qwen25/"
    echo "   - Review detailed JSON output for correction statistics"
    echo ""
    echo "📊 Dataset-specific analysis:"
    if [ "$dataset_name" = "spider-dk" ]; then
        echo "   - Spider-DK: Tests domain knowledge robustness"
    elif [ "$dataset_name" = "spider-syn" ]; then
        echo "   - Spider-Syn: Tests synonym robustness"
    elif [ "$dataset_name" = "spider-realistic" ]; then
        echo "   - Spider-Realistic: Tests realistic language robustness"
    fi
else
    echo ""
    echo "❌ Robustness test failed. Please check the logs above."
    exit 1
fi 