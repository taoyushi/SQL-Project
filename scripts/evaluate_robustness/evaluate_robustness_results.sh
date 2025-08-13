#!/bin/bash
# evaluate_robustness_results.sh
# 评估Spider-DK、Spider-Syn、Spider-Realistic数据集上的鲁棒性测试结果

set -e

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

model_name="resdsql_$1_natsql_qwen25_robustness"

# 设置数据集参数
if [ "$2" = "spider-dk" ]
then
    dataset_name="spider-dk"
    gold_file="./data/spider-dk/Spider-DK-hybrid.json"  # 使用混合版本的数据集
    table_file="./data/spider-dk/tables-hybrid.json"  # 使用混合版本的tables.json
    pred_file="./predictions/robustness/$model_name/spider-dk/pred_qwen25.sql"
    db_path="./data/spider-dk/database_hybrid"
    
elif [ "$2" = "spider-syn" ]
then
    dataset_name="spider-syn"
    gold_file="./data/spider-syn/dev_syn.json"  # 使用同义词版本
    table_file="./data/spider/tables.json"  # 使用原始Spider的tables.json
    pred_file="./predictions/robustness/$model_name/spider-syn/pred_qwen25.sql"
    db_path="./database"
    
elif [ "$2" = "spider-realistic" ]
then
    dataset_name="spider-realistic"
    gold_file="./data/spider-realistic/spider-realistic.json"
    table_file="./data/spider/tables.json"  # 使用原始Spider的tables.json
    pred_file="./predictions/robustness/$model_name/spider-realistic/pred_qwen25.sql"
    db_path="./database"
    
else
    echo "❌ 错误: 第二个参数必须是 [spider-dk, spider-syn, spider-realistic] 之一"
    exit 1
fi

# 检查预测文件是否存在
if [ ! -f "$pred_file" ]; then
    echo "❌ 错误: 预测文件不存在: $pred_file"
    echo "请先运行推理脚本: bash infer_text2natsql_robustness.sh $1 $2"
    exit 1
fi

echo "🔍 Evaluating robustness test results..."
echo "📊 Model: $model_name"
echo "📁 Dataset: $dataset_name"
echo "📄 Gold file: $gold_file"
echo "📄 Prediction file: $pred_file"
echo ""

# 创建评估结果目录
eval_results_dir="./eval_results/robustness/$model_name/$dataset_name"
mkdir -p "$eval_results_dir"

# 运行评估
echo "📈 Running evaluation..."
python -m utils.spider_metric.evaluator \
    --gold "$gold_file" \
    --pred "$pred_file" \
    --db "$db_path" \
    --table "$table_file" \
    --etype "all" \
    --output "$eval_results_dir/eval_results.json"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "📊 Results saved to: $eval_results_dir/eval_results.json"
    echo ""
    
    # 显示评估结果摘要
    if [ -f "$eval_results_dir/eval_results.json" ]; then
        echo "📋 Evaluation Summary:"
        echo "======================"
        python -c "
import json
with open('$eval_results_dir/eval_results.json', 'r') as f:
    results = json.load(f)
print(f'Exact Match: {results.get(\"exact_match\", 0):.4f}')
print(f'Execution Accuracy: {results.get(\"exec\", 0):.4f}')
print(f'Component Match: {results.get(\"component\", 0):.4f}')
print(f'Hardness Breakdown:')
for hardness, scores in results.get('hardness', {}).items():
    print(f'  {hardness}: {scores.get(\"exact_match\", 0):.4f}')
"
    fi
    
    echo ""
    echo "📊 Dataset-specific insights:"
    if [ "$dataset_name" = "spider-dk" ]; then
        echo "   - Spider-DK: 测试模型在领域知识方面的鲁棒性"
        echo "   - 关注点: 模型是否能处理特定领域的术语和概念"
    elif [ "$dataset_name" = "spider-syn" ]; then
        echo "   - Spider-Syn: 测试模型在同义词理解方面的鲁棒性"
        echo "   - 关注点: 模型是否能正确理解同义词表达的真实含义"
    elif [ "$dataset_name" = "spider-realistic" ]; then
        echo "   - Spider-Realistic: 测试模型在真实语言表达方面的鲁棒性"
        echo "   - 关注点: 模型是否能处理复杂的自然语言表达"
    fi
    
    echo ""
    echo "🔍 For detailed analysis:"
    echo "   - Check correction logs: ./correction_logs_qwen25/"
    echo "   - Review detailed predictions: ${pred_file%.*}_detailed.json"
    echo "   - Compare with baseline results"
    
else
    echo ""
    echo "❌ Evaluation failed. Please check the logs above."
    exit 1
fi 