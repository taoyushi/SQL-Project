#!/bin/bash
# run_all_robustness_tests.sh
# 批量运行Spider-DK、Spider-Syn、Spider-Realistic三个数据集的鲁棒性测试

set -e

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <model_size> [datasets...]"
    echo "  model_size: base, large, 3b"
    echo "  datasets: spider-dk, spider-syn, spider-realistic (可选，默认运行所有)"
    echo ""
    echo "示例:"
    echo "  $0 large                    # 运行所有数据集"
    echo "  $0 3b spider-dk spider-syn  # 只运行指定数据集"
    echo "  $0 base spider-realistic    # 只运行一个数据集"
    exit 1
fi

model_size="$1"
shift

# 如果没有指定数据集，默认运行所有
if [ $# -eq 0 ]; then
    datasets=("spider-dk" "spider-syn" "spider-realistic")
else
    datasets=("$@")
fi

# 验证数据集名称
valid_datasets=("spider-dk" "spider-syn" "spider-realistic")
for dataset in "${datasets[@]}"; do
    if [[ ! " ${valid_datasets[@]} " =~ " ${dataset} " ]]; then
        echo "❌ 错误: 无效的数据集名称 '$dataset'"
        echo "有效的数据集: ${valid_datasets[*]}"
        exit 1
    fi
done

echo "🚀 Starting comprehensive robustness tests..."
echo "📊 Model size: $model_size"
echo "📁 Datasets: ${datasets[*]}"
echo ""

# 创建总结果目录
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="./robustness_test_results_${model_size}_${timestamp}"
mkdir -p "$results_dir"

# 记录开始时间
start_time=$(date +%s)

# 运行每个数据集的测试
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "🔬 Testing on $dataset dataset..."
    echo "=========================================="
    
    # 运行推理
    echo "📋 Running inference..."
    if bash scripts/inference/infer_text2natsql_robustness.sh "$model_size" "$dataset"; then
        echo "✅ Inference completed for $dataset"
    else
        echo "❌ Inference failed for $dataset"
        continue
    fi
    
    echo ""
    
    # 运行评估
    echo "📈 Running evaluation..."
    if bash scripts/evaluate_robustness/evaluate_robustness_results.sh "$model_size" "$dataset"; then
        echo "✅ Evaluation completed for $dataset"
    else
        echo "❌ Evaluation failed for $dataset"
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ Completed $dataset"
    echo "=========================================="
    echo ""
done

# 计算总耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "🎉 All robustness tests completed!"
echo "⏱️  Total time: ${hours}h ${minutes}m ${seconds}s"
echo ""

# 生成汇总报告
echo "📊 Generating summary report..."
summary_file="$results_dir/summary_report.md"

cat > "$summary_file" << EOF
# Robustness Test Summary Report

**Model Size:** $model_size  
**Test Date:** $(date)  
**Total Duration:** ${hours}h ${minutes}m ${seconds}s

## Test Configuration
- **Datasets Tested:** ${datasets[*]}
- **Model:** resdsql_${model_size}_natsql_qwen25_robustness
- **Self-correction:** Enabled (qwen2.5)

## Results Summary

EOF

# 收集每个数据集的结果
for dataset in "${datasets[@]}"; do
    model_name="resdsql_${model_size}_natsql_qwen25_robustness"
    eval_file="./eval_results/robustness/$model_name/$dataset/eval_results.json"
    
    if [ -f "$eval_file" ]; then
        echo "### $dataset" >> "$summary_file"
        
        # 提取评估结果
        exact_match=$(python -c "
import json
try:
    with open('$eval_file', 'r') as f:
        results = json.load(f)
    print(f'{results.get(\"exact_match\", 0):.4f}')
except:
    print('N/A')
")
        
        exec_acc=$(python -c "
import json
try:
    with open('$eval_file', 'r') as f:
        results = json.load(f)
    print(f'{results.get(\"exec\", 0):.4f}')
except:
    print('N/A')
")
        
        echo "- **Exact Match:** $exact_match" >> "$summary_file"
        echo "- **Execution Accuracy:** $exec_acc" >> "$summary_file"
        echo "" >> "$summary_file"
        
        # 显示结果
        echo "📊 $dataset Results:"
        echo "   Exact Match: $exact_match"
        echo "   Execution Accuracy: $exec_acc"
        echo ""
    else
        echo "❌ No evaluation results found for $dataset"
        echo "### $dataset" >> "$summary_file"
        echo "- **Status:** Failed/No results" >> "$summary_file"
        echo "" >> "$summary_file"
    fi
done

# 添加数据集说明
cat >> "$summary_file" << EOF

## Dataset Descriptions

### Spider-DK
- **Purpose:** Tests domain knowledge robustness
- **Focus:** Model's ability to handle domain-specific terminology and concepts
- **Challenge:** Understanding specialized vocabulary and domain context

### Spider-Syn
- **Purpose:** Tests synonym robustness  
- **Focus:** Model's ability to understand synonymous expressions
- **Challenge:** Correctly interpreting the true meaning behind synonym usage

### Spider-Realistic
- **Purpose:** Tests realistic language robustness
- **Focus:** Model's ability to handle complex natural language expressions
- **Challenge:** Processing real-world, complex linguistic patterns

## Files Generated

- **Predictions:** \`./predictions/robustness/resdsql_${model_size}_natsql_qwen25_robustness/\`
- **Evaluation Results:** \`./eval_results/robustness/resdsql_${model_size}_natsql_qwen25_robustness/\`
- **Correction Logs:** \`./correction_logs_qwen25/\`
- **Detailed Results:** \`*_detailed.json\` files

## Next Steps

1. Compare results across datasets to identify model weaknesses
2. Analyze correction logs to understand self-correction effectiveness
3. Compare with baseline (non-corrected) results
4. Investigate specific failure cases for each dataset type
EOF

echo "📄 Summary report saved to: $summary_file"
echo ""

# 显示最终汇总
echo "📋 Final Summary:"
echo "================="
echo "📊 Model: resdsql_${model_size}_natsql_qwen25_robustness"
echo "📁 Datasets tested: ${datasets[*]}"
echo "📄 Summary report: $summary_file"
echo ""
echo "🔍 For detailed analysis:"
echo "   - Check individual evaluation results in ./eval_results/robustness/"
echo "   - Review correction logs in ./correction_logs_qwen25/"
echo "   - Compare with baseline results"
echo ""
echo "🎯 Key insights to look for:"
echo "   - Which dataset type is most challenging for the model?"
echo "   - How effective is self-correction across different robustness challenges?"
echo "   - Are there patterns in the types of errors made on each dataset?" 