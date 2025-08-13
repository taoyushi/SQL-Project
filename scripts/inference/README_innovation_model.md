# BART+CopyNet创新模型说明

## 模型架构

您的创新点是一个基于BART的CopyNet机制模型，用于替换RESDSQL的第二阶段Text2SQL生成模块。

### 核心组件

1. **CopyMechModule**: 复制机制模块
   - 计算生成概率 p_gen 和复制概率 p_copy
   - 结合词汇表生成和复制机制
   - 使用交叉注意力机制实现复制

2. **BartForConditionalGenerationWithCopyMech**: 主模型
   - 继承自BART的序列到序列生成模型
   - 集成了CopyMechModule
   - 支持从输入中复制表名、列名等实体

### 创新特点

1. **CopyNet机制**: 
   - 可以从输入的自然语言中直接复制表名、列名
   - 减少词汇表外词汇的生成错误
   - 提高SQL生成的准确性

2. **英文优化**:
   - 使用英文BART基础模型 (`facebook/bart-base`)
   - 针对英文SQL生成任务优化

3. **端到端训练**:
   - 与RESDSQL的第一阶段Schema Classifier无缝集成
   - 支持端到端的训练和推理

## 使用方式

### 训练
```bash
python text2sql.py --mode train --save_path ./models/model_best --model_name_or_path facebook/bart-base
```

### 推理
```bash
# 使用提供的推理脚本
./scripts/inference/infer_english_sql.sh
```

## 与原始RESDSQL的对比

| 组件 | 原始RESDSQL | 您的创新模型 |
|------|-------------|-------------|
| 第一阶段 | Schema Item Classifier | Schema Item Classifier (不变) |
| 第二阶段 | T5/MT5生成模型 | BART+CopyNet生成模型 |
| 主要优势 | 标准序列到序列生成 | 集成复制机制，更适合实体复制任务 |

## 文件结构

```
bart_nl2sql/
├── src/nl2sql/
│   ├── modeling.py          # 核心模型定义
│   ├── train.py            # 训练脚本
│   └── argument.py         # 参数配置
└── requirements.txt        # 依赖包
```

## 模型文件

训练好的模型保存在 `./models/model_best/` 目录下：
- `model.pt`: 模型权重文件
- `vocab.json`: 词汇表
- `tokenizer_config.json`: 分词器配置
- `special_tokens_map.json`: 特殊token映射
- `merges.txt`: BPE合并规则 