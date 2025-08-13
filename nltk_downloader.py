import nltk
import os
import sys

# 配置NLTK数据路径
nltk_data_path = "/bin/RESDSQL/nltk_data"
nltk.data.path.insert(0, nltk_data_path)

print("NLTK下载器运行中...")
print(f"使用数据目录: {nltk_data_path}")

# 验证所有资源
resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
all_available = True

for resource in resources:
    path_prefix = 'tokenizers/' if resource == 'punkt' else 'corpora/'
    path_prefix = 'taggers/' if resource == 'averaged_perceptron_tagger' else path_prefix
    
    try:
        nltk.data.find(f'{path_prefix}{resource}')
        print(f"✓ {resource} 可用")
    except LookupError:
        print(f"✗ {resource} 不可用")
        all_available = False

if all_available:
    print("\n所有NLTK资源已成功配置!")
else:
    print("\n警告: 某些NLTK资源不可用。")
    sys.exit(1)
