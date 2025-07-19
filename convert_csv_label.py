"""
从CSV文件中读取标签，将字符串标签转换为数字标签，并保存为新的CSV文件
此外，还能更新JSON文件中的标签映射并重新进行标签转换
"""
import pandas as pd
import json
import os

# 配置路径
input_csv = '/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/datasets/0717/multi_inputs_0717.csv'
label_json = '/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json'
output_csv = '/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/datasets/0717/multi_inputs_0717_train.csv'

# 读取标签映射（数字->字符串）
with open(label_json, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# 构造字符串->数字的反向映射
str2num = {v: int(k) for k, v in label_map.items()}

# 读取CSV
# 假设B列为第二列（索引1）
df = pd.read_csv(input_csv, encoding='utf-8')

# 处理B列（从第二行开始）
missing_labels = set()
for idx in range(0, len(df)):
    orig_value = str(df.iloc[idx, 4])  # B列
    mapped = str2num.get(orig_value)
    if mapped is not None:
        df.iat[idx, 4] = mapped
    else:
        missing_labels.add(orig_value)

# 打印未映射的标签
if missing_labels:
    print('未找到数字标签的字符串:')
    for s in missing_labels:
        print(s)

# 保存到新CSV
os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f'转换完成，结果已保存到: {os.path.abspath(output_csv)}') 