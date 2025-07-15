import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import argparse
import json

parser = argparse.ArgumentParser(description='Prepare data for classification task.')
parser.add_argument('--balance', action='store_true', help='Whether to balance the label distribution.')
parser.add_argument('--input', type=str, default='./新国标分类/新国标固定资产_去重.xlsx', help='Input Excel file path.')
parser.add_argument('--text_col', type=str, default='资产名称', help='Text column name.')
parser.add_argument('--label_col', type=str, default='新国标分类', help='Label column name.')
parser.add_argument('--train_out', type=str, default='./新国标分类/train.csv', help='Output train file.')
parser.add_argument('--dev_out', type=str, default='./新国标分类/dev.csv', help='Output dev file.')
parser.add_argument('--filter_single', action='store_true', help='是否筛除只出现一次的类别（如不加则全部保留，单样本类别全部进train）')
args = parser.parse_args()

# 读取数据
df = pd.read_excel(args.input)
df = df[[args.text_col, args.label_col]].dropna()

if args.balance:
    min_count = 10
    vc = df[args.label_col].value_counts()
    valid_labels = vc[vc >= min_count].index
    df = df[df[args.label_col].isin(valid_labels)]
    balanced_df = []
    for label, group in df.groupby(args.label_col):
        balanced_group = resample(group, replace=False, n_samples=min_count, random_state=42)
        balanced_df.append(balanced_group)
    df = pd.concat(balanced_df).reset_index(drop=True)
    print(f'已进行标签平衡，每类样本数: {min_count}（仅保留样本数大于等于{min_count}的类别）')

# 标签编码
label2id = {label: idx for idx, label in enumerate(df[args.label_col].unique())}
id2label = {idx: label for label, idx in label2id.items()}
df['label'] = df[args.label_col].map(label2id)

if args.filter_single:
    # 只保留样本数大于1的类别
    vc = df['label'].value_counts()
    valid_labels = vc[vc > 1].index
    df = df[df['label'].isin(valid_labels)]
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
else:
    # 自定义分层逻辑
    vc = df['label'].value_counts()
    single_labels = vc[vc == 1].index
    single_df = df[df['label'].isin(single_labels)]
    multi_df = df[~df['label'].isin(single_labels)]
    if len(multi_df) > 0:
        train_multi, dev_multi = train_test_split(
            multi_df, test_size=0.1, random_state=42, stratify=multi_df['label']
        )
    else:
        train_multi, dev_multi = pd.DataFrame(), pd.DataFrame()
    train_df = pd.concat([train_multi, single_df]).reset_index(drop=True)
    dev_df = dev_multi.reset_index(drop=True)

# 输出标签分布
print('Train标签分布:')
print(train_df[args.label_col].value_counts())
print('Dev标签分布:')
print(dev_df[args.label_col].value_counts())

# 保存
train_df[[args.text_col, 'label']].to_csv(args.train_out, index=False)
dev_df[[args.text_col, 'label']].to_csv(args.dev_out, index=False)
print(f'Train集保存到: {args.train_out}')
print(f'Dev集保存到: {args.dev_out}')

# 保存id2label映射
id2label_path = args.train_out.replace('.csv', '_id2label.json')
with open(id2label_path, 'w', encoding='utf-8') as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)
print(f'id2label映射已保存到: {id2label_path}')
