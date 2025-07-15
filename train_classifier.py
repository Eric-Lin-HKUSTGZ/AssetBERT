# # python3 新国标分类/train_classifier.py --min_count 10 --epochs 5 --batch_size 32 --output_dir ./output
# import os
# from datetime import datetime
# import pandas as pd
# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score
# import argparse
# from sklearn.utils import resample
# import logging
# import os
# from collections import defaultdict

# # 创建带时间戳的输出目录
# current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
# output_dir_with_time = os.path.join("model_output/single_input", current_time)
# os.makedirs(output_dir_with_time, exist_ok=True)

# # 在解析参数前设置输出目录
# parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
# # parser.add_argument('--train_file', type=str, default='./datasets/single_char_data/single_char_train.csv', help='Path to train CSV file')
# # parser.add_argument('--dev_file', type=str, default='./datasets/single_char_data/single_char_dev.csv', help='Path to dev CSV file')
# parser.add_argument('--train_file', type=str, default='./datasets/multi_chars_data/multi_inputs_train.csv', help='Path to train CSV file')
# parser.add_argument('--dev_file', type=str, default='./datasets/multi_chars_data/multi_inputs_dev.csv', help='Path to dev CSV file')
# parser.add_argument('--model_name', type=str, default='uer/roberta-base-finetuned-chinanews-chinese', help='Base model name')
# parser.add_argument('--output_dir', type=str, default=output_dir_with_time, help='Output directory for model')  # 修改默认值
# parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
# parser.add_argument('--balance', action='store_true', help='Balance the training set')
# parser.add_argument('--min_count', type=int, default=10, help='Minimum samples per class when balancing')
# args = parser.parse_args()

# # 设置logger（保持不变，但会自动使用新的输出目录）
# os.makedirs(args.output_dir, exist_ok=True)
# log_file = os.path.join(args.output_dir, 'train.log')

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(message)s',
#     handlers=[
#         logging.FileHandler(log_file, mode='w', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# logger.info(f"参数: {args}")

# # 读取数据
# df_train = pd.read_csv(args.train_file)
# df_dev = pd.read_csv(args.dev_file)

# # 只对训练集做标签平衡
# if args.balance:
#     vc = df_train['label'].value_counts()
#     valid_labels = vc[vc >= args.min_count].index
#     df_train = df_train[df_train['label'].isin(valid_labels)]
#     balanced_df = []
#     for label, group in df_train.groupby('label'):
#         balanced_group = resample(group, replace=False, n_samples=args.min_count, random_state=42)
#         balanced_df.append(balanced_group)
#     df_train = pd.concat(balanced_df).reset_index(drop=True)
#     logger.info(f'已对训练集进行标签平衡，每类样本数: {args.min_count}（仅保留样本数大于等于{args.min_count}的类别）')

# # 自动检测类别数
# num_labels = df_train['label'].nunique()
# logger.info(f"训练集类别数: {num_labels}")

# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModelForSequenceClassification.from_pretrained(
#     args.model_name,
#     num_labels=num_labels,
#     ignore_mismatched_sizes=True
# )

# def preprocess(examples):
#     return tokenizer(examples['资产名称'], truncation=True, padding='max_length', max_length=32)

# # 转为Hugging Face Dataset
# train_dataset = Dataset.from_pandas(df_train)
# val_dataset = Dataset.from_pandas(df_dev)
# train_dataset = train_dataset.map(preprocess, batched=True)
# val_dataset = val_dataset.map(preprocess, batched=True)

# # 只保留必要字段
# columns = ['input_ids', 'attention_mask', 'label']
# train_dataset.set_format(type='torch', columns=columns)
# val_dataset.set_format(type='torch', columns=columns)

# # 评估函数
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=-1)
#     acc = accuracy_score(labels, preds)

#     # top5
#     top5_preds = logits.argsort(axis=-1)[:, -5:][:, ::-1]  # shape: (batch, 5)
#     top5_correct = [label in top5 for label, top5 in zip(labels, top5_preds)]
#     top5_acc = sum(top5_correct) / len(top5_correct)

#     logger.info(f"Eval accuracy: {acc:.4f}, Top5 accuracy: {top5_acc:.4f}")

#     # 记录top5标签
#     compute_metrics.top5_preds = top5_preds
#     compute_metrics.labels = labels

#     return {'accuracy': acc, 'top5_accuracy': top5_acc}

# # 训练参数
# training_args = TrainingArguments(
#     output_dir=args.output_dir,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=1e-4,
#     per_device_train_batch_size=args.batch_size,
#     per_device_eval_batch_size=args.batch_size,
#     num_train_epochs=args.epochs,
#     weight_decay=0.01,
#     logging_dir=f'{args.output_dir}/logs',
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     report_to="none",
#     fp16=torch.cuda.is_available(),
#     logging_strategy="steps",
#     log_level="info",
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
# )

# logger.info("开始训练...")
# trainer.train()

# # 验证集评估
# logger.info("验证集评估...")
# results = trainer.evaluate()
# logger.info(f"验证集准确率: {results['eval_accuracy']:.4f}")
# logger.info(f"验证集Top5准确率: {results['eval_top5_accuracy']:.4f}")
# print(f"验证集准确率: {results['eval_accuracy']:.4f}")
# print(f"验证集Top5准确率: {results['eval_top5_accuracy']:.4f}")

# # 保存top5预测结果
# id2label_path = args.train_file.replace('train.csv', 'train_id2label.json')
# if os.path.exists(id2label_path):
#     import json
#     with open(id2label_path, 'r', encoding='utf-8') as f:
#         id2label = json.load(f)
# else:
#     # fallback: 用label本身
#     id2label = {str(i): str(i) for i in range(num_labels)}

# # 资产名称、真实标签、top5预测标签
# asset_names = df_dev['资产名称'].tolist()
# true_labels = df_dev['label'].tolist()
# top5_preds = compute_metrics.top5_preds
# top5_labels = [[id2label[str(idx)] for idx in row] for row in top5_preds]

# top5_df = pd.DataFrame({
#     '资产名称': asset_names,
#     '真实标签': [id2label[str(l)] for l in true_labels],
#     'top5预测标签': ["|".join(row) for row in top5_labels]
# })
# top5_df.to_csv(os.path.join(args.output_dir, 'dev_top5_preds.csv'), index=False)
# logger.info(f"top5预测结果已保存到: {os.path.join(args.output_dir, 'dev_top5_preds.csv')}")

# # 保存分词器和模型
# # model.save_pretrained(args.output_dir)
# # tokenizer.save_pretrained(args.output_dir)
# # logger.info(f"模型和分词器已保存到: {args.output_dir}")

# python3 新国标分类/train_classifier.py --min_count 10 --epochs 5 --batch_size 32 --output_dir ./output
import os
from datetime import datetime
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import argparse
from sklearn.utils import resample
import logging
import os
from collections import defaultdict

# 创建带时间戳的输出目录
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_dir_with_time = os.path.join("model_output/single_input", current_time)
os.makedirs(output_dir_with_time, exist_ok=True)

# 在解析参数前设置输出目录
parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
# parser.add_argument('--train_file', type=str, default='./datasets/single_char_data/single_char_train.csv', help='Path to train CSV file')
# parser.add_argument('--dev_file', type=str, default='./datasets/single_char_data/single_char_dev.csv', help='Path to dev CSV file')
parser.add_argument('--train_file', type=str, default='./datasets/multi_chars_data/multi_inputs_train.csv', help='Path to train CSV file')
parser.add_argument('--dev_file', type=str, default='./datasets/multi_chars_data/multi_inputs_dev.csv', help='Path to dev CSV file')
parser.add_argument('--model_name', type=str, default='uer/roberta-base-finetuned-chinanews-chinese', help='Base model name')
parser.add_argument('--output_dir', type=str, default=output_dir_with_time, help='Output directory for model')  # 修改默认值
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
parser.add_argument('--balance', action='store_true', help='Balance the training set')
parser.add_argument('--min_count', type=int, default=10, help='Minimum samples per class when balancing')
args = parser.parse_args()

# 设置logger（保持不变，但会自动使用新的输出目录）
os.makedirs(args.output_dir, exist_ok=True)
log_file = os.path.join(args.output_dir, 'train.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"参数: {args}")

# 读取数据
df_train = pd.read_csv(args.train_file)
df_dev = pd.read_csv(args.dev_file)

# 检查数据集列名
logger.info(f"训练集列名: {df_train.columns.tolist()}")
logger.info(f"验证集列名: {df_dev.columns.tolist()}")

# 确定标签列名
label_column = '新国标分类' if '新国标分类' in df_train.columns else 'label'
logger.info(f"使用标签列: {label_column}")

# 只对训练集做标签平衡
if args.balance:
    vc = df_train[label_column].value_counts()
    valid_labels = vc[vc >= args.min_count].index
    df_train = df_train[df_train[label_column].isin(valid_labels)]
    balanced_df = []
    for label, group in df_train.groupby(label_column):
        balanced_group = resample(group, replace=False, n_samples=args.min_count, random_state=42)
        balanced_df.append(balanced_group)
    df_train = pd.concat(balanced_df).reset_index(drop=True)
    logger.info(f'已对训练集进行标签平衡，每类样本数: {args.min_count}（仅保留样本数大于等于{args.min_count}的类别）')

# 自动检测类别数
num_labels = df_train[label_column].nunique()
logger.info(f"训练集类别数: {num_labels}")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# 数据预处理：拼接多输入字段
def combine_features(row):
    """将多个输入字段拼接成一个文本"""
    # 确保所有字段都是字符串
    asset_name = str(row['资产名称'])
    
    return f"资产名称: {asset_name}"

# 应用拼接函数
df_train['combined_text'] = df_train.apply(combine_features, axis=1)
df_dev['combined_text'] = df_dev.apply(combine_features, axis=1)

def preprocess(examples):
    return tokenizer(examples['combined_text'], truncation=True, padding='max_length', max_length=64)

# 转为Hugging Face Dataset
train_dataset = Dataset.from_pandas(df_train[['combined_text', label_column]])
val_dataset = Dataset.from_pandas(df_dev[['combined_text', label_column]])

# 创建标签映射（基于训练集）
unique_labels = df_train[label_column].unique().tolist()  # 转换为Python列表
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
logger.info(f"标签映射: {label_mapping}")
logger.info(f"训练集标签范围: 0-{len(label_mapping)-1}")

# 预处理函数：处理拼接后的文本
def preprocess_with_labels(examples):
    # 将标签转换为数值
    examples['label'] = [label_mapping.get(label, -1) for label in examples[label_column]]
    
    # 分词处理
    tokenized = tokenizer(
        examples['combined_text'], 
        truncation=True, 
        padding='max_length', 
        max_length=64
    )
    
    return {**tokenized, 'label': examples['label']}

train_dataset = train_dataset.map(preprocess_with_labels, batched=True)
val_dataset = val_dataset.map(preprocess_with_labels, batched=True)

# 过滤掉验证集中训练集没有的标签（label为-1的样本）
def filter_valid_labels(example):
    return example['label'] != -1

val_dataset = val_dataset.filter(filter_valid_labels)
logger.info(f"过滤后验证集样本数: {len(val_dataset)}")

# 只保留必要字段
columns = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

# 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    # 过滤掉无效标签（-1）
    valid_mask = labels != -1
    if valid_mask.sum() == 0:
        logger.warning("验证集中没有有效的标签")
        return {'accuracy': 0.0, 'top5_accuracy': 0.0}
    
    valid_labels = labels[valid_mask]
    valid_preds = preds[valid_mask]
    valid_logits = logits[valid_mask]
    
    acc = accuracy_score(valid_labels, valid_preds)

    # top5
    top5_preds = valid_logits.argsort(axis=-1)[:, -5:][:, ::-1]  # shape: (batch, 5)
    top5_correct = [label in top5 for label, top5 in zip(valid_labels, top5_preds)]
    top5_acc = sum(top5_correct) / len(top5_correct)

    logger.info(f"Eval accuracy: {acc:.4f}, Top5 accuracy: {top5_acc:.4f}")

    # 记录top5标签
    compute_metrics.top5_preds = top5_preds
    compute_metrics.labels = valid_labels

    return {'accuracy': acc, 'top5_accuracy': top5_acc}

# 训练参数
training_args = TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    logging_dir=f'{args.output_dir}/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    log_level="info",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

logger.info("开始训练...")
trainer.train()

# 验证集评估
logger.info("验证集评估...")
results = trainer.evaluate()
logger.info(f"验证集准确率: {results['eval_accuracy']:.4f}")
logger.info(f"验证集Top5准确率: {results['eval_top5_accuracy']:.4f}")
print(f"验证集准确率: {results['eval_accuracy']:.4f}")
print(f"验证集Top5准确率: {results['eval_top5_accuracy']:.4f}")

# 创建id2label映射
id2label = {str(int(idx)): label for label, idx in label_mapping.items()}

# 保存标签映射
id2label_path = os.path.join(args.output_dir, 'id2label.json')
import json
with open(id2label_path, 'w', encoding='utf-8') as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)
logger.info(f"标签映射已保存到: {id2label_path}")

# 资产名称、真实标签、top5预测标签
asset_names = df_dev['资产名称'].tolist()
models = df_dev['型号'].tolist()
departments = df_dev['用途'].tolist() if '用途' in df_dev.columns else df_dev['使用部门'].tolist()

# 获取验证集中有效的样本索引（训练集有的标签）
valid_dev_indices = []
for i, label in enumerate(df_dev[label_column].tolist()):
    if label in label_mapping:
        valid_dev_indices.append(i)

# 使用评估函数中记录的标签和预测
true_labels = compute_metrics.labels.tolist() if hasattr(compute_metrics, 'labels') else []
top5_preds = compute_metrics.top5_preds.tolist() if hasattr(compute_metrics, 'top5_preds') else []

# 调试信息
logger.info(f"true_labels 长度: {len(true_labels)}")
logger.info(f"top5_preds 长度: {len(top5_preds)}")
if len(true_labels) > 0:
    logger.info(f"true_labels 范围: {min(true_labels)} - {max(true_labels)}")

# 确保所有元素都是字符串
top5_labels = []
for row in top5_preds:
    row_labels = []
    for idx in row:
        try:
            label = id2label[str(int(idx))]
            row_labels.append(str(label))  # 确保是字符串
        except (KeyError, ValueError) as e:
            logger.warning(f"无法转换索引 {idx} (类型: {type(idx)}) 为标签: {e}")
            row_labels.append("未知")
    top5_labels.append(row_labels)

# 只包含有效的验证集样本
valid_asset_names = [asset_names[i] for i in valid_dev_indices]
valid_models = [models[i] for i in valid_dev_indices]
valid_departments = [departments[i] for i in valid_dev_indices]

# 处理真实标签
true_label_names = []
for l in true_labels:
    try:
        label_name = id2label[str(int(l))]
        true_label_names.append(str(label_name))  # 确保是字符串
    except (KeyError, ValueError) as e:
        logger.warning(f"无法转换真实标签 {l} (类型: {type(l)}) 为标签名: {e}")
        true_label_names.append("未知")

top5_df = pd.DataFrame({
    '资产名称': valid_asset_names,
    '型号': valid_models,
    '用途': valid_departments,
    '真实标签': true_label_names,
    'top5预测标签': ["|".join(row) for row in top5_labels]
})
top5_df.to_csv(os.path.join(args.output_dir, 'dev_top5_preds.csv'), index=False)
logger.info(f"top5预测结果已保存到: {os.path.join(args.output_dir, 'dev_top5_preds.csv')}")



 
 