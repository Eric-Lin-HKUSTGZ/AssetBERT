
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
# import numpy as np
# import json

# # 创建带时间戳的输出目录
# current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
# output_dir_with_time = os.path.join("model_output/multi_inputs", current_time)
# os.makedirs(output_dir_with_time, exist_ok=True)

# # 解析参数
# parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
# parser.add_argument('--train_file', type=str, default='./datasets/0717/multi_inputs_train.csv', help='Path to train CSV file')
# parser.add_argument('--dev_file', type=str, default='./datasets/0717/multi_inputs_dev.csv', help='Path to dev CSV file')
# parser.add_argument('--model_name', type=str, default='uer/roberta-base-finetuned-chinanews-chinese', help='Base model name')
# parser.add_argument('--output_dir', type=str, default=output_dir_with_time, help='Output directory for model')
# parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
# parser.add_argument('--batch_size', type=int, default=24, help='Batch size per device')
# parser.add_argument('--balance', action='store_true', help='Balance the training set')
# parser.add_argument('--min_count', type=int, default=10, help='Minimum samples per class when balancing')
# parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length for tokenizer')
# args = parser.parse_args()

# # 设置logger
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

# # 检查必要的列是否存在
# required_columns = ['资产名称', '型号', '用途', '新国标分类', '使用部门']
# for df in [df_train, df_dev]:
#     missing_cols = [col for col in required_columns if col not in df.columns]
#     if missing_cols:
#         logger.error(f"数据集缺少必要的列: {missing_cols}")
#         raise ValueError(f"数据集缺少必要的列: {missing_cols}")

# # 数据预处理：拼接多输入字段
# def combine_features(row):
#     return f"资产名称: {str(row['资产名称'])} 型号: {str(row['型号'])} 用途: {str(row['用途'])} 使用部门: {str(row['使用部门'])}"

# df_train['combined_text'] = df_train.apply(combine_features, axis=1)
# df_dev['combined_text'] = df_dev.apply(combine_features, axis=1)

# # 标签平衡（可选）
# if args.balance:
#     vc = df_train['新国标分类'].value_counts()
#     valid_labels = vc[vc >= args.min_count].index
#     df_train = df_train[df_train['新国标分类'].isin(valid_labels)]
#     balanced_df = []
#     for label, group in df_train.groupby('新国标分类'):
#         balanced_group = resample(group, replace=False, n_samples=args.min_count, random_state=42)
#         balanced_df.append(balanced_group)
#     df_train = pd.concat(balanced_df).reset_index(drop=True)
#     logger.info(f'已对训练集进行标签平衡，每类样本数: {args.min_count}')


# # 加载预定义的标签映射
# train_id2label_path = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json"
# if os.path.exists(train_id2label_path):
#     with open(train_id2label_path, 'r', encoding='utf-8') as f:
#         train_id2label = json.load(f)
#     logger.info(f"加载预定义标签映射，共 {len(train_id2label)} 个类别")
    
#     # 直接使用预定义标签映射的类别数
#     num_labels = len(train_id2label)
#     logger.info(f"使用预定义标签映射，共 {num_labels} 个类别")


# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModelForSequenceClassification.from_pretrained(
#     args.model_name,
#     num_labels=num_labels,
#     ignore_mismatched_sizes=True
# )

# # 转为Dataset
# train_dataset = Dataset.from_pandas(df_train[['combined_text' , '新国标分类']])
# val_dataset = Dataset.from_pandas(df_dev[['combined_text', '新国标分类']])

# # 预处理函数
# def preprocess(examples):
#     examples['label'] = [int(label) for label in examples['新国标分类']]
#     tokenized = tokenizer(
#         examples['combined_text'],
#         truncation=True,
#         padding='max_length',
#         max_length=args.max_length
#     )
#     return {**tokenized, 'label': examples['label']}

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

#     # Top5 Accuracy
#     top5_preds = logits.argsort(axis=-1)[:, -5:][:, ::-1]
#     top5_correct = [label in top5 for label, top5 in zip(labels, top5_preds)]
#     top5_acc = sum(top5_correct) / len(top5_correct)

#     logger.info(f"Eval accuracy: {acc:.4f}, Top5 accuracy: {top5_acc:.4f}")

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

# # 保存模型和分词器
# model.save_pretrained(args.output_dir)
# tokenizer.save_pretrained(args.output_dir)
# logger.info(f"模型和分词器已保存到: {args.output_dir}")


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
import numpy as np
import json

# 创建带时间戳的输出目录
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_dir_with_time = os.path.join("model_output/multi_inputs", current_time)
os.makedirs(output_dir_with_time, exist_ok=True)

# 解析参数
parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
parser.add_argument('--train_file', type=str, default='./datasets/0717/multi_inputs_train.csv', help='Path to train CSV file')
parser.add_argument('--dev_file', type=str, default='./datasets/0717/multi_inputs_dev.csv', help='Path to dev CSV file')
parser.add_argument('--model_name', type=str, default='uer/roberta-base-finetuned-chinanews-chinese', help='Base model name')
parser.add_argument('--output_dir', type=str, default=output_dir_with_time, help='Output directory for model')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
parser.add_argument('--balance', action='store_true', help='Balance the training set')
parser.add_argument('--min_count', type=int, default=10, help='Minimum samples per class when balancing')
parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length for tokenizer')
args = parser.parse_args()

# 设置logger
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

# 检查必要的列是否存在
required_columns = ['资产名称', '型号', '用途', '新国标分类', '使用部门']
for df in [df_train, df_dev]:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"数据集缺少必要的列: {missing_cols}")
        raise ValueError(f"数据集缺少必要的列: {missing_cols}")

# 数据预处理：拼接多输入字段
def combine_features(row):
    return f"资产名称: {str(row['资产名称'])} 型号: {str(row['型号'])} 用途: {str(row['用途'])} 使用部门: {str(row['使用部门'])}"

df_train['combined_text'] = df_train.apply(combine_features, axis=1)
df_dev['combined_text'] = df_dev.apply(combine_features, axis=1)

# 标签平衡（可选）
if args.balance:
    vc = df_train['新国标分类'].value_counts()
    valid_labels = vc[vc >= args.min_count].index
    df_train = df_train[df_train['新国标分类'].isin(valid_labels)]
    balanced_df = []
    for label, group in df_train.groupby('新国标分类'):
        balanced_group = resample(group, replace=False, n_samples=args.min_count, random_state=42)
        balanced_df.append(balanced_group)
    df_train = pd.concat(balanced_df).reset_index(drop=True)
    logger.info(f'已对训练集进行标签平衡，每类样本数: {args.min_count}')


# 加载预定义的标签映射
train_id2label_path = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json"
if os.path.exists(train_id2label_path):
    with open(train_id2label_path, 'r', encoding='utf-8') as f:
        train_id2label = json.load(f)
    logger.info(f"加载预定义标签映射，共 {len(train_id2label)} 个类别")
    
    # 直接使用预定义标签映射的类别数
    num_labels = len(train_id2label)
    logger.info(f"使用预定义标签映射，共 {num_labels} 个类别")


# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# 转为Dataset
train_dataset = Dataset.from_pandas(df_train[['combined_text' , '新国标分类']])
val_dataset = Dataset.from_pandas(df_dev[['combined_text', '新国标分类']])

# 预处理函数
def preprocess(examples):
    examples['label'] = [int(label) for label in examples['新国标分类']]
    tokenized = tokenizer(
        examples['combined_text'],
        truncation=True,
        padding='max_length',
        max_length=args.max_length
    )
    return {**tokenized, 'label': examples['label']}

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# 只保留必要字段
columns = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

# 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)

    # Top3 Accuracy
    top3_preds = logits.argsort(axis=-1)[:, -3:][:, ::-1]
    top3_correct = [label in top3 for label, top3 in zip(labels, top3_preds)]
    top3_acc = sum(top3_correct) / len(top3_correct)

    logger.info(f"Eval accuracy: {acc:.4f}, Top3 accuracy: {top3_acc:.4f}")

    return {'accuracy': acc, 'top3_accuracy': top3_acc}

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
    metric_for_best_model="top3_accuracy",
    report_to="none",
    fp16=torch.cuda.is_available(),
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
logger.info(f"验证集Top3准确率: {results['eval_top3_accuracy']:.4f}")
print(f"验证集准确率: {results['eval_accuracy']:.4f}")
print(f"验证集Top3准确率: {results['eval_top3_accuracy']:.4f}")

# 保存模型和分词器
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
logger.info(f"模型和分词器已保存到: {args.output_dir}")