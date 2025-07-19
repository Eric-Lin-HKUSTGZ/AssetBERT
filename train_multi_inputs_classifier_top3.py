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

# 定义自定义损失函数
def custom_top_k_loss(logits, labels, k=3):
    """
    自定义Top-K损失函数，对top3预测的错误结果进行惩罚
    Args:
        logits: 模型输出的logits [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        k: top-k的数量，默认为3
    Returns:
        loss: 自定义损失值
    """
    # 计算softmax概率
    probs = torch.softmax(logits, dim=-1)
    
    # 获取top-k预测
    topk_values, topk_indices = torch.topk(probs, k, dim=-1)
    
    # 检查真实标签是否在top-k中
    correct_mask = (labels.unsqueeze(-1) == topk_indices).any(dim=-1)
    
    # 计算标准交叉熵损失
    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    
    # 对不在top-k中的样本增加额外惩罚
    topk_penalty = torch.where(correct_mask, torch.zeros_like(ce_loss), ce_loss * 2.0)
    
    # 总损失 = 交叉熵损失 + top-k惩罚
    total_loss = ce_loss + topk_penalty
    
    return total_loss.mean()

# 定义自定义Trainer类
class CustomTopKTrainer(Trainer):
    """
    自定义Trainer类，使用Top-K损失函数
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = 3  # 默认使用top3
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写损失计算函数，使用自定义的Top-K损失
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 使用自定义的Top-K损失函数
        loss = custom_top_k_loss(logits, labels, k=self.k)
        
        # 调试信息：确保损失值正确
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：损失值为 {loss}，可能存在数值问题")
        
        return (loss, outputs) if return_outputs else loss

# 创建带时间戳的输出目录
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_dir_with_time = os.path.join("model_output/multi_inputs", current_time)
os.makedirs(output_dir_with_time, exist_ok=True)

# 解析参数
parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
parser.add_argument('--train_file', type=str, default='./datasets/0717/multi_inputs_train.csv', help='Path to train CSV file')
parser.add_argument('--dev_file', type=str, default='./datasets/0717/multi_inputs_dev.csv', help='Path to dev CSV file')
parser.add_argument('--model_name', type=str, default='hfl/chinese-macbert-base', help='Base model name')
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

trainer = CustomTopKTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

logger.info("使用自定义Top3损失函数进行训练...")
logger.info("损失函数将对top3预测的错误结果进行惩罚，提升top3预测精度")
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

# # 定义改进的损失函数
# def improved_loss(logits, labels, alpha=0.7, smoothing=0.1, k=3):
#     """
#     改进的损失函数，结合标签平滑和top-k损失
#     Args:
#         logits: 模型输出的logits [batch_size, num_classes]
#         labels: 真实标签 [batch_size]
#         alpha: 交叉熵损失和top-k损失的权重平衡参数，默认0.7
#         smoothing: 标签平滑参数，默认0.1
#         k: top-k的数量，默认为3
#     Returns:
#         loss: 组合损失值
#     """
#     # CrossEntropyLoss with Label Smoothing
#     num_classes = logits.size(-1)
#     one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
#     smooth_labels = one_hot * (1 - smoothing) + smoothing / num_classes
#     log_probs = torch.log_softmax(logits, dim=-1)
#     ce_loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

#     # Top-k loss with improved numerical stability
#     topk_probs = torch.softmax(logits, dim=-1)
#     topk_values, topk_indices = torch.topk(topk_probs, k, dim=-1)
#     correct_mask = (labels.unsqueeze(-1) == topk_indices).any(dim=-1)
    
#     # 使用更稳定的top-k损失计算
#     # 计算真实标签在top-k中的概率
#     true_probs = torch.gather(topk_probs, 1, labels.unsqueeze(1)).squeeze(1)
#     in_topk = correct_mask.float()
    
#     # 如果真实标签在top-k中，使用较小的损失；否则使用较大的损失
#     epsilon = 1e-8
#     # 确保所有操作都使用张量
#     epsilon_tensor = torch.tensor(epsilon, device=logits.device, dtype=logits.dtype)
#     topk_loss_per_sample = in_topk * (-torch.log(true_probs + epsilon_tensor)) + (1 - in_topk) * (-torch.log(epsilon_tensor))
#     topk_loss = topk_loss_per_sample.mean()

#     # Combined loss
#     total_loss = alpha * ce_loss + (1 - alpha) * topk_loss
    
#     return total_loss

# # 定义自定义Trainer类
# class CustomTopKTrainer(Trainer):
#     """
#     自定义Trainer类，使用改进的损失函数（标签平滑 + top-k损失）
#     """
#     def __init__(self, *args, alpha=0.7, smoothing=0.1, k=3, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = alpha  # 交叉熵损失权重
#         self.smoothing = smoothing  # 标签平滑参数
#         self.k = k  # top-k数量
    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         """
#         重写损失计算函数，使用改进的损失函数
#         """
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
        
#         # 使用改进的损失函数
#         loss = improved_loss(logits, labels, alpha=self.alpha, smoothing=self.smoothing, k=self.k)
        
#         # 调试信息：确保损失值正确
#         if torch.isnan(loss) or torch.isinf(loss):
#             print(f"警告：损失值为 {loss}，可能存在数值问题")
        
#         return (loss, outputs) if return_outputs else loss

# # 创建带时间戳的输出目录
# current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
# output_dir_with_time = os.path.join("model_output/multi_inputs", current_time)
# os.makedirs(output_dir_with_time, exist_ok=True)

# # 解析参数
# parser = argparse.ArgumentParser(description='Train a classifier using Hugging Face Transformers.')
# parser.add_argument('--train_file', type=str, default='./datasets/0717/multi_inputs_train.csv', help='Path to train CSV file')
# parser.add_argument('--dev_file', type=str, default='./datasets/0717/multi_inputs_dev.csv', help='Path to dev CSV file')
# parser.add_argument('--model_name', type=str, default='KoichiYasuoka/deberta-base-chinese', help='Base model name')
# parser.add_argument('--output_dir', type=str, default=output_dir_with_time, help='Output directory for model')
# parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
# parser.add_argument('--balance', action='store_true', help='Balance the training set')
# parser.add_argument('--min_count', type=int, default=10, help='Minimum samples per class when balancing')
# parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length for tokenizer')
# parser.add_argument('--alpha', type=float, default=1.0, help='Weight for cross-entropy loss in combined loss function')
# parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing parameter')
# parser.add_argument('--top_k', type=int, default=3, help='Top-k value for top-k loss')
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

#     # Top3 Accuracy
#     top3_preds = logits.argsort(axis=-1)[:, -3:][:, ::-1]
#     top3_correct = [label in top3 for label, top3 in zip(labels, top3_preds)]
#     top3_acc = sum(top3_correct) / len(top3_correct)

#     logger.info(f"Eval accuracy: {acc:.4f}, Top3 accuracy: {top3_acc:.4f}")

#     return {'accuracy': acc, 'top3_accuracy': top3_acc}

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
#     metric_for_best_model="top3_accuracy",
#     report_to="none",
#     fp16=torch.cuda.is_available(),
# )

# trainer = CustomTopKTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
#     alpha=args.alpha,  # 交叉熵损失权重
#     smoothing=args.smoothing,  # 标签平滑参数
#     k=args.top_k  # top-k数量
# )

# logger.info("使用改进的损失函数进行训练...")
# logger.info(f"损失函数配置: alpha={args.alpha} (交叉熵权重), smoothing={args.smoothing} (标签平滑), k={args.top_k} (top-k)")
# logger.info("结合标签平滑和top-k损失，提升模型泛化能力和top3预测精度")
# logger.info("开始训练...")
# trainer.train()

# # 验证集评估
# logger.info("验证集评估...")
# results = trainer.evaluate()
# logger.info(f"验证集准确率: {results['eval_accuracy']:.4f}")
# logger.info(f"验证集Top3准确率: {results['eval_top3_accuracy']:.4f}")
# print(f"验证集准确率: {results['eval_accuracy']:.4f}")
# print(f"验证集Top3准确率: {results['eval_top3_accuracy']:.4f}")

# # 保存模型和分词器
# model.save_pretrained(args.output_dir)
# tokenizer.save_pretrained(args.output_dir)
# logger.info(f"模型和分词器已保存到: {args.output_dir}")