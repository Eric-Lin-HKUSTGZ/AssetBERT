import argparse
import json
import os
import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────── 配置日志 ───────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────── 参数解析 ───────────
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='./datasets/0717/multi_inputs_train.csv')
parser.add_argument('--val_data', type=str, default='./datasets/0717/multi_inputs_dev.csv')
parser.add_argument('--model_dirs', nargs='+', type=str, required=True)
parser.add_argument('--tokenizer_name', type=str, default=None)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--extract_batch_size', type=int, default=8, help='特征提取时的批次大小')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_path', type=str, default='/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/stacking/meta_model.pth')
args = parser.parse_args()

# ─────────── 1. 加载标签映射 ───────────
train_id2label_path = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json"
if os.path.exists(train_id2label_path):
    with open(train_id2label_path, 'r', encoding='utf-8') as f:
        train_id2label = json.load(f)
    logger.info(f"加载预定义标签映射，共 {len(train_id2label)} 个类别")
    
    # 直接使用预定义标签映射的类别数
    num_labels = len(train_id2label)
    logger.info(f"使用预定义标签映射，共 {num_labels} 个类别")
else:
    logger.error(f"预定义标签映射文件不存在: {train_id2label_path}")
    raise FileNotFoundError(f"必须提供标签映射文件: {train_id2label_path}")

# ─────────── 2. 数据集类 ───────────
class StackingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 调试：检查数据集中的标签分布
        train_labels = set(df['新国标分类'].unique())
        logger.info(f"数据集标签数量: {len(train_labels)}")
        logger.info(f"数据集标签示例: {list(train_labels)[:10]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"资产名称: {row['资产名称']} 型号: {row['型号']} 用途: {row['用途']} 使用部门: {row['使用部门']}"
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # 直接使用整数标签，与train_multi_inputs_classifier.py保持一致
        label = int(row['新国标分类'])
        return inputs, label

# ─────────── 3. 加载模型 ───────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")

# 加载 tokenizer
tok_name = args.tokenizer_name or args.model_dirs[0]
tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

# 加载多个模型
models = []
for model_dir in args.model_dirs:
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    models.append(model)
logger.info(f"加载 {len(models)} 个模型成功")


# ─────────── 4. 元模型定义 ───────────
class MetaModel(nn.Module):
    def __init__(self, num_models, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_models * num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

meta_model = MetaModel(len(models), num_labels).to(device)
optimizer = optim.Adam(meta_model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# ─────────── 5. 提取模型输出特征（logits） ───────────
def get_model_features(dataloader):
    all_features = []
    all_labels = []
    
    # 使用指定的批次大小进行特征提取
    extract_batch_size = args.extract_batch_size
    
    with torch.no_grad():
        for batch_idx, (inputs, label) in enumerate(tqdm(dataloader, desc="提取特征")):
            # 分批处理，避免内存溢出
            batch_size = inputs['input_ids'].size(0)
            batch_features = []
            
            for i in range(0, batch_size, extract_batch_size):
                end_idx = min(i + extract_batch_size, batch_size)
                
                # 提取当前小批次
                batch_inputs = {k: v[i:end_idx].to(device) for k, v in inputs.items()}
                batch_label = label[i:end_idx]
                
                # 处理所有模型的预测
                model_logits = []
                for model in models:
                    try:
                        with torch.cuda.amp.autocast():
                            logits = model(**batch_inputs).logits
                        model_logits.append(logits)
                    except RuntimeError as e:
                        logger.error(f"模型预测失败: {e}")
                        # 如果单个模型失败，使用零张量填充
                        model_logits.append(torch.zeros(end_idx - i, num_labels, device=device))
                
                # 拼接所有模型的logits
                combined = torch.cat(model_logits, dim=1)  # [B, M * C]
                batch_features.append(combined.cpu())
                
                # 清理GPU内存
                del batch_inputs, model_logits, combined
                torch.cuda.empty_cache()
            
            # 合并当前批次的所有特征
            if batch_features:
                batch_combined = torch.cat(batch_features, dim=0)
                all_features.append(batch_combined)
                all_labels.append(label)
                
                # 定期清理内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
    
    # 最终拼接所有特征
    if not all_features:
        raise ValueError("没有提取到任何特征！")
    
    final_features = torch.cat(all_features, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"特征提取完成 - 特征形状: {final_features.shape}, 标签形状: {final_labels.shape}")
    return final_features, final_labels

# ─────────── 6. 训练函数 ───────────
def train_epoch(features, labels):
    meta_model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 使用较小的训练批次大小
    train_batch_size = min(args.batch_size, 32)
    
    # 随机打乱数据
    indices = torch.randperm(len(features))
    
    # 使用tqdm显示训练进度
    for i in tqdm(range(0, len(features), train_batch_size), desc="训练中"):
        batch_indices = indices[i:i+train_batch_size]
        batch_x = features[batch_indices].to(device)
        batch_y = labels[batch_indices].to(device)

        optimizer.zero_grad()
        
        try:
            with torch.cuda.amp.autocast():
                outputs = meta_model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        except RuntimeError as e:
            logger.error(f"训练批次失败: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch Loss: {avg_loss:.4f}")
    return avg_loss

# ─────────── 7. 评估函数 ───────────
def evaluate(features, labels):
    meta_model.eval()
    total_correct = 0
    total_samples = 0
    
    eval_batch_size = min(args.batch_size, 32)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(features), eval_batch_size), desc="评估中"):
            batch_x = features[i:i+eval_batch_size].to(device)
            batch_y = labels[i:i+eval_batch_size]
            
            try:
                with torch.cuda.amp.autocast():
                    outputs = meta_model(batch_x)
                
                preds = outputs.argmax(dim=1).cpu()
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)
                
            except RuntimeError as e:
                logger.error(f"评估批次失败: {e}")
                continue
    
    acc = total_correct / max(total_samples, 1)
    logger.info(f"Validation Accuracy: {acc:.4f}")
    return acc

# ─────────── 8. 加载训练集和验证集 ───────────
def load_data(path):
    df = pd.read_csv(path)
    required_columns = ['资产名称', '型号', '用途', '新国标分类', '使用部门']
    assert all(col in df.columns for col in required_columns), "列缺失"
    return df

train_df = load_data(args.train_data)
val_df = load_data(args.val_data)

train_dataset = StackingDataset(train_df, tokenizer, args.max_length)
val_dataset = StackingDataset(val_df, tokenizer, args.max_length)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

logger.info("提取训练集特征...")
train_features, train_labels = get_model_features(train_loader)
logger.info("提取验证集特征...")
val_features, val_labels = get_model_features(val_loader)

# ─────────── 9. 开始训练 ───────────
logger.info("开始训练元模型...")
logger.info(f"训练样本数: {len(train_features)}")
logger.info(f"验证样本数: {len(val_features)}")

# 监控GPU内存使用
if torch.cuda.is_available():
    logger.info(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

for epoch in range(args.epochs):
    logger.info(f"Epoch {epoch + 1}/{args.epochs}")
    
    # 训练
    train_loss = train_epoch(train_features, train_labels)
    
    # 验证
    val_acc = evaluate(val_features, val_labels)
    
    # 监控GPU内存使用
    if torch.cuda.is_available():
        logger.info(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 清理GPU内存
    torch.cuda.empty_cache()

logger.info("训练完成")

# ─────────── 10. 保存元模型 ───────────
torch.save(meta_model.state_dict(), args.save_path)
logger.info(f"元模型已保存到 {args.save_path}")




