#!/usr/bin/env python
# train_stacking.py —— 使用多个模型的输出训练一个元模型（Meta Model）

import os
# 设置环境变量，避免tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import gc
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────── 配置日志 ───────────
def setup_logging(save_path):
    """设置日志配置，将日志保存到与模型相同的目录"""
    # 获取保存目录
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成日志文件名
    log_filename = os.path.join(save_dir, 'training_linear.log')
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 清除现有的处理器
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 注意：这里先不调用setup_logging，等解析参数后再调用
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────── 参数解析 ───────────
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--val_data', type=str, required=True)
parser.add_argument('--model_dirs', nargs='+', type=str, required=True)
parser.add_argument('--tokenizer_name', type=str, default=None)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=16)  # 减小默认批次大小
parser.add_argument('--epochs', type=int, default=20)      # 增加epoch数量
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--max_train_samples', type=int, default=100000, help='最大训练样本数')
parser.add_argument('--max_val_samples', type=int, default=20000, help='最大验证样本数')
parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
parser.add_argument('--min_delta', type=float, default=0.001, help='早停最小改进值')
# RWKV模型参数
parser.add_argument('--rwkv_d_model', type=int, default=256, help='RWKV模型维度')
parser.add_argument('--rwkv_layers', type=int, default=4, help='RWKV层数')
parser.add_argument('--rwkv_dropout', type=float, default=0.2, help='RWKV dropout率')
parser.add_argument('--model_type', type=str, default='linear', choices=['rwkv', 'linear', 'transformer'], help='元模型类型')
args = parser.parse_args()

# 监控内存使用
def log_memory_usage():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"GPU内存使用: {mem:.2f} GB (峰值: {max_mem:.2f} GB)")
    else:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"内存使用: {mem.used/1024**3:.2f}/{mem.total/1024**3:.2f} GB")

# ─────────── 1. 加载标签映射 ───────────
def load_label_map(path):
    if not os.path.exists(path):
        logger.error(f"标签映射文件不存在: {path}")
        raise FileNotFoundError(f"必须提供标签映射文件: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    
    logger.info(f"加载标签映射成功，共 {len(id2label)} 个类别")
    return id2label

train_id2label_path = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json"
id2label = load_label_map(train_id2label_path)
num_labels = len(id2label)

# ─────────── 2. 优化数据集类 ───────────
class EfficientStackingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, max_samples=None):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 调试：检查数据集中的标签分布
        train_labels = set(df['新国标分类'].unique())
        logger.info(f"数据集标签数量: {len(train_labels)}")
        logger.info(f"数据集标签示例: {list(train_labels)[:10]}")
        
        # 限制样本数量
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(max_samples, random_state=42)
            logger.info(f"限制数据集大小为 {max_samples} 个样本")
        
        logger.info(f"有效样本数: {len(self.df)}")
        log_memory_usage()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"资产名称: {row['资产名称']} 型号: {row['型号']} 用途: {row['用途']} 使用部门: {row['使用部门']}"
        # 直接使用整数标签，与train_multi_inputs_classifier.py保持一致
        label = int(row['新国标分类'])
        return text, label

# ─────────── 3. 设备设置 ───────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")

# ─────────── 4. 动态批处理函数 ───────────
def dynamic_batch_processing(texts, tokenizer, max_length, batch_size):
    """动态批处理文本数据"""
    inputs_batch = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        inputs_batch.append(inputs)
    return inputs_batch

# ─────────── 5. 加载tokenizer和模型 ───────────
def load_tokenizer(name):
    if os.path.exists(name):
        return AutoTokenizer.from_pretrained(
            name, 
            trust_remote_code=True,
            local_files_only=True
        )
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)

tok_name = args.tokenizer_name or args.model_dirs[0]
tokenizer = load_tokenizer(tok_name)
logger.info(f"加载tokenizer: {tokenizer.__class__.__name__}")

def load_model(path):
    try:
        if os.path.exists(path):
            model = AutoModelForSequenceClassification.from_pretrained(
                path, 
                local_files_only=True,
                trust_remote_code=True
            ).to(device).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                path, 
                trust_remote_code=True
            ).to(device).eval()
        
        # 减少内存占用
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {path}, 错误: {str(e)}")
        return None

models = [m for m in (load_model(path) for path in args.model_dirs) if m is not None]
if not models:
    raise RuntimeError("没有成功加载任何模型")
logger.info(f"加载 {len(models)} 个模型成功")
log_memory_usage()

# ─────────── 6. 多种元模型定义 ───────────
class SimpleLinearMetaModel(nn.Module):
    """简单的线性元模型，作为稳定基线"""
    def __init__(self, num_models, num_classes, dropout=0.2):
        super().__init__()
        input_dim = num_models * num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class TransformerMetaModel(nn.Module):
    """Transformer元模型"""
    def __init__(self, num_models, num_classes, d_model=256, num_layers=4, dropout=0.2):
        super().__init__()
        input_dim = num_models * num_classes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_models * num_classes]
        x = self.input_proj(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        x = self.transformer(x)  # [batch_size, 1, d_model]
        x = x.squeeze(1)  # [batch_size, d_model]
        x = self.output_proj(x)
        return x

class SimpleRWKVBlock(nn.Module):
    """简化的RWKV Block，更稳定的实现"""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # 简化的RWKV组件
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 权重参数
        self.r_weight = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.k_weight = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        # 第一个残差连接
        residual = x
        
        # Layer norm + RWKV attention
        x = self.ln1(x)
        r = torch.sigmoid(self.r_proj(x))
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 简化的注意力机制
        attn_out = r * (k * v).sum(dim=-1, keepdim=True) / (k.sum(dim=-1, keepdim=True) + 1e-8)
        x = residual + self.residual_weight * attn_out
        
        # 第二个残差连接
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.residual_weight * x
        
        return x

class StableRWKVMetaModel(nn.Module):
    """稳定的RWKV元模型"""
    def __init__(self, num_models, num_classes, d_model=256, num_layers=4, dropout=0.2):
        super().__init__()
        
        # Input projection with normalization
        input_dim = num_models * num_classes
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # RWKV layers
        self.rwkv_layers = nn.ModuleList([
            SimpleRWKVBlock(d_model, dropout=dropout) for _ in range(num_layers)
        ])
        
        # 额外的层归一化
        self.final_ln = nn.LayerNorm(d_model)
        
        # Output projection with skip connection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        # x shape: [batch_size, num_models * num_classes]
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, d_model]
        
        # Add sequence dimension for RWKV processing
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Process through RWKV layers
        for layer in self.rwkv_layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_ln(x)
        
        # Global pooling
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Output projection
        x = self.output_proj(x)
        
        return x

# 根据参数选择模型
if args.model_type == 'linear':
    meta_model = SimpleLinearMetaModel(len(models), num_labels, dropout=args.rwkv_dropout).to(device)
    logger.info(f"使用简单线性元模型")
elif args.model_type == 'transformer':
    meta_model = TransformerMetaModel(
        len(models), num_labels, 
        d_model=args.rwkv_d_model, 
        num_layers=args.rwkv_layers, 
        dropout=args.rwkv_dropout
    ).to(device)
    logger.info(f"使用Transformer元模型: d_model={args.rwkv_d_model}, layers={args.rwkv_layers}")
else:  # rwkv
    meta_model = StableRWKVMetaModel(
        len(models), 
        num_labels, 
        d_model=args.rwkv_d_model, 
        num_layers=args.rwkv_layers,
        dropout=args.rwkv_dropout
    ).to(device)
    logger.info(f"使用RWKV元模型: d_model={args.rwkv_d_model}, layers={args.rwkv_layers}, dropout={args.rwkv_dropout}")

# 使用AdamW优化器，带权重衰减
optimizer = optim.AdamW(
    meta_model.parameters(), 
    lr=args.lr, 
    weight_decay=1e-4,
    eps=1e-8,
    betas=(0.9, 0.999)
)

# 改进的学习率调度器
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    epochs=args.epochs,
    steps_per_epoch=100,  # 估计的每epoch步数
    pct_start=0.1,  # 10%的时间用于warmup
    anneal_strategy='cos',
    div_factor=25.0,  # 初始学习率 = max_lr / 25
    final_div_factor=1000.0  # 最终学习率 = max_lr / 1000
)

criterion = nn.CrossEntropyLoss()

# 梯度裁剪
max_grad_norm = 1.0

# ─────────── 7. 流式特征提取 ───────────
def stream_features(dataset, models, tokenizer, max_length, batch_size=8):
    """流式提取特征，避免内存溢出"""
    all_features = []
    all_labels = []
    
    # 先提取所有文本和标签
    texts, labels = zip(*[dataset[i] for i in range(len(dataset))])
    labels = torch.tensor(labels)
    
    # 动态批处理文本
    text_batches = dynamic_batch_processing(texts, tokenizer, max_length, batch_size * 4)
    
    logger.info(f"开始提取特征，共 {len(text_batches)} 个文本批次...")
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(text_batches, desc="特征提取")):
            # 处理每个模型
            batch_logits = []
            for model in models:
                try:
                    # 移动到设备
                    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 使用with torch.inference_mode()减少内存
                    with torch.inference_mode():
                        outputs = model(**inputs_gpu)
                        logits = outputs.logits
                    
                    batch_logits.append(logits.cpu())
                    del inputs_gpu, outputs
                except RuntimeError as e:
                    logger.error(f"模型预测失败: {e}")
                    # 使用零张量填充
                    batch_logits.append(torch.zeros(len(inputs['input_ids']), num_labels))
            
            # 拼接所有模型的logits
            combined = torch.cat(batch_logits, dim=1)
            all_features.append(combined)
            
            # 每10个批次清理一次
            if batch_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                log_memory_usage()
    
    # 返回特征和标签
    return torch.cat(all_features, dim=0), labels

# ─────────── 8. 分批训练函数 ───────────
def train_streaming(train_dataset, val_dataset):
    """流式训练函数"""
    logger.info("开始流式训练...")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    best_acc = 0.0
    patience_counter = 0
    patience = args.patience
    min_delta = args.min_delta
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # 避免多进程问题
    )
    
    logger.info(f"每epoch步数: {len(train_loader)}")
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*20} Epoch {epoch + 1}/{args.epochs} {'='*20}")
        
        # 训练阶段
        train_loss = train_epoch_streaming(train_loader, epoch)
        
        # 验证阶段
        val_acc = evaluate_streaming(val_dataset)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录详细信息
        logger.info(f"Epoch {epoch + 1} 结果:")
        logger.info(f"  训练损失: {train_loss:.4f}")
        logger.info(f"  验证准确率: {val_acc:.4f}")
        logger.info(f"  当前学习率: {current_lr:.2e}")
        
        # 早停检查
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(meta_model.state_dict(), args.save_path)
            logger.info(f"  ✓ 新的最佳验证准确率: {best_acc:.4f}")
            logger.info(f"  ✓ 模型已保存")
        else:
            patience_counter += 1
            logger.info(f"  - 验证准确率未提升，patience: {patience_counter}/{patience}")
        
        # 内存监控
        log_memory_usage()
        
        if patience_counter >= patience:
            logger.info(f"早停触发，在epoch {epoch + 1}停止训练")
            break
    
    return best_acc

def train_epoch_streaming(train_loader, epoch):
    """单个epoch的训练函数"""
    meta_model.train()
    total_loss = 0.0
    total_batches = 0
    
    logger.info(f"开始训练 Epoch {epoch + 1}...")
    
    # 使用tqdm显示进度
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练中", leave=False)
    
    for batch_idx, (texts, labels) in enumerate(progress_bar):
        # 动态处理文本批次
        text_batches = dynamic_batch_processing(texts, tokenizer, args.max_length, 8)
        
        # 初始化特征收集
        batch_features = []
        
        for inputs in text_batches:
            # 处理每个模型
            model_logits = []
            for model in models:
                inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs_gpu)
                    model_logits.append(outputs.logits.cpu())
                del inputs_gpu, outputs
            
            # 拼接特征
            features = torch.cat(model_logits, dim=1)
            batch_features.append(features)
        
        # 合并批次特征
        batch_features = torch.cat(batch_features, dim=0).to(device)
        labels = labels.to(device)
        
        # 训练步骤
        optimizer.zero_grad()
        outputs = meta_model(batch_features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 每50个批次清理内存
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # 计算平均训练损失
    avg_train_loss = total_loss / max(total_batches, 1)
    logger.info(f"Epoch {epoch + 1} 训练完成，平均损失: {avg_train_loss:.4f}")
    
    return avg_train_loss

# ─────────── 9. 流式评估函数 ───────────
def evaluate_streaming(dataset):
    """流式评估函数"""
    meta_model.eval()
    total_correct = 0
    total_samples = 0
    
    logger.info("开始验证...")
    
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 避免多进程问题
        pin_memory=False
    )
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="验证中", leave=False)
        
        for batch_idx, (texts, labels) in enumerate(progress_bar):
            # 动态处理文本批次
            text_batches = dynamic_batch_processing(texts, tokenizer, args.max_length, 8)
            
            # 收集特征
            batch_features = []
            for inputs in text_batches:
                model_logits = []
                for model in models:
                    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        outputs = model(**inputs_gpu)
                        model_logits.append(outputs.logits.cpu())
                    del inputs_gpu, outputs
                
                features = torch.cat(model_logits, dim=1)
                batch_features.append(features)
            
            # 预测
            features = torch.cat(batch_features, dim=0).to(device)
            outputs = meta_model(features)
            preds = outputs.argmax(dim=1).cpu()
            
            # 更新统计
            batch_correct = (preds == labels).sum().item()
            total_correct += batch_correct
            total_samples += len(labels)
            
            # 更新进度条
            current_acc = total_correct / total_samples
            progress_bar.set_postfix(acc=f"{current_acc:.4f}")
            
            # 定期清理内存
            if batch_idx % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    
    final_acc = total_correct / max(total_samples, 1)
    logger.info(f"验证完成，准确率: {final_acc:.4f} ({total_correct}/{total_samples})")
    
    return final_acc

# ─────────── 10. 主函数 ───────────
def main():
    # 设置日志
    global logger
    logger = setup_logging(args.save_path)
    
    logger.info("=" * 60)
    logger.info("开始训练堆叠模型")
    logger.info("=" * 60)
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"验证数据: {args.val_data}")
    logger.info(f"模型目录: {args.model_dirs}")
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"学习率: {args.lr}")
    logger.info("=" * 60)
    
    # 加载数据
    def load_data(path):
        df = pd.read_csv(path)
        required_columns = ['资产名称', '型号', '用途', '新国标分类', '使用部门']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据集缺少必要的列: {missing_cols}")
        logger.info(f"加载数据: {path}, 样本数: {len(df)}")
        return df

    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)

    # 创建数据集
    train_dataset = EfficientStackingDataset(
        train_df, tokenizer, args.max_length, 
        max_samples=args.max_train_samples
    )
    val_dataset = EfficientStackingDataset(
        val_df, tokenizer, args.max_length, 
        max_samples=args.max_val_samples
    )

    logger.info("开始流式训练...")
    logger.info("=" * 60)
    
    # 开始训练
    best_acc = train_streaming(train_dataset, val_dataset)
    
    logger.info("=" * 60)
    logger.info(f"训练完成！最佳验证准确率: {best_acc:.4f}")
    logger.info(f"模型已保存到: {args.save_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

"""
nohup python train_stacking_rwkv.py   --model_dirs /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-12-13 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-14-47 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-16-32   --train_data ./datasets/0717/multi_inputs_train.csv   --val_data ./datasets/0717/multi_inputs_dev.csv --save_path /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/stacking/meta_rwkv_model.pth &
"""