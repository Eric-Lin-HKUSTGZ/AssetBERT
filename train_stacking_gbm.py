#!/usr/bin/env python
# train_stacking_lightgbm.py —— 使用LightGBM进行模型集成

import os
# 设置环境变量，避免tokenizers并行化警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
import json
import logging
import gc
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────── 配置日志 ───────────
def setup_logging(save_path):
    """设置日志配置，将日志保存到与模型相同的目录"""
    # 获取保存目录
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成日志文件名
    log_filename = os.path.join(save_dir, 'training_lightgbm.log')
    
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
parser.add_argument('--batch_size', type=int, default=32, help='特征提取批次大小')
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--max_train_samples', type=int, default=100000, help='最大训练样本数')
parser.add_argument('--max_val_samples', type=int, default=20000, help='最大验证样本数')
parser.add_argument('--lgbm_params', type=str, default='{}', help='LightGBM参数字典的JSON字符串')
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
class EfficientStackingDataset:
    def __init__(self, df, tokenizer, max_length, max_samples=None):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 限制样本数量
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(max_samples, random_state=42)
            logger.info(f"限制数据集大小为 {max_samples} 个样本")
        
        logger.info(f"有效样本数: {len(self.df)}")
        log_memory_usage()

    def __len__(self):
        return len(self.df)

    def get_texts(self):
        """获取所有文本"""
        texts = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            text = f"资产名称: {row['资产名称']} 型号: {row['型号']} 用途: {row['用途']} 使用部门: {row['使用部门']}"
            texts.append(text)
        return texts
    
    def get_labels(self):
        """获取所有标签"""
        return self.df['新国标分类'].astype(int).values

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

# ─────────── 6. 特征提取函数 ───────────
def extract_features(dataset, models, tokenizer, max_length, batch_size=32):
    """提取特征（基础模型的预测logits）"""
    texts = dataset.get_texts()
    labels = dataset.get_labels()
    
    # 动态批处理文本
    text_batches = dynamic_batch_processing(texts, tokenizer, max_length, batch_size)
    
    logger.info(f"开始提取特征，共 {len(text_batches)} 个文本批次...")
    
    all_features = []
    
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
                    
                    batch_logits.append(logits.cpu().numpy())
                    del inputs_gpu, outputs
                except RuntimeError as e:
                    logger.error(f"模型预测失败: {e}")
                    # 使用零张量填充
                    batch_logits.append(np.zeros((len(inputs['input_ids']), num_labels)))
            
            # 拼接所有模型的logits
            # 将每个模型的logits在特征维度上拼接
            batch_features = np.concatenate(batch_logits, axis=1)
            all_features.append(batch_features)
            
            # 每10个批次清理一次
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # 返回特征和标签
    features = np.concatenate(all_features, axis=0)
    logger.info(f"特征提取完成 - 特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    return features, labels

# ─────────── 7. LightGBM模型训练 ───────────
def train_lightgbm(X_train, y_train, X_val, y_val, params):
    """训练LightGBM模型"""
    logger.info("开始训练LightGBM模型...")
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 默认参数
    default_params = {
        'objective': 'multiclass',
        'num_class': num_labels,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'num_threads': 16
    }
    
    # 更新用户自定义参数
    try:
        user_params = json.loads(args.lgbm_params)
        default_params.update(user_params)
        logger.info(f"使用自定义参数: {user_params}")
    except json.JSONDecodeError:
        logger.warning(f"无法解析lgbm_params参数，使用默认参数")
    
    logger.info(f"最终训练参数: {default_params}")
    
    # 训练模型
    # evals_result = {}
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
        # evals_result=evals_result
    )
    
    # 评估模型
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    val_preds_labels = np.argmax(val_preds, axis=1)
    val_acc = accuracy_score(y_val, val_preds_labels)
    
    logger.info(f"验证准确率: {val_acc:.4f}")
    logger.info(f"最佳迭代次数: {model.best_iteration}")
    
    return model

# ─────────── 8. 主函数 ───────────
def main():
    # 设置日志
    global logger
    logger = setup_logging(args.save_path)
    
    logger.info("=" * 60)
    logger.info("开始训练LightGBM堆叠模型")
    logger.info("=" * 60)
    logger.info(f"训练数据: {args.train_data}")
    logger.info(f"验证数据: {args.val_data}")
    logger.info(f"模型目录: {args.model_dirs}")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"特征提取批次大小: {args.batch_size}")
    logger.info(f"类别数量: {num_labels}")
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
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 提取特征
    logger.info("提取训练集特征...")
    X_train, y_train = extract_features(train_dataset, models, tokenizer, args.max_length, args.batch_size)
    
    logger.info("提取验证集特征...")
    X_val, y_val = extract_features(val_dataset, models, tokenizer, args.max_length, args.batch_size)
    
    # 训练LightGBM模型
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, args.lgbm_params)
    
    # 保存模型
    model_path = args.save_path
    if not model_path.endswith('.txt'):
        model_path += '.txt'
    
    lgb_model.save_model(model_path)
    logger.info(f"LightGBM模型已保存到: {model_path}")
    
    # 同时保存为二进制格式以便快速加载
    bin_path = model_path.replace('.txt', '.bin')
    joblib.dump(lgb_model, bin_path)
    logger.info(f"LightGBM模型已保存为二进制格式: {bin_path}")
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

"""
python train_stacking_gbm.py --model_dirs /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-12-13 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-14-47 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-16-32 --train_data ./datasets/0717/multi_inputs_train.csv  --val_data ./datasets/0717/multi_inputs_dev.csv --save_path ./model_output/multi_inputs/stacking/lightgbm_model.txt --lgbm_params '{"num_leaves": 63, "learning_rate": 0.1, "num_iterations": 500}'
"""