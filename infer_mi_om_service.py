#!/usr/bin/env python
# infer_service.py  —— 多输入模型推理服务

#python infer_service.py --input dev.csv --model_dir ./model_output/multi_inputs --top_k 5
"""
只是用模型进行推理，不使用规则库"""
import argparse, json, os, sys
from collections import defaultdict
import logging

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────── CLI ───────────
p = argparse.ArgumentParser(description='Multi-input inference service')
p.add_argument('--input', type=str, default='./datasets/multi_chars_data/multi_inputs_dev.csv')
p.add_argument('--model_dir', type=str, default='./model_output/multi_inputs/2025-07-14-20-21')
p.add_argument('--tokenizer_name', type=str, default=None,
               help='若 tokenizer 与模型目录不同可单独指定')
p.add_argument('--id2label', type=str, default=None)
p.add_argument('--top_k', type=int, default=5)
p.add_argument('--bs', type=int, default=32, help='batch size for model')
p.add_argument('--max_length', type=int, default=64, help='max sequence length')
args = p.parse_args()

# 自动确定id2label路径
if args.id2label is None:
    args.id2label = "train_id2label.json"

LABEL_COL, K = '新国标分类', args.top_k

# ─────────── 1. id ↔ label 映射 ───────────
id2label = {}
if os.path.exists(args.id2label):
    with open(args.id2label, 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    logger.info(f"加载标签映射，共 {len(id2label)} 个类别")
else:
    logger.error(f"标签映射文件不存在: {args.id2label}")
    sys.exit(1)

# ─────────── 2. 载入数据 ───────────
df = pd.read_csv(args.input)
required_columns = ['资产名称', '型号', '用途']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    logger.error(f"输入文件缺少必要的列: {missing_cols}")
    sys.exit(1)

has_true = LABEL_COL in df.columns
if has_true:
    logger.info(f"检测到真实标签列: {LABEL_COL}")
    # 确保真实标签与预测标签使用相同的映射
    # 参考 infer_service.py 的处理方式
    if id2label:
        df[LABEL_COL] = df[LABEL_COL].astype(str).map(id2label).fillna(df[LABEL_COL].astype(str))
    true_labels = df[LABEL_COL].astype(str).tolist()
else:
    logger.info("未检测到真实标签列，将只进行预测")
    true_labels = None

# ─────────── 3. 加载模型 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")

try:
    tok_name = args.tokenizer_name or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, ignore_mismatched_sizes=True).to(device)
    model.eval()  # 确保模型处于评估模式
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    sys.exit(1)

@torch.no_grad()
def model_predict(asset_names, models, purposes):
    """
    多输入模型预测函数
    """
    # 构建多输入文本
    combined_texts = []
    for asset_name, model_name, purpose in zip(asset_names, models, purposes):
        # 使用与训练时相同的拼接格式
        combined_text = f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose}"
        combined_texts.append(combined_text)
    
    all_predictions = []
    
    # 分批处理
    for i in range(0, len(combined_texts), args.bs):
        batch_texts = combined_texts[i:i+args.bs]
        batch_tokenized = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
            return_tensors='pt'
        )
        
        # 移动到设备
        batch_inputs = {k: v.to(device) for k, v in batch_tokenized.items()}
        # 模型推理
        outputs = model(**batch_inputs)
        logits = outputs.logits.cpu()
        
        # 获取top-k预测
        top_idx = logits.topk(K, dim=-1).indices
        
        for row in top_idx:
            predictions = []
            for idx in row:
                pred_label = id2label.get(str(int(idx)), "未知")
                predictions.append(pred_label)
            all_predictions.append(predictions)
    
    return all_predictions

# ─────────── 4. 推理 ----------
logger.info("开始模型推理...")

# 提取输入数据
asset_names = df['资产名称'].tolist()
models = df['型号'].tolist()
purposes = df['用途'].tolist()

# 批量预测
predictions = model_predict(asset_names, models, purposes)
logger.info(f"完成 {len(predictions)} 个样本的预测")

# ─────────── 5. 评估 & 输出 ----------
# 添加预测结果到DataFrame
df['候选标签'] = ['|'.join(x) for x in predictions]
df['top1'] = [x[0] for x in predictions]

# 保存结果
out_csv = args.input.replace('.csv', f'_pred_top{K}.csv')
df.to_csv(out_csv, index=False, encoding='utf-8')
logger.info(f'✅ 结果已保存到 {out_csv}')

# 统计信息
logger.info(f"总样本数: {len(df)}")
logger.info(f"预测完成: {len(predictions)}")

if has_true and true_labels:
    # 确保标签类型一致，都转换为字符串
    true_labels_str = [str(label) for label in true_labels]
    pred_labels_str = [str(label) for label in df['top1']]
    
    # 计算准确率
    acc1 = accuracy_score(true_labels_str, pred_labels_str)
    acck = sum(t in c for t, c in zip(true_labels_str, predictions)) / len(df)
    
    logger.info(f'Overall  Accuracy@1={acc1:.4f}  Accuracy@{K}={acck:.4f}')
    
    # # 详细分类报告
    # try:
    #     report = classification_report(true_labels_str, pred_labels_str, output_dict=True)
    #     logger.info("分类报告:")
    #     for label, metrics in report.items():
    #         if isinstance(metrics, dict) and 'precision' in metrics:
    #             logger.info(f"  {label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1-score']:.4f}")
    # except Exception as e:
    #     logger.warning(f"生成分类报告时出错: {e}")
else:
    logger.info('⚠️  无真实标签列，跳过准确率评估')

# 显示前几个预测结果作为示例
logger.info("预测结果示例:")
for i in range(min(5, len(df))):
    logger.info(f"  样本{i+1}: {df['资产名称'].iloc[i]} | {df['型号'].iloc[i]} | {df['用途'].iloc[i]} -> {df['top1'].iloc[i]}")