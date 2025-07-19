#!/usr/bin/env python
# infer_service.py —— 多输入模型推理服务（支持多模型加权投票）

# python infer_service.py --input dev.csv --model_dirs ./model1 ./model2 ./model3 --top_k 5 --weights 0.4 0.3 0.3

import argparse, json, os, sys
from collections import defaultdict, Counter
import logging

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# ─────────── 日志配置 ───────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────── CLI 参数 ───────────
p = argparse.ArgumentParser(description='Multi-input inference service with weighted voting')
p.add_argument('--input', type=str, default='./datasets/multi_chars_data/multi_inputs_dev.csv')
p.add_argument('--model_dirs', nargs='+', type=str, required=True,
               help='多个模型目录，顺序对应权重')
p.add_argument('--tokenizer_name', type=str, default=None,
               help='若 tokenizer 与模型目录不同可单独指定')
p.add_argument('--id2label', type=str, default=None)
p.add_argument('--top_k', type=int, default=5)
p.add_argument('--bs', type=int, default=32, help='batch size for model')
p.add_argument('--max_length', type=int, default=64, help='max sequence length')
p.add_argument('--weights', nargs='+', type=float, default=None,
               help='每个模型的权重，若不指定则平均权重')
args = p.parse_args()

LABEL_COL, K = '新国标分类', args.top_k

# ─────────── 1. id ↔ label 映射 ───────────
if args.id2label is None:
    args.id2label = "train_id2label_0717.json"

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
required_columns = ['资产名称', '型号', '用途', '使用部门']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    logger.error(f"输入文件缺少必要的列: {missing_cols}")
    sys.exit(1)

has_true = LABEL_COL in df.columns
if has_true:
    logger.info(f"检测到真实标签列: {LABEL_COL}")
    if id2label:
        df[LABEL_COL] = df[LABEL_COL].astype(str).map(id2label).fillna(df[LABEL_COL].astype(str))
    true_labels = df[LABEL_COL].astype(str).tolist()
else:
    logger.info("未检测到真实标签列，将只进行预测")
    true_labels = None

# ─────────── 3. 加载模型和权重 ───────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")

# 检查权重数量是否与模型数量一致
if args.weights is not None:
    if len(args.weights) != len(args.model_dirs):
        logger.error("权重数量与模型数量不一致")
        sys.exit(1)
    weights = args.weights
else:
    weights = [1.0 / len(args.model_dirs)] * len(args.model_dirs)

# 加载多个模型
try:
    tok_name = args.tokenizer_name or args.model_dirs[0]
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    models = []
    for model_dir in args.model_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, ignore_mismatched_sizes=True).to(device)
        model.eval()
        models.append(model)
    logger.info(f"加载 {len(models)} 个模型成功，权重: {weights}")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    sys.exit(1)

# ─────────── 4. 多模型加权预测函数 ───────────
@torch.no_grad()
def weighted_logits_predict(asset_names, model_names, purposes, departments, weights):
    combined_texts = [
        f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose} 使用部门: {department}"
        for asset_name, model_name, purpose, department in zip(asset_names, model_names, purposes, departments)
    ]
    
    all_weighted_predictions = []

    for i in range(0, len(combined_texts), args.bs):
        batch_texts = combined_texts[i:i + args.bs]
        batch_tokenized = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
            return_tensors='pt'
        ).to(device)

        # 初始化加权logits
        total_logits = None

        for model, weight in zip(models, weights):
            batch_inputs = {k: v.to(device) for k, v in batch_tokenized.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits.cpu()

            if total_logits is None:
                total_logits = logits * weight
            else:
                total_logits += logits * weight

        # softmax
        probs = torch.softmax(total_logits, dim=-1)
        top_idx = probs.topk(K, dim=-1).indices

        # 解码预测
        batch_preds = []
        for row in top_idx:
            predictions = [id2label.get(str(int(idx)), "未知") for idx in row]
            batch_preds.append(predictions)
        all_weighted_predictions.extend(batch_preds)

    return all_weighted_predictions

@torch.no_grad()
def weighted_model_predict(asset_names, model_names, purposes, departments, weights):
    combined_texts = [
        f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose} 使用部门: {department}"
        for asset_name, model_name, purpose, department in zip(asset_names, model_names, purposes, departments)
    ]
    
    all_weighted_predictions = []

    for i in range(0, len(combined_texts), args.bs):
        batch_texts = combined_texts[i:i + args.bs]
        batch_tokenized = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
            return_tensors='pt'
        ).to(device)

        # 用于存储每个模型对每个样本的加权得分
        batch_weighted_scores = [{} for _ in range(len(batch_texts))]

        for model, weight in zip(models, weights):
            batch_inputs = {k: v.to(device) for k, v in batch_tokenized.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits.cpu()

            top_idx = logits.topk(K, dim=-1).indices
            top_scores = torch.gather(logits, dim=-1, index=top_idx)

            for j, (indices, scores) in enumerate(zip(top_idx, top_scores)):
                for idx, score in zip(indices, scores):
                    pred_label = id2label.get(str(int(idx)), "未知")
                    batch_weighted_scores[j][pred_label] = \
                        batch_weighted_scores[j].get(pred_label, 0.0) + float(score) * weight

        # 取每个样本加权后得分最高的类别
        final_predictions = []
        for scores_dict in batch_weighted_scores:
            sorted_preds = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            top_preds = [x[0] for x in sorted_preds[:K]]
            final_predictions.append(top_preds)

        all_weighted_predictions.extend(final_predictions)

    return all_weighted_predictions

# ─────────── 5. 推理执行 ───────────
logger.info("开始多模型加权投票推理...")

asset_names = df['资产名称'].tolist()
model_names = df['型号'].tolist()  # 重命名变量，避免与模型列表冲突
purposes = df['用途'].tolist()
departments = df['使用部门'].tolist()

predictions = weighted_model_predict(asset_names, model_names, purposes, departments, weights)
logger.info(f"完成 {len(predictions)} 个样本的加权预测")

# ─────────── 6. 输出结果 ───────────
df['候选标签'] = ['|'.join(x) for x in predictions]
df['top1'] = [x[0] for x in predictions]

out_csv = args.input.replace('.csv', f'_weighted_pred_top{K}.csv')
df.to_csv(out_csv, index=False, encoding='utf-8')
logger.info(f'✅ 结果已保存到 {out_csv}')

# ─────────── 7. 评估 ───────────
if has_true and true_labels:
    true_labels_str = [str(label) for label in true_labels]
    pred_labels_str = [str(label) for label in df['top1']]

    acc1 = accuracy_score(true_labels_str, pred_labels_str)
    acck = sum(t in c for t, c in zip(true_labels_str, predictions)) / len(df)

    logger.info(f'Overall  Accuracy@1={acc1:.4f}  Accuracy@{K}={acck:.4f}')
else:
    logger.info('⚠️ 无真实标签列，跳过准确率评估')

# 显示预测示例
logger.info("预测结果示例:")
for i in range(min(5, len(df))):
    logger.info(f"  样本{i+1}: {df['资产名称'].iloc[i]} | {df['型号'].iloc[i]} | {df['用途'].iloc[i]} | {df['使用部门'].iloc[i]} -> {df['top1'].iloc[i]}")


"""
程序运行命令
python infer_mi_om_assemble_service.py \
  --input ./datasets/0717/multi_inputs_dev.csv \
  --model_dirs /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-12-13 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-14-47 /hpc2hdd/home/qxiao183/linweiquan/AssetBERT/model_output/multi_inputs/2025-07-18-16-32 \
  --weights 0.3 0.3 0.4 \
  --top_k 3
"""