#!/usr/bin/env python
# infer_service.py  —— 多输入模型推理服务

#python infer_service.py --input dev.csv --model_dir ./model_output/multi_inputs --top_k 5
"""
使用规则库+模型的方案
"""
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
p = argparse.ArgumentParser(description='Hybrid inference service with multi-input model')
# p.add_argument('--input', type=str, default='./datasets/multi_chars_data/multi_inputs_dev.csv')
# p.add_argument('--rule_xlsx', type=str, default='rule_新国标分类测试集.xlsx')
# p.add_argument('--model_dir', type=str, default='./model_output/multi_inputs/2025-07-14-20-21')
p.add_argument('--input', type=str, default='./datasets/0717/multi_inputs_dev.csv')
p.add_argument('--rule_xlsx', type=str, default='rule_all_data_0717.xlsx')
p.add_argument('--model_dir', type=str, default='./model_output/multi_inputs/2025-07-18-12-13')
p.add_argument('--tokenizer_name', type=str, default=None,
               help='若 tokenizer 与模型目录不同可单独指定')
p.add_argument('--id2label', type=str, default=None)
p.add_argument('--top_k', type=int, default=3)
p.add_argument('--bs', type=int, default=32, help='batch size for model')
p.add_argument('--max_length', type=int, default=64, help='max sequence length')
p.add_argument('--no-fallback', action='store_true', help='不使用模型兜底，仅使用规则')
args = p.parse_args()

# 自动确定id2label路径
if args.id2label is None:
    args.id2label = "train_id2label_0717.json"

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
required_columns = ['资产名称', '型号', '用途','使用部门']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    logger.error(f"输入文件缺少必要的列: {missing_cols}")
    sys.exit(1)

has_true = LABEL_COL in df.columns
if has_true:
    logger.info(f"检测到真实标签列: {LABEL_COL}")
    # 确保真实标签与预测标签使用相同的映射
    if id2label:
        df[LABEL_COL] = df[LABEL_COL].astype(str).map(id2label).fillna(df[LABEL_COL].astype(str))
    true_labels = df[LABEL_COL].astype(str).tolist()
else:
    logger.info("未检测到真实标签列，将只进行预测")
    true_labels = None

# ─────────── 3. 规则库 → dict ----------
logger.info("加载规则库...")
rule_df = pd.read_excel(args.rule_xlsx)[['资产名称', LABEL_COL]].dropna()
rule_map = defaultdict(list)
for _, r in rule_df.iterrows():
    rule_map[str(r['资产名称']).strip().lower()].append(str(r[LABEL_COL]).strip())
for k in rule_map:
    seen = set()
    rule_map[k] = [x for x in rule_map[k] if not (x in seen or seen.add(x))]
logger.info(f"规则库加载完成，共 {len(rule_map)} 条规则")

def rule_lookup(text):
    return rule_map.get(text.strip().lower(), [])

# ─────────── 4. （可选）加载模型 ----------
use_fallback = not args.no_fallback
if use_fallback:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    try:
        tok_name = args.tokenizer_name or args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, ignore_mismatched_sizes=True).to(device).eval()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        sys.exit(1)
else:
    logger.info("跳过模型加载，仅使用规则")

@torch.no_grad()
def model_predict(asset_names, models, purposes, departments):
    """
    多输入模型预测函数
    """
    # 构建多输入文本
    combined_texts = []
    for asset_name, model_name, purpose, department in zip(asset_names, models, purposes, departments):
        # 使用与训练时相同的拼接格式
        combined_text = f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose} 使用部门: {department}"
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

# ─────────── 5. 推理 ----------
logger.info("开始推理...")

# 规则优先 + 模型补全
candidates, need_model = [], []
for i, row in df.iterrows():
    asset_name = str(row['资产名称'])
    model_name = str(row['型号'])
    purpose = str(row['用途'])
    department = str(row['使用部门'])
    
    # 先尝试规则匹配
    labs = rule_lookup(asset_name)
    if labs:
        # 规则命中，检查是否需要模型补全
        if len(labs) >= K:
            # 规则结果足够，直接使用
            candidates.append(labs[:K])
            need_model.append(None)
        else:
            # 规则结果不足，需要模型补全
            candidates.append(labs)  # 先保存规则结果
            need_model.append((asset_name, model_name, purpose, department, len(labs)))  # 记录需要补全的数量
    else:
        # 规则未命中，需要模型预测
        if use_fallback:
            need_model.append((asset_name, model_name, purpose, department, 0))  # 0表示完全由模型预测
            candidates.append([])
        else:
            candidates.append([''] * K)
            need_model.append(None)

# 调用模型进行补全预测
if use_fallback and any(x is not None for x in need_model):
    logger.info("使用模型进行补全预测...")
    
    # 提取需要模型预测的样本
    model_inputs = [x for x in need_model if x is not None]
    if model_inputs:
        asset_names_model = [x[0] for x in model_inputs]
        models_model = [x[1] for x in model_inputs]
        purposes_model = [x[2] for x in model_inputs]
        department_model = [x[3] for x in model_inputs]
        
        # 批量预测
        model_predictions = model_predict(asset_names_model, models_model, purposes_model, department_model)
        
        # 将模型预测结果填入candidates
        model_idx = 0
        for i, (candidate, model_input) in enumerate(zip(candidates, need_model)):
            if model_input is not None:
                existing_count = model_input[4]  # 已有的规则结果数量
                model_pred = model_predictions[model_idx]
                
                if existing_count == 0:
                    # 完全由模型预测
                    candidates[i] = model_pred
                else:
                    # 规则 + 模型补全
                    # 过滤掉已经在规则结果中的标签，避免重复
                    existing_labels = set(candidate)
                    additional_labels = [label for label in model_pred if label not in existing_labels]
                    # 补全到K个标签
                    candidates[i] = candidate + additional_labels[:K - existing_count]
                
                model_idx += 1

logger.info(f"推理完成，共处理 {len(candidates)} 个样本")

# ─────────── 6. 评估 & 输出 ----------
# 添加预测结果到DataFrame
df['候选标签'] = ['|'.join(x) for x in candidates]
df['top1'] = [x[0] for x in candidates]

# 保存结果
out_csv = args.input.replace('.csv', f'_pred_top{K}.csv')
df.to_csv(out_csv, index=False, encoding='utf-8')
logger.info(f'✅ 结果已保存到 {out_csv}')

# 统计信息
rule_hits = sum(bool(rule_lookup(str(row['资产名称']))) for _, row in df.iterrows())
model_completion = sum(1 for x in need_model if x is not None)
logger.info(f"总样本数: {len(df)}")
logger.info(f"规则命中条数: {rule_hits} / {len(df)}")
if use_fallback:
    logger.info(f"模型补全条数: {model_completion}")

if has_true and true_labels:
    # 确保标签类型一致，都转换为字符串
    true_labels_str = [str(label) for label in true_labels]
    pred_labels_str = [str(label) for label in df['top1']]
    
    # 计算准确率
    acc1 = accuracy_score(true_labels_str, pred_labels_str)
    acck = sum(t in c for t, c in zip(true_labels_str, candidates)) / len(df)
    
    logger.info(f'Overall  Accuracy@1={acc1:.4f}  Accuracy@{K}={acck:.4f}')
    
    # 分别评估规则 / 模型
    # 规则预测：完全由规则预测的样本
    rule_only_idx = [i for i, (candidate, model_input) in enumerate(zip(candidates, need_model)) 
                     if model_input is None]
    # 模型预测：包含模型预测的样本（完全模型预测或规则+模型补全）
    model_involved_idx = [i for i, model_input in enumerate(need_model) if model_input is not None]

    if rule_only_idx:
        r_acc1 = accuracy_score([true_labels_str[i] for i in rule_only_idx],
                                [pred_labels_str[i] for i in rule_only_idx])
        logger.info(f'纯规则预测 Accuracy@1={r_acc1:.4f} (size={len(rule_only_idx)})')
    if model_involved_idx and use_fallback:
        m_acc1 = accuracy_score([true_labels_str[i] for i in model_involved_idx],
                                [pred_labels_str[i] for i in model_involved_idx])
        logger.info(f'模型参与预测 Accuracy@1={m_acc1:.4f} (size={len(model_involved_idx)})')
    
    # 详细分类报告
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
    if i < len(need_model) and need_model[i] is not None:
        existing_count = need_model[i][4]
        if existing_count == 0:
            method = "模型"
        else:
            method = f"规则+模型补全({existing_count}+{K-existing_count})"
    else:
        method = "规则"
    logger.info(f"  样本{i+1}: {df['资产名称'].iloc[i]} | {df['型号'].iloc[i]} | {df['用途'].iloc[i]} | {df['使用部门'].iloc[i]} -> {df['top1'].iloc[i]} ({method})")