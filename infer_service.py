#!/usr/bin/env python
# infer_service.py  —— 规则优先 + (可选) 模型兜底 + 详细指标

#python infer_service.py   --input dev.csv   --rule_xlsx 新国标固定资产.xlsx   --model_dir ./model_output   --top_k 5

import argparse, json, os, sys
from collections import defaultdict

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# ─────────── CLI ───────────
p = argparse.ArgumentParser(description='Hybrid inference service')
p.add_argument('--input', type=str, default='./datasets/test.csv')
p.add_argument('--rule_xlsx', type=str, default='新国标固定资产.xlsx')
p.add_argument('--model_dir', type=str, default='model_output')
p.add_argument('--tokenizer_name', type=str, default=None,
               help='若 tokenizer 与模型目录不同可单独指定')
p.add_argument('--id2label', type=str, default='train_id2label.json')
p.add_argument('--top_k', type=int, default=5)
p.add_argument('--bs', type=int, default=32, help='batch size for model')
p.add_argument('--no-fallback', action='store_true')
args = p.parse_args()

TEXT_COL, LABEL_COL, K = '资产名称', '新国标分类', args.top_k

# ─────────── 1. id ↔ label 映射 ───────────
id2label = {}
if os.path.exists(args.id2label):
    id2label = {str(k): str(v) for k, v in json.load(open(args.id2label, encoding='utf-8')).items()}

# ─────────── 2. 载入数据 ───────────

df = pd.read_csv(args.input)
if TEXT_COL not in df.columns:
    sys.exit(f'[ERR] {args.input} 缺少列: {TEXT_COL}')
has_true = LABEL_COL in df.columns
if has_true and id2label:
    df[LABEL_COL] = df[LABEL_COL].astype(str).map(id2label).fillna(df[LABEL_COL].astype(str))
true_labels = df[LABEL_COL].tolist() if has_true else None

# ─────────── 3. 规则库 → dict ----------
rule_df = pd.read_excel(args.rule_xlsx)[[TEXT_COL, LABEL_COL]].dropna()
rule_map = defaultdict(list)
for _, r in rule_df.iterrows():
    rule_map[str(r[TEXT_COL]).strip().lower()].append(str(r[LABEL_COL]).strip())
for k in rule_map:
    seen = set()
    rule_map[k] = [x for x in rule_map[k] if not (x in seen or seen.add(x))]

def rule_lookup(text):
    return rule_map.get(text.strip().lower(), [])

# ─────────── 4. （可选）加载模型 ----------
use_fallback = not args.no_fallback
if use_fallback:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok_name = args.tokenizer_name or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, ignore_mismatched_sizes=True).to(device).eval()

    # 若模型自身带 id2label，优先使用，以防顺序不一致
    # if hasattr(model.config, 'id2label') and model.config.id2label:
    #     id2label = {str(i): l for i, l in model.config.id2label.items()}

    @torch.no_grad()
    def model_predict(texts):
        ds = Dataset.from_dict({TEXT_COL: texts})
        ds = ds.map(lambda x: tokenizer(
            x[TEXT_COL], truncation=True, padding='max_length',
            max_length=32), batched=True)
        ds.set_format('pt', columns=['input_ids', 'attention_mask'])
        loader = torch.utils.data.DataLoader(ds, batch_size=args.bs)

        out = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu()
            top_idx = logits.topk(K, dim=-1).indices          # ← 修正
            out.extend([[id2label.get(str(int(i)), str(int(i))) for i in row]
                        for row in top_idx])
        return out

# ─────────── 5. 推理 ----------
candidates, need_model = [], []
for txt in df[TEXT_COL]:
    labs = rule_lookup(txt)
    if labs:
        candidates.append((labs + [''] * K)[:K])
    else:
        if use_fallback:
            need_model.append(txt)
            candidates.append(None)
        else:
            candidates.append([''] * K)

# 调模型
if need_model:
    pred_from_model = model_predict(need_model)
    it = iter(pred_from_model)
    for i, c in enumerate(candidates):
        if c is None:
            candidates[i] = next(it)

# ─────────── 6. 评估 & 输出 ----------
df['候选标签'] = ['|'.join(x) for x in candidates]
print(df['候选标签'])
df['top1'] = [x[0] for x in candidates]
out_csv = args.input.replace('.csv', f'_pred_top{K}.csv')
df.to_csv(out_csv, index=False, encoding='utf-8')
print(f'✅ 结果已保存到 {out_csv}')

# 统计
rule_hits = sum(bool(rule_lookup(t)) for t in df[TEXT_COL])
print(f'规则命中条数: {rule_hits} / {len(df)}')
if use_fallback:
    print(f'模型兜底条数: {len(need_model)}')

if has_true:
    # 整体
    acc1 = accuracy_score(true_labels, df['top1'])
    acck = sum(t in c for t, c in zip(true_labels, candidates)) / len(df)
    print(f'Overall  Accuracy@1={acc1:.4f}  Accuracy@{K}={acck:.4f}')

    # 分别评估规则 / 模型
    rule_idx = [i for i, txt in enumerate(df[TEXT_COL]) if rule_lookup(txt)]
    model_idx = [i for i in range(len(df)) if i not in rule_idx]

    if rule_idx:
        r_acc1 = accuracy_score([true_labels[i] for i in rule_idx],
                                [df["top1"][i] for i in rule_idx])
        print(f'规则子集 Accuracy@1={r_acc1:.4f} (size={len(rule_idx)})')
    if model_idx:
        m_acc1 = accuracy_score([true_labels[i] for i in model_idx],
                                [df["top1"][i] for i in model_idx])
        print(f'模型子集 Accuracy@1={m_acc1:.4f} (size={len(model_idx)})')
else:
    print('⚠️  无真实标签列，跳过准确率评估')