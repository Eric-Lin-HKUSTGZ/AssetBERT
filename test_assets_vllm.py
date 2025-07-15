#!/usr/bin/env python
# test_assets_vllm_pipe_topk.py   ——  一次生成 k 标签，默认 k=5

import argparse, json, re
from collections import defaultdict

import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ─────────── 参数 ───────────
parser = argparse.ArgumentParser(description='资产名称分类 (一次生成 k 标签)')
parser.add_argument('--model_path', type=str,
    default='/hpc2hdd/home/qxiao183/models/Qwen2.5-7B-Instruct')
parser.add_argument('--gpu_util', type=float, default=0.9)
parser.add_argument('--input', type=str, default='./新国标固定资产.xlsx')
parser.add_argument('--text_col', type=str, default='资产名称')
parser.add_argument('--label_col', type=str, default='新国标分类')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--prompt_style', type=str, default='labels',
    choices=['labels', 'labels_examples'])
parser.add_argument('--examples_per_label', type=int, default=1)
parser.add_argument('--top_k', type=int, default=5,            # ★ 默认 5
    help='要求模型一次返回的候选标签个数')
parser.add_argument('--out_csv', type=str, default='./assets_pred_pipe.csv')
args = parser.parse_args()

# ─────────── 读取并去重 ───────────
df = pd.read_excel(args.input)[[args.text_col, args.label_col]].dropna()
df = df.drop_duplicates(subset=[args.text_col]).reset_index(drop=True)
print(f'✔ 去重后样本数: {len(df)}')

# ─────────── 构造 label_prompt ───────────
labels = df[args.label_col].unique().tolist()
if args.prompt_style == 'labels':
    label_prompt = "、".join(labels)
else:
    sample_dict = defaultdict(list)
    for _, r in df.iterrows():
        if len(sample_dict[r[args.label_col]]) < args.examples_per_label:
            sample_dict[r[args.label_col]].append(r[args.text_col])
        if all(len(v) >= args.examples_per_label for v in sample_dict.values()):
            break
    label_prompt = "\n".join(
        f"{l}（例：{'；'.join(sample_dict[l])}）" for l in labels
    )

# ─────────── 初始化 vLLM ───────────
print('⏳ 加载模型 …')
llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_util)
sampling = SamplingParams(
    max_tokens=32,
    temperature=0.0,
    stop=["\n", "。", "，", ",", "。"]
)

# ─────────── Prompt 模板 ───────────
prompt_tpl = (
    "你是一位固定资产分类专家。\n"
    "请从下列可选类别中选出可能性最高的 {k} 个，用竖线“|”分隔，按可能性从高到低排列。\n"
    "不得输出列表之外的类别，也不要输出多余字符。\n"
    "可选类别：\n{label_prompt}\n"
    "资产名称：{asset}\n"
    "类别："
)

def make_prompts(batch):
    return [
        prompt_tpl.format(k=args.top_k, label_prompt=label_prompt, asset=x)
        for x in batch
    ]

def split_labels(gen: str, k: int) -> list[str]:
    """把模型生成的“标签1|标签2|…”拆成列表；不足 k 用空字符串补齐。"""
    parts = re.split(r"[|、,，\s]+", gen.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return (parts + [""] * k)[:k]

# ─────────── 推理 ───────────
candidates, top1 = [], []
print('🚀 推理 …')
for i in tqdm(range(0, len(df), args.batch_size)):
    batch_txt = df[args.text_col].iloc[i:i+args.batch_size].tolist()
    outs = llm.generate(make_prompts(batch_txt), sampling)
    for o in outs:
        gen = o.outputs[0].text
        cand = split_labels(gen, args.top_k)
        candidates.append(cand)
        top1.append(cand[0])

# ─────────── 评估 ───────────
truth = df[args.label_col].tolist()
acc_k = sum(t in cand for t, cand in zip(truth, candidates)) / len(df)
acc1  = accuracy_score(truth, top1)

print(f"\n🔎 Accuracy@{args.top_k}: {acc_k:.4f}")
print(f"🔎 Top-1 Accuracy   : {acc1:.4f}\n")
print("—— Top-1 classification report ——")
print(classification_report(truth, top1, digits=4))

# # ─────────── 保存 ───────────
# df['候选标签'] = ["|".join(c) for c in candidates]
# df['top1'] = top1
# df.to_csv(args.out_csv, index=False, encoding='utf-8')
# print(f'✅ 结果已保存到: {args.out_csv}')

# id2label = {i: l for i, l in enumerate(sorted(labels))}
# with open(args.out_csv.replace('.csv', '_id2label.json'), 'w', encoding='utf-8') as f:
#     json.dump(id2label, f, ensure_ascii=False, indent=2)
# print('✅ id2label.json 已生成')