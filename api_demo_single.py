"""
单变量输入方案的API服务
使用规则库+模型的方案
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import json, os

# 配置
TEXT_COL = '资产名称'
LABEL_COL = '新国标分类'
RULE_XLSX = '新国标固定资产.xlsx'
MODEL_DIR = 'model_output'
ID2LABEL_PATH = 'train_id2label.json'
TOP_K = 5

# 1. id ↔ label 映射
id2label = {}
if os.path.exists(ID2LABEL_PATH):
    id2label = {str(k): str(v) for k, v in json.load(open(ID2LABEL_PATH, encoding='utf-8')).items()}

# 2. 规则库
rule_df = pd.read_excel(RULE_XLSX)[[TEXT_COL, LABEL_COL]].dropna()
rule_map = defaultdict(list)
for _, r in rule_df.iterrows():
    rule_map[str(r[TEXT_COL]).strip().lower()].append(str(r[LABEL_COL]).strip())
for k in rule_map:
    seen = set()
    rule_map[k] = [x for x in rule_map[k] if not (x in seen or seen.add(x))]

def rule_lookup(text):
    return rule_map.get(text.strip().lower(), [])

# 3. 加载模型
use_fallback = True
if use_fallback:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, ignore_mismatched_sizes=True).to(device).eval()

    @torch.no_grad()
    def model_predict(texts):
        ds = Dataset.from_dict({TEXT_COL: texts})
        ds = ds.map(lambda x: tokenizer(
            x[TEXT_COL], truncation=True, padding='max_length', max_length=32), batched=True)
        ds.set_format('pt', columns=['input_ids', 'attention_mask'])
        loader = torch.utils.data.DataLoader(ds, batch_size=32)
        out = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu()
            top_idx = logits.topk(TOP_K, dim=-1).indices
            out.extend([[id2label.get(str(int(i)), str(int(i))) for i in row] for row in top_idx])
        return out

# 4. FastAPI服务
app = FastAPI()

class AssetRequest(BaseModel):
    资产名称: str

@app.post("/predict")
async def predict(req: AssetRequest):
    text = req.资产名称
    labs = rule_lookup(text)
    if labs and len(labs) >= TOP_K:
        candidates = labs[:TOP_K]
    elif labs and len(labs) < TOP_K:
        # 规则命中但不足5个，用模型补足
        model_cands = model_predict([text])[0]
        fill = [x for x in model_cands if x not in labs]
        candidates = labs + fill[:TOP_K - len(labs)]
    else:
        candidates = model_predict([text])[0]
    return {
        "top_1": candidates[0],
        "top_5": candidates[:TOP_K]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=22)