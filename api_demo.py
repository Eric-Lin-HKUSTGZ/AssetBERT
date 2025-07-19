"""
多变量输入的API服务 - 模型优先+规则补充方案
使用模型优先+规则补充的方案
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Union, List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import json, os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
TEXT_COL = '资产名称'
LABEL_COL = '新国标分类'
RULE_XLSX = 'rule_all_data_0717.xlsx'
MODEL_DIR = './model_output/multi_inputs/2025-07-18-12-13'
ID2LABEL_PATH = 'train_id2label_0717.json'
TOP_K = 3
MAX_LENGTH = 64
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.9  # 模型置信度阈值

# 1. id ↔ label 映射
id2label = {}
if os.path.exists(ID2LABEL_PATH):
    with open(ID2LABEL_PATH, 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    logger.info(f"加载标签映射，共 {len(id2label)} 个类别")
else:
    logger.error(f"标签映射文件不存在: {ID2LABEL_PATH}")
    raise FileNotFoundError(f"标签映射文件不存在: {ID2LABEL_PATH}")

# 2. 规则库
logger.info("加载规则库...")
rule_df = pd.read_excel(RULE_XLSX)[[TEXT_COL, LABEL_COL]].dropna()
rule_map = defaultdict(list)
for _, r in rule_df.iterrows():
    rule_map[str(r[TEXT_COL]).strip().lower()].append(str(r[LABEL_COL]).strip())
for k in rule_map:
    seen = set()
    rule_map[k] = [x for x in rule_map[k] if not (x in seen or seen.add(x))]
logger.info(f"规则库加载完成，共 {len(rule_map)} 条规则")

def rule_lookup(text):
    return rule_map.get(text.strip().lower(), [])

# 3. 加载模型
use_fallback = True
if use_fallback:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    logger.info(f"模型置信度阈值: {CONFIDENCE_THRESHOLD}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, ignore_mismatched_sizes=True).to(device).eval()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise RuntimeError(f"模型加载失败: {e}")

    @torch.no_grad()
    def model_predict_single(asset_name, model_name, purpose, department):
        """
        单样本模型预测函数
        """
        combined_text = f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose} 使用部门: {department}"
        
        tokenized = tokenizer(
            [combined_text],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        batch_inputs = {k: v.to(device) for k, v in tokenized.items()}
        outputs = model(**batch_inputs)
        logits = outputs.logits.cpu()
        
        # 获取top-k预测和置信度
        top_idx = logits.topk(TOP_K, dim=-1).indices[0]
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        predictions = []
        for idx in top_idx:
            pred_label = id2label.get(str(int(idx)), "未知")
            predictions.append(pred_label)
        
        return predictions, confidence

    @torch.no_grad()
    def model_predict_batch(asset_names, model_names, purposes, departments):
        """
        批量模型预测函数
        """
        # 构建多输入文本
        combined_texts = []
        for asset_name, model_name, purpose, department in zip(asset_names, model_names, purposes, departments):
            combined_text = f"资产名称: {asset_name} 型号: {model_name} 用途: {purpose} 使用部门: {department}"
            combined_texts.append(combined_text)
        
        all_predictions = []
        all_confidences = []
        
        # 分批处理
        for i in range(0, len(combined_texts), BATCH_SIZE):
            batch_texts = combined_texts[i:i+BATCH_SIZE]
            batch_tokenized = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            
            batch_inputs = {k: v.to(device) for k, v in batch_tokenized.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits.cpu()
            
            # 获取top-k预测和置信度
            top_idx = logits.topk(TOP_K, dim=-1).indices
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
            
            for indices, conf in zip(top_idx, confidences):
                predictions = []
                for idx in indices:
                    pred_label = id2label.get(str(int(idx)), "未知")
                    predictions.append(pred_label)
                all_predictions.append(predictions)
                all_confidences.append(conf.item())
        
        return all_predictions, all_confidences

# 4. FastAPI服务
app = FastAPI(
    title="多输入资产分类API - 模型优先方案",
    description="基于模型优先+规则补充的多输入资产分类服务",
    version="1.0.0"
)

class AssetRequest(BaseModel):
    资产名称: str
    型号: str = ""
    用途: str = ""
    使用部门: str = ""

class AssetData(BaseModel):
    top1: str
    top3: list[str]

class APIResponse(BaseModel):
    code: int
    success: bool
    message: str
    data: Any = None

@app.get("/")
async def root():
    return APIResponse(
        code=0,
        success=True,
        message="多输入资产分类API服务运行正常",
        data={
            "service": "多输入资产分类API - 模型优先方案",
            "version": "1.0.0",
            "status": "running",
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    )

@app.get("/health")
async def health_check():
    return APIResponse(
        code=0,
        success=True,
        message="服务健康检查通过",
        data={
            "status": "healthy",
            "model_loaded": use_fallback,
            "device": device if use_fallback else "N/A",
            "rule_count": len(rule_map),
            "label_count": len(id2label),
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    )

@app.post("/predict", response_model=APIResponse)
async def predict(req: AssetRequest):
    """
    资产分类预测接口 - 模型优先+规则补充
    
    Args:
        req: 包含资产名称、型号、用途、使用部门的请求对象
        
    Returns:
        包含code、success、message和data的标准API响应
    """
    try:
        asset_name = req.资产名称.strip()
        model_name = req.型号.strip() if req.型号 else "[型号缺失]"
        purpose = req.用途.strip() if req.用途 else "[用途缺失]"
        department = req.使用部门.strip() if req.使用部门 else "[部门缺失]"
        
        if not asset_name:
            return APIResponse(
                code=400,
                success=False,
                message="资产名称不能为空",
                data=None
            )
        
        # 模型优先 + 规则补充
        if use_fallback:
            # 先使用模型预测
            model_predictions, confidence = model_predict_single(asset_name, model_name, purpose, department)
            
            # 如果模型置信度低于阈值，尝试使用规则补充
            if confidence < CONFIDENCE_THRESHOLD:
                rule_labs = rule_lookup(asset_name)
                if rule_labs:
                    # 规则命中，检查是否需要补充
                    if len(rule_labs) >= TOP_K:
                        # 规则结果足够，使用规则结果
                        candidates = rule_labs[:TOP_K]
                        method = "规则"
                        confidence = None
                    else:
                        # 规则结果不足，用模型结果补充
                        existing_labels = set(rule_labs)
                        additional_labels = [label for label in model_predictions if label not in existing_labels]
                        final_predictions = rule_labs + additional_labels[:TOP_K - len(rule_labs)]
                        candidates = final_predictions
                        method = "规则+模型补充"
                        confidence = None
                else:
                    # 规则未命中，仍使用模型预测
                    candidates = model_predictions
                    method = "模型(低置信度)"
            else:
                # 模型置信度足够，直接使用模型预测
                candidates = model_predictions
                method = "模型"
        else:
            # 不使用模型，仅使用规则
            rule_labs = rule_lookup(asset_name)
            if rule_labs:
                candidates = rule_labs[:TOP_K]
                method = "规则"
                confidence = None
            else:
                candidates = [''] * TOP_K
                method = "无匹配"
                confidence = None
        
        return APIResponse(
            code=0,
            success=True,
            message="预测成功",
            data=AssetData(
                top1=candidates[0] if candidates[0] else "未知",
                top3=candidates[:TOP_K]
            )
        )
        
    except Exception as e:
        logger.error(f"预测过程中出现错误: {e}")
        return APIResponse(
            code=500,
            success=False,
            message=f"预测失败: {str(e)}",
            data=None
        )

@app.post("/batch_predict", response_model=APIResponse)
async def batch_predict(requests: list[AssetRequest]):
    """
    批量资产分类预测接口 - 模型优先+规则补充
    
    Args:
        requests: 包含多个资产信息的请求列表
        
    Returns:
        包含code、success、message和data的标准API响应
    """
    try:
        if not requests:
            return APIResponse(
                code=400,
                success=False,
                message="请求列表不能为空",
                data=None
            )
        
        results = []
        
        # 提取输入数据
        asset_names = [req.资产名称.strip() for req in requests]
        model_names = [req.型号.strip() if req.型号 else "[型号缺失]" for req in requests]
        purposes = [req.用途.strip() if req.用途 else "[用途缺失]" for req in requests]
        departments = [req.使用部门.strip() if req.使用部门 else "[部门缺失]" for req in requests]
        
        # 验证输入
        valid_indices = []
        valid_asset_names = []
        valid_model_names = []
        valid_purposes = []
        valid_departments = []
        
        for i, asset_name in enumerate(asset_names):
            if not asset_name:
                results.append({
                    "error": "资产名称不能为空",
                    "top1": "未知",
                    "top3": ["未知"] * TOP_K
                })
            else:
                valid_indices.append(i)
                valid_asset_names.append(asset_name)
                valid_model_names.append(model_names[i])
                valid_purposes.append(purposes[i])
                valid_departments.append(departments[i])
        
        if not valid_asset_names:
            return APIResponse(
                code=400,
                success=False,
                message="没有有效的输入数据",
                data={"results": results}
            )
        
        # 批量模型预测
        if use_fallback:
            model_predictions, confidences = model_predict_batch(
                valid_asset_names, valid_model_names, valid_purposes, valid_departments
            )
            
            # 根据置信度决定是否使用规则
            for i, (model_pred, confidence) in enumerate(zip(model_predictions, confidences)):
                asset_name = valid_asset_names[i]
                
                # 如果模型置信度低于阈值，尝试使用规则补充
                if confidence < CONFIDENCE_THRESHOLD:
                    rule_labs = rule_lookup(asset_name)
                    if rule_labs:
                        # 规则命中，检查是否需要补充
                        if len(rule_labs) >= TOP_K:
                            # 规则结果足够，使用规则结果
                            candidates = rule_labs[:TOP_K]
                            method = "规则"
                            confidence = None
                        else:
                            # 规则结果不足，用模型结果补充
                            existing_labels = set(rule_labs)
                            additional_labels = [label for label in model_pred if label not in existing_labels]
                            final_predictions = rule_labs + additional_labels[:TOP_K - len(rule_labs)]
                            candidates = final_predictions
                            method = "规则+模型补充"
                            confidence = None
                    else:
                        # 规则未命中，仍使用模型预测
                        candidates = model_pred
                        method = "模型(低置信度)"
                else:
                    # 模型置信度足够，直接使用模型预测
                    candidates = model_pred
                    method = "模型"
        else:
            # 不使用模型，仅使用规则
            for i, asset_name in enumerate(valid_asset_names):
                rule_labs = rule_lookup(asset_name)
                if rule_labs:
                    candidates = rule_labs[:TOP_K]
                    method = "规则"
                    confidence = None
                else:
                    candidates = [''] * TOP_K
                    method = "无匹配"
                    confidence = None
                
                results.append({
                    "top1": candidates[0] if candidates[0] else "未知",
                    "top3": candidates[:TOP_K]
                })
            
            return APIResponse(
                code=0,
                success=True,
                message=f"批量预测成功，共处理 {len(results)} 个样本",
                data={"results": results}
            )
        
        # 将结果填入正确位置
        for i, valid_idx in enumerate(valid_indices):
            if i < len(model_predictions):
                candidates = model_predictions[i]
                confidence = confidences[i]
                
                # 根据置信度决定是否使用规则
                if confidence < CONFIDENCE_THRESHOLD:
                    asset_name = valid_asset_names[i]
                    rule_labs = rule_lookup(asset_name)
                    if rule_labs:
                        if len(rule_labs) >= TOP_K:
                            candidates = rule_labs[:TOP_K]
                            method = "规则"
                            confidence = None
                        else:
                            existing_labels = set(rule_labs)
                            additional_labels = [label for label in candidates if label not in existing_labels]
                            final_predictions = rule_labs + additional_labels[:TOP_K - len(rule_labs)]
                            candidates = final_predictions
                            method = "规则+模型补充"
                            confidence = None
                    else:
                        method = "模型(低置信度)"
                else:
                    method = "模型"
                
                results.insert(valid_idx, {
                    "top1": candidates[0] if candidates[0] else "未知",
                    "top3": candidates[:TOP_K]
                })
        
        return APIResponse(
            code=0,
            success=True,
            message=f"批量预测成功，共处理 {len(results)} 个样本",
            data={"results": results}
        )
        
    except Exception as e:
        logger.error(f"批量预测过程中出现错误: {e}")
        return APIResponse(
            code=500,
            success=False,
            message=f"批量预测失败: {str(e)}",
            data=None
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

"""
使用示例:

# 单样本预测
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "资产名称": "显示器",
         "型号": "DELL P2422H",
         "用途": "行政办公用",
         "使用部门": "综合事务处"
      }'

# 批量预测
curl -X POST "http://localhost:8080/batch_predict" \
     -H "Content-Type: application/json" \
     -d '[
         {
             "资产名称": "显示器",
             "型号": "DELL P2422H",
             "用途": "行政办公用",
             "使用部门": "综合事务处"
         },
         {
             "资产名称": "笔记本电脑",
             "型号": "ThinkPad X1",
             "用途": "办公用",
             "使用部门": "技术部"
         }
     ]'
"""
