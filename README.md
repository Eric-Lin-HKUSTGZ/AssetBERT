# AssetBERT - 资产分类系统

## 📜 项目简介
本项目基于深度学习和规则引擎实现新国标资产分类任务。用户输入资产名称等信息，系统通过多模态处理机制输出标准化的资产分类结果。核心功能包括：
- 🧠 基于RoBERTa的AssetBERT模型（支持单变量/多变量输入）
- ⚖️ 规则+模型双阶段推理引擎
- 🌐 支持API服务化部署
- 📊 全流程数据处理工具链

<!-- ## 📚 文件功能说明

### 模型训练
| 文件路径 | 功能描述 |
|---------|---------|
| `train_classifier.py` | 单变量输入模型训练 |
| `train_multi_inputsclassifier.py` | 多变量输入基础模型训练 |
| `train_multi_inputsclassifier_advance.py` | 多变量输入增强训练（含新训练策略） |
| `train_stacking_deep.py` | 三模型集成学习训练（含RWKV、MLP和Transformer） |
| `train_stacking_gbm.py` | 三模型集成学习训练（LightGBM方案） |

### 数据预处理
| 文件路径 | 功能描述 |
|---------|---------|
| `split_dataset.py` | 训练集/测试集划分 |
| `process_illegal_char.py` | 无效字符转换处理 |
| `extract_colunms.py` | Excel列数据抽取（规则构建） |
| `convert_xlsx_data.py` | Excel数据抽取（数据集构建） |
| `convert_csv_label.py` | 标签格式转换（字符串↔数字） |
| `data_cleaning.py` | 数据清洗主程序 |

### 推理服务
| 文件路径 | 功能描述 |
|---------|---------|
| `infer_service.py` | 单输入规则+模型推理 |
| `infer_mi_om_service.py` | 纯多输入模型推理 |
| `infer_mi_om_assemble_service.py` | 三模型加权投票推理 |
| `infer_mi_rm_service.py` | 规则+多输入模型推理 |
| `infer_mi_mr_service.py` | 多输入模型推理+规则 |
| `test_assets_vllm.py` | 模型性能测试工具 |

### API服务
| 文件路径 | 功能描述 |
|---------|---------|
| `api_demo_single.py` | 单输入模型API服务 |
| `api_demo.py` | 多输入模型API服务 |

### 关键资源
| 文件路径 | 功能描述 |
|---------|---------|
| `rule_新国标分类测试集.xlsx` | 规则判断依据文件 |
| `新国标固定资产.xlsx` | 原始分类标准参考 |
| `datasets/` | 训练集/测试集存储目录 |
| `model_output/` | 模型权重存储目录 |
| `train_id2label.json` | 标签映射配置文件 | -->

## 🏗️ 项目结构
```bash
AssetBERT/
├── datasets/                # 存放训练集和测试集
│   ├── multi_inputs_train.csv
│   ├── multi_inputs_dev.csv
│   └── single_input_train.csv
│       └── single_input_dev.csv
├── model_output/            # 模型权重保存目录
│   ├── multi_inputs/
│   │   └── [timestamp]/     # 每次训练生成的时间戳目录
│   │   └── stacking/        # 集成学习模型保存路径
│   └── single_input/
│       └── [timestamp]/
├── .gitignore               # 忽略文件配置
├── api_demo_multi.py        # 基于多输入模型的API服务代码
├── api_demo_single.py       # 基于单输入模型的API服务代码
├── api_demo.py              # 部署API服务的代码，模型+规则补充方案
├── convert_csv_label.py     # 将数据中的字符串标签转换为数字标签
├── convert_xlsx_data.py     # 列数据抽取代码，用于构建数据集
├── data_cleaning.py         # 数据清洗脚本
├── extract_columns.py       # Excel 文件列数据抽取代码
├── infer_mi_mr_service.py   # 多输入，模型+规则补充推理代码
├── infer_mi_rm_service.py   # 多输入，规则+模型补充推理代码
├── infer_mi_om_service.py   # 多输入模型推理代码
├── infer_mi_om_assemble_service.py   # 三模型集成学习推理代码
├── infer_service.py         # 单输入规则+模型推理代码
├── process_illegal_char.py  # 无效字符转换代码
├── excel_json_processor.py  # 根据数据集创建新的id2label文件
├── rule_新国标分类测试集.xlsx  # 规则判断依据文件
├── rule_all_data_0717.xlsx  # 根据最新数据构建的规则判断依据文件
├── split_dataset.py         # 划分训练和测试集代码
├── test_assets_vilm.py      # 测试脚本
├── train_classifier.py       # 单变量输入模型训练文件
├── train_multi_inputs_classifier.py  # 多变量输入模型训练文件
├── train_multi_inputs_classifier_advance.py  # 带有高级训练策略的多变量输入模型训练文件
├── train_multi_inputs_classifier_top3.py  # 加入使用部门字段，输出top3
├── train_stacking_gbm.py  # 基于LightGBM的集成学习训练
├── train_stacking_deep.py  # 基于深度网络（RWKV、MLP、Transformer）的集成学习训练
└── README.md                # 项目说明文档
```

## API请求
### 启动命令：
```bash
nohup python api_demo.py &
```

### 单量请求
请求指令：
```bash
curl -X POST "http://10.120.20.176:20805/predict" -H "Content-Type: application/json" -d '{"资产名称": "六分量天平", "型号": "5N", "用途": "科研用", "使用部门": "生物启发工程中央实验室"}'
```
输出：
```bash
{"code":0,"success":true,"message":"预测成功","data":{"top_1":"分析天平及专用天平","top_3":["分析天平及专用天平","测力仪器","其他海洋仪器设备"]}}
```

### 批量请求
请求指令：
```bash
curl -X POST "http://10.120.20.176:20805/batch_predict"  -H "Content-Type: application/json"  -d '[
       {
         "资产名称": "显示器",
         "型号": "DELL P2422H",
         "用途": "行政办公用"
       },
       {
         "资产名称": "学习椅",
         "型号": "椅子1 SHCT5ACH",
         "用途": "培训教学"
       },
       {
         "资产名称": "屏风工位",
         "型号": "长江",
         "用途": "后勤保障用"
       }]'
```
输出：
```bash
{"code":0,"success":true,"message":"批量预测成功，共处理 3 个样本","data":{"results":[{"top_1":"液晶显示器","top_5":["液晶显示器","其他办公设备","等离子显示器","其他信息化设备","其他计算机"]},{"top_1":"教学、实验用桌","top_5":["教学、实验用桌","教学仪器","试验箱及气候环境试验设备","直流电源","其他台、桌类"]},{"top_1":"办公桌","top_5":["办公桌","组合家具","其他家具","会议桌","其他厨卫用具"]}]}}
```

## BERT变体文本分类实验报告
| 实验编号 | 模型名称                     | Hugging Face 模型版本                   | 权重路径                                | 训练策略       | 推理速度 | Top1准确率 | Top5准确率 |
|----------|------------------------------|-----------------------------------------|-----------------------------------------|----------------|----------|------------|------------|
| 1        | RoBERTa (baseline)           | `uer/roberta-base-finetuned-chinanews-chinese`           | `/model_output/multi_inputs/2025-07-14-20-21` | 标准训练       | 正常     | 0.8885     | 0.9505     |
| 2        | RoBERTa + 复杂策略           | `uer/roberta-base-finetuned-chinanews-chinese`           | `/model_output/multi_inputs/2025-07-15-13-29` | 增强策略       | 正常     | 0.8849     | 0.9544     |
| 3        | **MacBERT**                  | `hfl/chinese-macbert-base`              | `/model_output/multi_inputs/2025-07-15-17-15` | 标准训练       | 正常     | 0.8864     | 0.9518     |
| 4        | **ERNIE 3.0**                | `nghuyong/ernie-3.0-base-zh`            | `/model_output/multi_inputs/2025-07-15-17-34` | 标准训练       | 正常     | 0.8862     | 0.9493     |
| 5        | **DeBERTa**                  | `KoichiYasuoka/deberta-base-chinese`    | `/model_output/multi_inputs/2025-07-15-17-57` | 标准训练       | ⚠️ **最慢** | 0.8917     | 0.9533     |
| 6        | **WoBERT**                   | `junnyu/wobert_chinese_plus_base`       | `/model_output/multi_inputs/2025-07-15-19-45` | 标准训练       | 正常     | 0.8875     | 0.9514     |

## 0717实验记录（资产名称、型号、用途和使用部门）

### 单模型实验结果
| 模型类型 | 优化方法 | 模型路径 | top1 | top3 | 性能提升 |
|----------|----------|----------|-------|-------|----------|
| RoBERTa | 无 | `/model_output/multi_inputs/2025-07-18-11-12` | 0.9343 | 0.9604 | - |
| RoBERTa | top_k损失(未考虑原始损失和标签平滑) | `/model_output/multi_inputs/2025-07-18-12-13` | 0.9338 | 0.9617 | ✓ |
| RoBERTa | top_k损失(考虑原始损失和标签平滑) | `/model_output/multi_inputs/2025-07-18-13-29` | - | - | ✗ |
| WoBERT | top_k损失(未考虑原始损失和标签平滑) | `/model_output/multi_inputs/2025-07-18-14-47` | 0.9330 | 0.9617 | ✓ |
| MacBERT | top_k损失(未考虑原始损失和标签平滑) | `/model_output/multi_inputs/2025-07-18-16-32` | 0.9331 | 0.9618 | ✓ |

### 集成学习实验 (RoBERTa+WoBERT+MacBERT)
| 集成方法 | 参数/模型 | 模型路径 | top1 | top3 | 提升幅度 |
|----------|-----------|----------|-------|-------|----------|
| 加权投票 | 权重[0.3,0.3,0.4] | 无 | 0.9355 | 0.9634 | +0.2% |
| Meta model | MLP | /model_output/multi_inputs/stacking/meta_linear_model.pth | 0.9347 | - | - |
| Meta model | RWKV | /model_output/multi_inputs/stacking/meta_rwkv_model.pth | 0.9267 | - | - |
| Meta model | Transformer | /model_output/multi_inputs/stacking/meta_transformer_model.pth | 0.9314 | - | - |
| Meta model | LightGBM | /model_output/multi_inputs/stacking/lightgbm_model.bin | 0.8838 | - | - |

## 推理策略对比
规则+模型补充：
Accuracy@1=0.8934  Accuracy@3=0.9847

模型+规则补充（模型阈值0.9最优）：
Accuracy@1=0.9541  Accuracy@3=0.9784
