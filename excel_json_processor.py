import json
import os
import pandas as pd

def update_json_with_excel(excel_path, json_path, output_path, sheet_name=0):
    """
    从Excel文件提取H列数据，与JSON文件内容比对
    将不存在的新项添加到JSON末尾，并保存到新文件
    
    :param excel_path: Excel文件路径 (.xlsx)
    :param json_path: JSON文件路径
    :param output_path: 输出JSON文件路径
    :param sheet_name: 工作表名称或索引，默认为第一个工作表
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 获取JSON中所有值和当前最大索引
    existing_values = set(json_data.values())
    max_index = max(int(k) for k in json_data.keys()) if json_data else -1
    
    # 从Excel读取数据
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        
        # 检查是否有足够的列
        if len(df.columns) < 8:
            raise ValueError(f"Excel文件至少需要8列，但只有{len(df.columns)}列")
        
        # 获取H列数据（第8列，索引为7）
        h_column = df.iloc[:, 7].dropna().astype(str).str.strip()
        h_column = h_column[h_column != '']
        
        # 找出不在JSON中的新项
        new_items = []
        for value in h_column.unique():
            if value not in existing_values:
                existing_values.add(value)
                new_items.append(value)
    
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return
    
    # 添加新项到JSON数据
    for item in new_items:
        max_index += 1
        json_data[str(max_index)] = item
    
    # 保存更新后的JSON到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return new_items

if __name__ == "__main__":
    # 配置文件路径（请根据实际情况修改）
    EXCEL_PATH = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/datasets/0717/all_data_0717.xlsx"     # 输入的Excel文件路径
    JSON_PATH = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label.json"      # 原始的JSON文件路径
    OUTPUT_PATH = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/train_id2label_0717.json" # 输出的JSON文件路径
    SHEET_NAME = 0                     # 工作表名称或索引（0表示第一个工作表）
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 执行更新操作
    new_items = update_json_with_excel(EXCEL_PATH, JSON_PATH, OUTPUT_PATH, SHEET_NAME)
    
    if new_items is not None:
        print(f"处理完成! 添加了 {len(new_items)} 个新项")
        print(f"新增项: {', '.join(new_items)}")
        print(f"更新后的JSON已保存至: {OUTPUT_PATH}")