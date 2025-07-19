"""
将CSV数据集按1:9比例划分为训练集和验证集，确保训练集包含所有类别"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import argparse
from sklearn.model_selection import train_test_split

def load_label_map(label_map_json):
    """加载标签映射JSON文件"""
    try:
        with open(label_map_json, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        # 获取所有类别数字（字符串形式）
        all_labels = set(label_map.keys())
        print(f"成功加载标签映射文件，共 {len(all_labels)} 个类别")
        return all_labels
    except Exception as e:
        print(f"加载标签映射文件失败: {str(e)}")
        return None

def split_dataset(input_csv, label_map_json, output_dir=None, 
                  train_prefix="multi_inputs_train", dev_prefix="multi_inputs_dev", 
                  seed=42, label_column="新国标分类"):
    """
    将CSV数据集按1:9比例划分为训练集和验证集，确保训练集包含所有类别
    
    参数:
    input_csv: 输入CSV文件路径
    label_map_json: 标签映射JSON文件路径
    output_dir: 输出目录（可选）
    train_prefix: 训练集文件名前缀
    dev_prefix: 验证集文件名前缀
    seed: 随机种子（确保可复现性）
    label_column: 包含类别标签的列名
    """
    # 加载标签映射
    all_labels = load_label_map(label_map_json)
    if all_labels is None:
        return None, None
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 读取CSV文件
    try:
        # 尝试不同编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                # 保留列名作为header
                df = pd.read_csv(input_csv, encoding=encoding, 
                                delimiter='\t' if input_csv.endswith('.tsv') else ',',
                                header=0)  # 明确指定第一行为列名
                print(f"成功读取文件，使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                # 尝试处理可能的制表符分隔文件
                try:
                    df = pd.read_csv(input_csv, encoding=encoding, delimiter='\t', header=0)
                    print(f"成功读取TSV文件，使用编码: {encoding}")
                    break
                except:
                    continue
        
        if df is None:
            print("错误：无法读取文件，请检查文件格式或编码")
            return None, None
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None
    
    # 检查数据是否为空
    if df.empty:
        print("错误：文件为空")
        return None, None
    
    # 显示数据信息
    print("\n数据摘要:")
    print(f"总记录数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    # 确保标签列存在
    if label_column not in df.columns:
        print(f"错误：数据集中不存在标签列 '{label_column}'")
        return None, None
    
    # 将标签列转换为字符串（确保与JSON键匹配）
    df[label_column] = df[label_column].astype(str)
    
    # 1. 检查数据集中的标签是否都在标签映射中
    unique_labels = set(df[label_column].unique())
    missing_labels = unique_labels - all_labels
    
    if missing_labels:
        print(f"\n警告：数据集中有 {len(missing_labels)} 个标签不在标签映射文件中:")
        print(list(missing_labels)[:10])  # 最多显示前10个
        print("这些记录将被排除")
        
        # 过滤掉无效标签的记录
        valid_mask = df[label_column].isin(all_labels)
        df = df[valid_mask].copy()
        
        print(f"过滤后剩余记录数: {len(df)}")
    
    # 2. 检查是否有缺失标签
    missing_in_data = all_labels - unique_labels
    if missing_in_data:
        print(f"\n警告：标签映射中有 {len(missing_in_data)} 个标签在数据集中不存在:")
        print(list(missing_in_data)[:10])  # 最多显示前10个
        print("这些标签将不会出现在训练数据中")
    
    # 3. 分层抽样 - 确保每个类别在训练集中都有代表性
    print("\n开始分层抽样...")
    
    # 按标签分组
    grouped = df.groupby(label_column)
    
    train_dfs = []
    dev_dfs = []
    
    # 对每个类别单独抽样
    for label, group in grouped:
        # 如果类别样本少于2个，全部放入训练集
        if len(group) <= 1:
            train_dfs.append(group)
            print(f"类别 {label}: 只有 {len(group)} 个样本，全部放入训练集")
            continue
        
        # 按9:1比例划分
        train_group, dev_group = train_test_split(
            group, test_size=0.1, random_state=seed
        )
        
        train_dfs.append(train_group)
        dev_dfs.append(dev_group)
        
        print(f"类别 {label}: 训练集 {len(train_group)} 个样本, 验证集 {len(dev_group)} 个样本")
    
    # 合并所有类别的训练集和验证集
    train_df = pd.concat(train_dfs)
    dev_df = pd.concat(dev_dfs)
    
    # 随机打乱数据（保持分层抽样后的分布）
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    dev_df = dev_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 检查训练集是否包含所有类别
    train_labels = set(train_df[label_column].unique())
    missing_in_train = all_labels - train_labels
    
    if missing_in_train:
        print("\n警告：训练集缺少以下类别:")
        print(list(missing_in_train))
        
        # 对于缺失的类别，从原始数据中随机取一个样本加入训练集
        print("从原始数据中添加样本到训练集...")
        for label in missing_in_train:
            # 获取该类别所有样本
            label_samples = df[df[label_column] == label]
            if len(label_samples) > 0:
                # 随机选择一个样本加入训练集
                sample = label_samples.sample(1, random_state=seed)
                train_df = pd.concat([train_df, sample])
                print(f"添加类别 {label} 的样本到训练集")
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文件名
    train_filename = f"{train_prefix}.csv"
    dev_filename = f"{dev_prefix}.csv"
    
    train_path = os.path.join(output_dir, train_filename)
    dev_path = os.path.join(output_dir, dev_filename)
    
    # 保存文件（包含列名）
    try:
        train_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)
        
        print("\n数据集划分结果:")
        print(f"训练集大小: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"验证集大小: {len(dev_df)} 条 ({len(dev_df)/len(df)*100:.1f}%)")
        print(f"训练集保存至: {train_path}")
        print(f"验证集保存至: {dev_path}")
        
        # 显示训练集中的类别分布
        print("\n训练集类别分布:")
        train_label_counts = train_df[label_column].value_counts()
        print(train_label_counts)
        
        return train_path, dev_path
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None, None

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将CSV数据集按1:9比例划分为训练集和验证集，确保训练集包含所有类别')
    parser.add_argument('--input_csv', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--label_map_json', type=str, required=True, help='标签映射JSON文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（可选，默认42）')
    
    args = parser.parse_args()
    
    print(f"开始处理: {args.input_csv}")
    print(f"标签映射文件: {args.label_map_json}")
    print(f"随机种子: {args.seed}")
    
    # 划分数据集
    train_path, dev_path = split_dataset(
        input_csv=args.input_csv,
        label_map_json=args.label_map_json,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if train_path and dev_path:
        print("\n处理完成！")
    else:
        print("\n处理失败，请检查错误信息")