import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse

def split_dataset(input_csv, output_dir=None, train_prefix="multi_inputs_train", dev_prefix="multi_inputs_dev", seed=42):
    """
    将CSV数据集按1:9比例划分为训练集和验证集
    
    参数:
    input_csv: 输入CSV文件路径
    output_dir: 输出目录（可选）
    train_prefix: 训练集文件名前缀
    dev_prefix: 验证集文件名前缀
    seed: 随机种子（确保可复现性）
    
    返回:
    训练集和验证集的文件路径
    """
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
    print("\n前5行数据:")
    print(df.head())
    
    # 随机打乱数据（不包括列名）
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 计算划分点
    split_index = int(len(df) * 0.9)
    
    # 划分数据集
    train_df = df.iloc[:split_index]
    dev_df = df.iloc[split_index:]
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 创建文件名
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
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
        
        # 验证输出文件的第一行
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"\n训练集第一行: {first_line}")
        
        with open(dev_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"验证集第一行: {first_line}")
        
        return train_path, dev_path
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None, None

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将CSV数据集按1:9比例划分为训练集和验证集')
    parser.add_argument('--input_csv', type=str, help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（可选，默认42）')
    
    args = parser.parse_args()
    
    print(f"开始处理: {args.input_csv}")
    print(f"随机种子: {args.seed}")
    
    # 划分数据集
    train_path, dev_path = split_dataset(
        args.input_csv,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if train_path and dev_path:
        print("\n处理完成！")
    else:
        print("\n处理失败，请检查错误信息")