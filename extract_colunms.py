"""
从Excel文件中提取特定列的数据并保存到新文件,用于规则判断输出
"""
import pandas as pd
import argparse
import os

def extract_columns(input_file, output_file):
    """
    从Excel文件中提取特定列的数据并保存到新文件
    
    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    
    提取的列:
    B列 (索引1), C列 (索引2), F列 (索引5), H列 (索引7)
    从第二行开始提取数据
    """
    try:
        # 读取Excel文件，跳过第一行（标题行）
        df = pd.read_excel(input_file, skiprows=1)
        
        # 检查文件是否有足够的列
        if df.shape[1] < 8:
            raise ValueError(f"输入文件需要至少8列，但只有{df.shape[1]}列")
        
        # 提取指定的列
        # 列索引: B=1, C=2, D=3,  H=7
        extracted_columns = df.iloc[:, [1, 2, 5, 7]]
        
        # 设置列名
        column_names = ['资产名称', '型号', '用途', '新国标分类']
        extracted_columns.columns = column_names
        
        # 保存到新文件
        extracted_columns.to_excel(output_file, index=False)
        
        print(f"成功提取并保存数据到: {output_file}")
        print(f"提取了 {len(extracted_columns)} 行和 {len(extracted_columns.columns)} 列数据")
        print(f"列名: {list(extracted_columns.columns)}")
        
        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='从CSV文件中提取特定列的数据')
    parser.add_argument('--input_file', type=str, help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建输出文件名
    input_basename = os.path.basename(args.input_file)
    output_filename = f"rule_{input_basename}"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"开始处理: {args.input_file}")
    print(f"将提取以下列: B列, C列, F列, H列")
    
    # 执行提取
    success = extract_columns(args.input_file, output_path)
    
    if success:
        print("处理完成!")
    else:
        print("处理失败，请检查错误信息")