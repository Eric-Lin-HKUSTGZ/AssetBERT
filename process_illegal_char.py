"""
将型号类别中的非法字符（*、-、/、无）转换为合法字符（[型号缺失]）。
"""
import pandas as pd
import os
from datetime import datetime

def replace_model_values(input_path, output_dir=None):
    """
    处理Excel表格，替换型号列中的*、-、\为[型号缺失]
    
    参数:
    input_path: 输入Excel文件路径
    output_dir: 输出目录（可选），默认与输入文件同目录
    
    返回:
    新文件的保存路径
    """
    # 读取Excel文件
    
    df = pd.read_csv(input_path)
   
    # 检查列名
    required_columns = ['资产名称', '型号', '用途', '新国标分类']
    if not all(col in df.columns for col in required_columns):
        print("错误：表格缺少必要的列名")
        print(f"需要的列名: {required_columns}")
        print(f"实际列名: {list(df.columns)}")
        return None
    
    # 替换型号列中的特定值
    replace_chars = ['*', '-', '/','无','']
    
    def replace_model_value(x):
        """替换型号值，包括空值和特定字符"""
        # 转换为字符串并去除空格
        x_str = str(x).strip()
        
        # 检查是否为空值或特定字符
        if x_str in replace_chars or x_str == '' or x_str == 'nan' or pd.isna(x):
            return '[型号缺失]'
        else:
            return x
    
    df['型号'] = df['型号'].apply(replace_model_value)
    
    # 设置输出路径
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # 创建带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(input_path)
    new_filename = f"legal_{filename}"
    output_path = os.path.join(output_dir, new_filename)
    
    # 保存处理后的文件
    
    df.to_csv(output_path, index=False)
    print(f"文件已保存至: {output_path}")
    print(f"处理了 {len(df)} 条记录")
    
    # 统计替换情况
    replaced_count = (df['型号'] == '[型号缺失]').sum()
    print(f"替换了 {replaced_count} 个型号缺失标记")
    
   

if __name__ == "__main__":
    # 示例使用
    input_file = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/datasets/0717/multi_inputs_0717_train.csv"  # 替换为你的Excel文件路径
    output_directory = "/hpc2hdd/home/qxiao183/linweiquan/AssetBERT/datasets/0717"  # 替换为你想要的输出目录
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 处理文件
    result_path = replace_model_values(input_file, output_directory)
   