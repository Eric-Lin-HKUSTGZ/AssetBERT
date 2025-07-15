"""
从原始的Excel文件中提取特定列数据并保存为CSV格式
"""
import pandas as pd
import os

def extract_columns(input_path, output_path, multi_inputs=False):
    """
    从Excel文件中提取B列(第2列)和H列(第8列)的数据并保存到新文件
    
    参数:
    input_path (str): 原始Excel文件路径
    output_path (str): 输出文件路径
    """
    try:
        # 读取原始Excel文件
        df = pd.read_excel(input_path, sheet_name='Sheet1')
        
        # 检查文件是否有足够的列
        if df.shape[1] < 8:
            raise ValueError(f"Excel文件只有 {df.shape[1]} 列，需要至少8列才能提取H列")
        
        # 获取列名（如果存在）
        column_names = df.columns.tolist()
        
        # 提取B列(索引1)和H列(索引7)
        # 注意：Pandas列索引从0开始，所以B列是索引1，H列是索引7
        b_column = df.iloc[:, 1]  # B列数据
        h_column = df.iloc[:, 7]  # H列数据
        
        # 创建新的DataFrame
        result_df = pd.DataFrame({
            'B列数据': b_column,
            'H列数据': h_column
        })

        if multi_inputs:
            c_column = df.iloc[:, 2]  # B列数据
            f_column = df.iloc[:, 5]  # H列数据'
            result_df = pd.DataFrame({
            'B列数据': b_column,
            'C列数据': c_column,
            'F列数据': f_column,
            'H列数据': h_column
            })

        
        # 保存到新文件（CSV格式，utf-8编码）
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"提取完成！结果已保存到: {os.path.abspath(output_path)}")
        print(f"提取了 {len(result_df)} 条记录")
        print(f"预览前5行数据:\n{result_df.head()}")
        
        return True
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 文件路径配置
    input_excel = './datasets/新国标分类测试集.xlsx'  # 原始文件
    output_csv = './datasets/multi_inputs_0701.csv'              # 输出文件（CSV）
    multi_inputs = True  # 是否处理多输入文件（当前示例只处理单个文件）
    
    # 执行提取操作
    success = extract_columns(input_excel, output_csv, multi_inputs)
    
    if success:
        # 添加示例：读取并展示输出文件的前几行
        try:
            print("\n输出文件预览:")
            output_df = pd.read_csv(output_csv, encoding='utf-8')
            print(output_df.head(5))
        except Exception as e:
            print(f"预览输出文件时出错: {str(e)}")
    else:
        print("处理失败，请检查原始文件格式和路径")