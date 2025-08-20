# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced Tools for Multi-Substance Multi-Target Descriptor Generation."""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from concurrent.futures import ProcessPoolExecutor
import warnings
import multiprocessing
import re
warnings.filterwarnings('ignore')

def calculate_descriptors(smiles):
    """计算单个SMILES的分子描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        calc = Calculator(descriptors, ignore_3D=False)
        descriptor_values = calc(mol)
        descriptors_dict = {}
        for key, value in descriptor_values.items():
            if value is not None:
                try:
                    float_value = float(value)
                    if not np.isnan(float_value) and not np.isinf(float_value):
                        descriptors_dict[str(key)] = float_value
                except (ValueError, TypeError):
                    continue
        return descriptors_dict
    except Exception as e:
        return None

def calculate_descriptors_parallel(smiles_list):
    """并行计算多个SMILES的分子描述符"""
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    valid_smiles = [s for s in smiles_list if isinstance(s, str) and s.strip() and s != 'nan']
    invalid_num = len(smiles_list) - len(valid_smiles)
    
    if not valid_smiles:
        return [], len(smiles_list)
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(calculate_descriptors, valid_smiles))
        return results, invalid_num
    except Exception as e:
        return [], len(smiles_list)

def clean_filename(filename):
    """清理文件名，替换特殊字符为下划线"""
    filename = str(filename)
    filename = filename.strip()
    invalid_chars = r'[\\/*?:"<>|\ \t\n\r]'
    return re.sub(invalid_chars, '_', filename)

def generate_multi_substance_descriptors(file_path: str, tool_context=None) -> str:
    """
    生成多物质多目标变量的描述符
    
    CSV格式要求：
    - 第一列：实验次数
    - 每三列代表一个物质：SubstanceX_Name, SubstanceX_SMILE, SubstanceX_ratio
    - 物质之间空一列分隔
    - 最后是目标变量：Target_a, Target_b, ...
    
    Args:
        file_path: CSV文件路径
        tool_context: 工具上下文（可选）
    
    Returns:
        处理结果信息字符串
    """
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 分析列结构
        columns = df.columns.tolist()
        
        # 识别物质列和目标列
        substance_groups = []
        target_columns = []
        
        i = 1  # 跳过第一列（实验次数）
        while i < len(columns):
            col_name = columns[i]
            
            # 检查是否是物质列组
            if 'Substance' in col_name and 'name' in col_name:
                # 提取物质名称（如SubstanceA, SubstanceB等）
                substance_name = col_name.split('_')[0]  # 获取SubstanceA部分
                
                # 检查后续两列是否存在
                if i + 2 < len(columns):
                    smile_col = columns[i + 1]
                    ratio_col = columns[i + 2]
                    
                    if 'SMILE' in smile_col and 'ratio' in ratio_col:
                        substance_groups.append({
                            'name': substance_name,
                            'name_col': col_name,
                            'smile_col': smile_col,
                            'ratio_col': ratio_col,
                            'start_index': i
                        })
                        i += 4  # 跳过三列物质列和空列
                        continue
            
            # 检查是否是目标列
            if col_name.startswith('Target_'):
                target_columns.append(col_name)
            
            i += 1
        

        
        if not substance_groups:
            return "Error: 未找到有效的物质列组"
        if not target_columns:
            return "Error: 未找到目标变量列"
        
        # 首先收集所有物质的ratio数据
        all_ratio_data = {}
        for group in substance_groups:
            substance_name = group['name']
            ratio_data = pd.to_numeric(df[group['ratio_col']].astype(str).str.strip(), errors='coerce')
            all_ratio_data[substance_name] = ratio_data
        
        # 处理每个物质组
        all_features = []
        substance_info = {}
        
        for group in substance_groups:
            substance_name = group['name']
            
            # 提取SMILES数据
            smiles_data = df[group['smile_col']].astype(str).str.strip()
            
            # 过滤有效的SMILES
            valid_mask = (smiles_data != 'nan') & (smiles_data != '') & (smiles_data.notna())
            valid_smiles = smiles_data[valid_mask].unique()

            if len(valid_smiles) == 0:
                continue
            
            
            # 计算分子描述符
            try:
                descriptors_list, invalid_count = calculate_descriptors_parallel(valid_smiles)
                
                # 创建SMILES到描述符的映射
                valid_smiles_list = list(valid_smiles)
                desc_map = {}
                for smile, desc in zip(valid_smiles_list, descriptors_list):
                    if desc is not None:
                        desc_map[smile] = desc
                
                if not desc_map:
                    continue
                
                # 创建特征矩阵
                features_df = pd.DataFrame()
                
                # 首先添加所有物质的ratio列到前面
                for ratio_substance_name, ratio_data in all_ratio_data.items():
                    features_df[f'{ratio_substance_name}_ratio'] = ratio_data
                #features_df[f'{substance_name}_ratio'] = all_ratio_data[substance_name]
                
                # 添加描述符列（保持原来的逻辑不变）
                all_desc_keys = set()
                for desc_dict in desc_map.values():
                    if desc_dict:
                        all_desc_keys.update(desc_dict.keys())
                
                for key in all_desc_keys:
                    features_df[f'{substance_name}_{key}'] = np.nan
                
                # 填充描述符值（保持原来的逻辑不变）
                for idx, row in df.iterrows():
                    smile = str(row[group['smile_col']]).strip()
                    if smile in desc_map and desc_map[smile]:
                        for key, value in desc_map[smile].items():
                            features_df.loc[idx, f'{substance_name}_{key}'] = value
                
                # 清理和填充缺失值
                features_df = features_df.dropna(axis=1, how='all')
                features_df = features_df.fillna(features_df.median())
                
                # 保存该物质的特征矩阵
                clean_name = clean_filename(substance_name)
                substance_file = f'features_{clean_name}.csv'
                features_df.to_csv(substance_file, index=False, encoding='utf-8-sig')
                
                # 记录物质信息
                substance_info[substance_name] = {
                    'features_file': substance_file,
                    'n_features': features_df.shape[1],
                    'n_samples': features_df.shape[0],
                    'n_descriptors': len(all_desc_keys)
                }
                
                all_features.append(features_df)
                
            except Exception as e:
                continue
        
        if not all_features:
            return "Error: 没有成功生成任何特征矩阵"
        
        # 合并所有特征
        combined_features = pd.concat(all_features, axis=1)
        
        # 添加实验次数列（如果存在）
        if len(df.columns) > 0:
            experiment_col = df.columns[0]
            if '实验' in experiment_col or '次数' in experiment_col or 'round' in experiment_col.lower():
                combined_features.insert(0, 'experiment_round', df[experiment_col])
        
        # 保存合并后的特征矩阵
        #combined_features.to_csv('features_matrix_combined.csv', index=False, encoding='utf-8-sig')
        
        # 保存目标变量
        targets_df = df[target_columns]
        targets_df.to_csv('target_variables.csv', index=False, encoding='utf-8-sig')
        
        # 生成处理报告
        report = f"""
描述符生成完成！

处理结果摘要:
- 成功处理 {len(substance_groups)} 个物质组
- 生成 {len(target_columns)} 个目标变量
- 总特征数量: {combined_features.shape[1]}
- 样本数量: {combined_features.shape[0]}

生成的文件:
- features_matrix_combined.csv: 合并后的完整特征矩阵
- target_variables.csv: 目标变量数据

各物质特征文件:
"""
        
        for substance_name, info in substance_info.items():
            report += f"- {info['features_file']}: {substance_name} 特征矩阵 ({info['n_features']} 个特征)\n"
        
        report += f"\n目标变量: {', '.join(target_columns)}"
        
        return report
        
    except Exception as e:
        return f"Error: 处理文件时发生错误: {str(e)}"

def main():
    """主函数，用于测试"""
    # 测试文件路径
    test_file = "example.csv"  # 请替换为实际的测试文件路径
    
    if os.path.exists(test_file):
        result = generate_multi_substance_descriptors(test_file)
        return result
    else:
        return f"测试文件 {test_file} 不存在，请提供正确的文件路径"

if __name__ == "__main__":
    print(main())
