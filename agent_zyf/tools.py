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

"""Tools."""

import os
import shutil
import uuid
from google.adk.tools import ToolContext
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from concurrent.futures import ProcessPoolExecutor
import warnings
import multiprocessing
import re
warnings.filterwarnings('ignore')

def handle_file_upload(file_path: str, tool_context: ToolContext) -> str:
    """
    Handles both initial and subsequent user file uploads.

    If it's the first upload, it initializes the session.
    If it's a subsequent upload, it updates the data for the next iteration.

    Args:
        file_path: The local path to the user's uploaded CSV file OR the CSV content directly.
        tool_context: The context for the current tool execution.

    Returns:
        A string message confirming the action taken.
    """
    state = tool_context.state
    session_id = state.get("session_id")

    # Check if file_path is actually file content (contains CSV data)
    # This happens when ADK passes file content directly instead of file path
    if ',' in file_path and '\n' in file_path and not os.path.exists(file_path):
        # file_path is actually CSV content, write it to a temporary file
        temp_file_path = f"temp_uploaded_data_{uuid.uuid4().hex[:8]}.csv"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(file_path)
        file_path = temp_file_path
        print(f"Received CSV content directly, wrote to temporary file: {file_path}")
    elif not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."

    # Case 1: First upload, session needs to be initialized.
    if session_id is None:
        print("--------------------------------")
        print("--------------------------------")
        print(file_path)
        print(os.getcwd())
        print("--------------------------------")
        print("--------------------------------")
        session_id = str(uuid.uuid4())
        session_dir = os.path.join("sessions", session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        current_round = 0
        new_file_name = f"data_round_{current_round}.csv"
        destination_path = os.path.join(session_dir, new_file_name)
        
        shutil.copy(file_path, destination_path)

        state["session_id"] = session_id
        state["session_dir"] = session_dir
        state["current_data_path"] = destination_path
        state["experiment_round"] = current_round
        # Align with orchestrator state machine in prompts.py
        state["status"] = "Data_Verification"
        
        return (f"Session `{session_id}` started. "
                f"File received and saved to `{destination_path}`. "
                "Proceeding with Data_Verification.")

    # Case 2: Subsequent upload for an existing session.
    else:
        session_dir = state.get("session_dir")
        if not session_dir or not os.path.exists(session_dir):
            return "Error: Session directory not found. Please start a new session."

        current_round = state.get("experiment_round", 0) + 1
        
        new_file_name = f"data_round_{current_round}.csv"
        destination_path = os.path.join(session_dir, new_file_name)

        shutil.copy(file_path, destination_path)
        
        # Update state for the new round
        state["current_data_path"] = destination_path
        state["experiment_round"] = current_round
        # Reset status to re-run the pipeline (matches orchestrator prompt)
        state["status"] = "Data_Verification"

        # Clean up old intermediate files for the new run
        state.pop("verified_data_path", None)
        state.pop("descriptors_path", None)
        state.pop("recommendations_path", None)

        return (f"Received updated data for session `{session_id}` (Round {current_round}). "
                f"File saved to `{destination_path}`. "
                "Restarting workflow with data verification.") 


def verification(file_path: str, tool_context: ToolContext) -> str:
    print("#########################")
    print(file_path)
    print("####################################")
    state = tool_context.state
    session_id = state.get("session_id")

    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."
    try:
        df = pd.read_csv(file_path)
        total_rows = df.shape[0] + 1  # 包含表头
        total_cols = df.shape[1]
        experiment_count = total_rows - 1
        cols =  df.columns
        substance_cols = [c for c in cols if 'Substance' in str(c) and 'name' in str(c)]
        substance_count = len(substance_cols)
        target_cols = [c for c in cols if str(c).startswith('Target_')]
        target_count = len(target_cols)

        if (substance_count==0 or target_count==0):
            return ("Error: Please modify the name of the independent variable or dependent variable according to the format.")
        return (f"File information: \n"
                f"- Total number of experiments: {experiment_count}\n"
                f"- Total number of substances: {substance_count}\n"
                f"- Name of substances: {substance_cols}\n"
                f"- Total number of targets: {target_count}\n"
                f"- Name of targets: {target_cols}")
    except Exception as e:
        return f"File processing error：{e}"



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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    calc = Calculator(descriptors, ignore_3D=False)  # 不忽略3D描述符
    descriptor_values = calc(mol)
    descriptors_dict = {}
    for key, value in descriptor_values.items():
        if value is not None:
            float_value = float(value)
            if not np.isnan(float_value) and not np.isinf(float_value):
                descriptors_dict[str(key)] = float_value
    return descriptors_dict

def calculate_descriptors_parallel(smiles_list):
    n_jobs = max(1,multiprocessing.cpu_count()-1)
    valid_smiles = [s for s in smiles_list if isinstance(s, str) and s.strip()]
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
    filename = str(filename)
    filename = filename.strip()
    invalid_chars = r'[\\/*?:"<>|\ \t\n\r]'
    return re.sub(invalid_chars, '_', filename)

def generate_descriptor(file_path: str, tool_context: ToolContext) -> str:
    state = tool_context.state
    session_id = state.get("session_id")
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        columns = df.columns.tolist()
        substance_groups = []
        target_columns = []
        i = 1
        while i < len(columns):
            col_name = columns[i]
            if 'Substance' in col_name and 'name' in col_name:
                substance_name = col_name.split('_')[0]  
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
                        i += 4 
                        continue
            if col_name.startswith('Target_'):
                target_columns.append(col_name)
            i += 1
        if not substance_groups:
            return "Error: No valid substance column group found"
        if not target_columns:
            return "Error: Target variable column not found"

        # 首先收集所有物质的ratio数据
        all_ratio_data = {}
        for group in substance_groups:
            substance_name = group['name']
            ratio_data = pd.to_numeric(df[group['ratio_col']].astype(str).str.strip(), errors='coerce')
            all_ratio_data[substance_name] = ratio_data

        all_features = []
        substance_info = {}
        for group in substance_groups:
            substance_name = group['name']
            smiles_data = df[group['smile_col']].astype(str).str.strip()
            valid_mask = (smiles_data != 'nan') & (smiles_data != '') & (smiles_data.notna())
            valid_smiles = smiles_data[valid_mask].unique()
            if len(valid_smiles) == 0:
                continue
            try:
                descriptors_list, invalid_count = calculate_descriptors_parallel(valid_smiles)
                valid_smiles_list = list(valid_smiles)
                desc_map = {}
                for smile, desc in zip(valid_smiles_list, descriptors_list):
                    if desc is not None:
                        desc_map[smile] = desc
                if not desc_map:
                    continue
                features_df = pd.DataFrame()
                
                # 首先添加所有物质的ratio列到前面
                for ratio_substance_name, ratio_data in all_ratio_data.items():
                    features_df[f'{ratio_substance_name}_ratio'] = ratio_data
                
                # 添加描述符列
                all_desc_keys = set()
                for desc_dict in desc_map.values():
                    if desc_dict:
                        all_desc_keys.update(desc_dict.keys())
                
                for key in all_desc_keys:
                    features_df[f'{substance_name}_{key}'] = np.nan
                
                # 填充描述符值
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
            return "Error: No feature matrix was generated successfully"
        
        # 保存目标变量
        targets_df = df[target_columns]
        targets_df.to_csv('target_variables.csv', index=False, encoding='utf-8-sig')
        
        # 生成处理报告
        report = f"""
Descriptor generation completed!

Summary of processing results:
- Successfully processed {len(substance_groups)} substance groups
- Generate {len(target_columns)} target variables

"""
        
        for substance_name, info in substance_info.items():
            report += f"- {info['features_file']}: {substance_name} Feature Matrix ({info['n_features']} features)\n"
        report += f"\nTarget variable: {', '.join(target_columns)}"
        return report
    except Exception as e:
        return f"Error: An error occurred while processing the file: {str(e)}"


from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import glob

def perform_feature_selection(X, y, target_name, substance_name):
    """对单个目标变量进行特征选择"""
    try:
        # 使用均值填充缺失值
        X_valid = X.copy()
        y_valid = y.copy()
        
        if y_valid.isna().any():
            y_valid = y_valid.fillna(y_valid.mean())
            
        #if len(y_valid) < 10:
        #    print(f"警告: {target_name} 的样本数量不足 ({len(y_valid)} < 10)")
        #    return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # 自适应交叉验证折数，确保每个测试折至少2个样本
        n_samples = len(y_valid)
        n_splits = min(5, max(2, n_samples // 2))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 创建RFECV对象
        selector = RFECV(
            estimator=RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
            step=5,  # 每次移除5个特征
            min_features_to_select=20,  # 至少保留20个特征
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        
        selector.fit(X_scaled, y_valid)
        
        # 获取RFECV结果
        #print(f"最优特征数量: {selector.n_features_}")
        #print(f"特征选择的R²得分: {selector.score(X_scaled, y_valid):.4f}")
        
        selected_features = X.columns[selector.support_]
        X_selected = X[selected_features]
        
        forest = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        forest.fit(X_selected, y_valid)
        importances = forest.feature_importances_
        
        # 将特征和重要性配对
        feature_importances = [(feature, importance) 
                              for feature, importance in zip(selected_features, importances)]
        
        # 按重要性降序排序
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        # 创建结果字典
        result_dict = {
            'substance_name': substance_name,
            'target_name': target_name,
            'n_features': selector.n_features_,
            'r2_score': selector.score(X_scaled, y_valid),
            'selected_features': selected_features.tolist(),
            'feature_importances': feature_importances,
            'feature_ranking': selector.ranking_.tolist()
        }
        return result_dict
    
    except Exception as e:
        return f"特征选择过程中出错 ({substance_name} - {target_name}): {str(e)}"

def aggregate_and_save_per_substance(selection_results):
    """按物质聚合多个因变量实验的特征分数，输出每个物质的合并排序结果"""
    for substance_name, substance_results in selection_results.items():
        # 收集该物质在不同目标变量中的所有选中特征及其分数
        collected = []
        for target_name, result in substance_results.items():
            if result is None:
                continue
            for feature, importance in result['feature_importances']:
                collected.append((feature, importance))
        
        sum_scores = {}
        count_scores = {}
        for feature, importance in collected:
            sum_scores[feature] = sum_scores.get(feature, 0.0) + float(importance)
            count_scores[feature] = count_scores.get(feature, 0) + 1
        avg_scores = {f: (sum_scores[f] / count_scores[f]) for f in sum_scores}
        
        total_experiments = len([r for r in substance_results.values() if r is not None])
        total_selected_features = len(collected)
        features_to_select = total_selected_features // max(1, total_experiments)
        if features_to_select <= 0:
            features_to_select = len(avg_scores)
        
        # 排序并取前N个不重复特征
        sorted_items = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:features_to_select]
        
        # 保存到CSV
        clean_substance = clean_filename(substance_name)
        out_df = pd.DataFrame(top_items, columns=['feature', 'score'])
        out_df.to_csv(f'top_ranked_features_{clean_substance}.csv', encoding='utf-8-sig', index=False)

def feature_selection(features_matrix_path: str, target_variables_path: str, tool_context: ToolContext) -> str:
    feature_files = glob.glob('features_Substance*.csv')
    target_variables = pd.read_csv('target_variables.csv', encoding='utf-8-sig')
    selection_results = {}
    for feature_file in feature_files:
        substance_name = feature_file.replace('features_', '').replace('.csv', '')
        features_matrix = pd.read_csv(feature_file, encoding='utf-8-sig')
        substance_results = {}
        for col in target_variables.columns:
            clean_target_name = clean_filename(col)
            clean_substance_name = clean_filename(substance_name)
            y = target_variables[col]
            result = perform_feature_selection(features_matrix, y, col, substance_name)
            if isinstance(result, str):
                return result
            substance_results[col] = result
            importance_df = pd.DataFrame(result['feature_importances'], columns=['feature', 'score'])
            importance_df.to_csv(
                f'selected_features_with_scores_{clean_substance_name}_{clean_target_name}.csv',
                encoding='utf-8-sig', index=False
            )
        selection_results[substance_name] = substance_results
    aggregate_and_save_per_substance(selection_results)
    total_experiments = sum(len(substance_results) for substance_results in selection_results.values())
    return f"feature selection done. Successfully process {len(selection_results)} target variables. The results are saved to each csv files."
    