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

def handle_file_upload(file_path: str, tool_context: ToolContext) -> str:
    """
    Handles both initial and subsequent user file uploads.

    If it's the first upload, it initializes the session.
    If it's a subsequent upload, it updates the data for the next iteration.

    Args:
        file_path: The local path to the user's uploaded CSV file.
        tool_context: The context for the current tool execution.

    Returns:
        A string message confirming the action taken.
    """
    state = tool_context.state
    session_id = state.get("session_id")

    if not os.path.exists(file_path):
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
        state["status"] = "Awaiting_Verification"
        
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
        state["status"] = "Awaiting_Verification" # Reset status to re-run the pipeline

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
warnings.filterwarnings('ignore')

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
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
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(calculate_descriptors, valid_smiles))
    return results, invalid_num

def generate_descriptor(file_path: str, tool_context: ToolContext) -> str:
    state = tool_context.state
    session_id = state.get("session_id")
    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."
    
    df = pd.read_csv(file_path,encodings='utf-8')

    if df.shape[1] <= 5:
            return "Error: CSV file has less than 6 columns, unable to find the target column。"
    
    target_col_index = df.shape[0] - 1
    target_col_name = df.columns[target_col_index]
    #print(f"目标列被确定为: 第{target_col_index + 1}列 '{target_col_name}'")

    df[target_col_name] = pd.to_numeric(df[target_col_name], errors='coerce')
    original_rows = len(df)
    df.dropna(subset=[target_col_name], inplace=True)
    filtered_rows = len(df)
    delete_rows = original_rows - filtered_rows
    df.reset_index(drop=True, inplace=True)

    smiles_col_index = 2#第三列是固化剂SMILES
    smiles_col_name = df.columns[smiles_col_index]
    unique_curing_smiles = df[smiles_col_name].astype(str).str.strip().dropna().unique()
    try:
        curing_descriptors_list, invalid_num = calculate_descriptors_parallel(unique_curing_smiles)
    except:
        return "Failed to generate descriptor"
    curing_desc_map = {smile: desc for smile, desc in zip(unique_curing_smiles, curing_descriptors_list) if desc is not None}
    features_df = pd.DataFrame()
    loading_ratio_col_index = 3#第四列是固化剂添加比例
    loading_ratio_col_name = df.columns[loading_ratio_col_index]
    features_df['固化剂添加比例'] = pd.to_numeric(df[loading_ratio_col_name].astype(str).str.strip(), errors='coerce')
    all_descriptor_keys = set()
    for desc_dict in curing_desc_map.values():
        if desc_dict:
            all_descriptor_keys.update(desc_dict.keys())
    for key in all_descriptor_keys:
        features_df[f'Curing_{key}'] = np.nan
    for i, row in df.iterrows():
        curing_smile = str(row[smiles_col_name]).strip()
        if curing_smile in curing_desc_map and curing_desc_map[curing_smile]:
            for key, value in curing_desc_map[curing_smile].items():
                features_df.loc[i, f'Curing_{key}'] = value
    features_df = features_df.dropna(axis=1, how='all')
    features_df = features_df.fillna(features_df.median())

    output_file = 'features_matrix.csv'
    features_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    targets_df = df[[target_col_name]]
    target_variables_file = 'target_variables.csv'
    targets_df.to_csv('target_variables.csv', index=False, encoding='utf-8-sig')

    return f"Descriptor generated successfully. The target column has been determined as: {target_col_index+1} column '{target_col_name}'. Saved to {output_file} and target variables to {target_variables_file}"


from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re

def clean_filename(filename):
    """清理文件名，替换特殊字符为下划线，确保文件名一致性"""
    filename = str(filename)
    # 先清理可能存在的制表符、回车符和换行符
    filename = filename.strip()
    # 替换所有非法字符为下划线
    invalid_chars = r'[\\/*?:"<>|\ \t\n\r]'
    return re.sub(invalid_chars, '_', filename)

def perform_feature_selection(X, y, target_name):
    """对单个目标变量进行特征选择"""
    try:
        X_valid = X.copy()
        y_valid = y.copy()
        if y_valid.isna().any():
            return "target_name has missing values"
        if len(y_valid) < 10:
            return "the number of samples is too small"
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        # 创建RFECV对象
        selector = RFECV(
            estimator=RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
            step=5,  # 每次移除5个特征
            min_features_to_select=20,  # 至少保留20个特征
            cv=5,  # 5折交叉验证
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        selector.fit(X_scaled, y_valid)
        selected_features = X.columns[selector.support_]
        X_selected = X[selected_features]
        
        # 绘制交叉验证得分曲线
        plt.figure(figsize=(10, 6))
        plt.xlabel("特征数量")
        plt.ylabel("交叉验证得分 (R²)")
        plt.title(f"{target_name} - RFECV性能曲线")
        
        # 兼容不同版本的scikit-learn
        if hasattr(selector, 'cv_results_') and 'mean_test_score' in selector.cv_results_:
            # 新版本scikit-learn
            cv_scores = selector.cv_results_['mean_test_score']
            # 确保x和y长度相等
            x_range = np.arange(len(cv_scores)) * 5 + min(20, len(X.columns))
            plt.plot(x_range, cv_scores)
        elif hasattr(selector, 'grid_scores_'):
            # 旧版本scikit-learn
            cv_scores = selector.grid_scores_
            # 确保x和y长度相等
            x_range = np.arange(len(cv_scores)) * 5 + min(20, len(X.columns))
            plt.plot(x_range, cv_scores)
        else:
            cv_scores = None
        
        # 保存图表
        clean_target = clean_filename(target_name)
        plt.tight_layout()
        plt.savefig(f'rfecv_curve_{clean_target}.png', dpi=300)
        plt.close()
        
        # 获取特征重要性
        # 使用选中的特征重新训练随机森林来获取特征重要性
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
            'target_name': target_name,
            'n_features': selector.n_features_,
            'r2_score': selector.score(X_scaled, y_valid),
            'selected_features': selected_features.tolist(),
            'feature_importances': feature_importances,
            'feature_ranking': selector.ranking_.tolist()
        }
        # 添加交叉验证结果
        if cv_scores is not None:
            result_dict['cv_scores'] = cv_scores.tolist()
            result_dict['cv_features'] = x_range.tolist()
        return result_dict
    except Exception as e:
        return "feature selection error"

def feature_selectoin(features_matrix_path: str, target_variables_path: str, tool_context: ToolContext) -> str:
    features_matrix = pd.read_csv(features_matrix_path, encoding='utf-8-sig')
    target_variables = pd.read_csv(target_variables_path, encoding='utf-8-sig')
    selection_results = {}
    for col in target_variables.columns:
        clean_target_name = clean_filename(col)
        y = target_variables[col]
        result = perform_feature_selection(features_matrix, y, col)
        if isinstance(result,str):
            return result
        
        selection_results[col] = result
        selected_features_df = features_matrix[result['selected_features']]
        selected_features_df.to_csv(f'selected_features_{clean_target_name}.csv', encoding='utf-8-sig')
        target_df = pd.DataFrame(y)
        target_df.to_csv(f'target_{clean_target_name}.csv', encoding='utf-8-sig')
        importance_df = pd.DataFrame(result['feature_importances'], 
                                          columns=['feature', 'importance'])
        importance_df.to_csv(f'feature_importance_{clean_target_name}.csv', encoding='utf-8-sig')
        if 'cv_scores' in result and 'cv_features' in result:
            cv_results_df = pd.DataFrame({
                'n_features': result['cv_features'],
                'cv_score': result['cv_scores']
            })
            cv_results_df.to_csv(f'cv_results_{clean_target_name}.csv', encoding='utf-8-sig')    
    summary_data = {}
    for target_name, result in selection_results.items():
        summary_data[target_name] = {
            '选择的特征数量': result['n_features'],
            'R²得分': result['r2_score']
        }                        
    summary_df = pd.DataFrame(summary_data).T
    summary_df.to_csv('feature_selection_summary.csv', encoding='utf-8-sig')
    return f"feature selection done. Successfully process {len(selection_results)} target variables. The results are saved to each csv files."
    