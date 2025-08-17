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

        all_features = []
        substance_info = {}
        for group in substance_groups:
            substance_name = group['name']
            smiles_data = df[group['smile_col']].astype(str).str.strip()
            ratio_data = pd.to_numeric(df[group['ratio_col']].astype(str).str.strip(), errors='coerce')
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
                features_df[f'{substance_name}_ratio'] = ratio_data
                
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

Generated files:
- features_matrix_combined.csv: The complete feature matrix after merging
- target_variables.csv: Target variable data
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
import matplotlib.pyplot as plt

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

def feature_selection(features_matrix_path: str, target_variables_path: str, tool_context: ToolContext) -> str:
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
    