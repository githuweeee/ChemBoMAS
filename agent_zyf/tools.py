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

        # 第一列（除去表头）的种类数和名称
        first_col_name = df.columns[0]
        first_col_values = df[first_col_name].dropna().unique()
        category_count = len(first_col_values)
        category_names = ', '.join(map(str, first_col_values))

        return (f"文件统计信息：\n" 
                f"- 总行数（含表头）：{total_rows}\n"
                f"- 总列数：{total_cols}\n"
                f"- 实验次数：{experiment_count}\n"
                f"- 第一列“{first_col_name}”的种类数：{category_count}\n"
                f"- 种类名称：{category_names}")
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