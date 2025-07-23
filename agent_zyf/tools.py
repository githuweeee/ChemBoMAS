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