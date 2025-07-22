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

"""This file contains the tools for the Verification Agent."""

import logging
from pathlib import Path
import pandas as pd
from google.adk.tools import ToolContext

# 定义被认为是目标（因变量）的列名的关键字
# 在分析反应物时将忽略这些列 请注意COST加进去了
TARGET_KEYWORDS = ["target", "yield", "output", "ee", "cost"]

def verify_and_summarize_data(tool_context: ToolContext) -> str:
    """
    加载、验证并分析用户上传的实验数据CSV文件。

    此函数会执行以下操作:
    1. 从会话状态(Session State)中读取初始数据的路径。
    2. 使用pandas加载CSV文件。
    3. 识别代表"反应物"的列（即，非数字且非目标的列）。
    4. 为每个反应物列创建来源频次统计。
    5. 将验证后的数据保存到新文件中。
    6. 更新会话状态，包含新文件路径和更新后的状态。
    7. 返回一份摘要字符串，供用户确认。

    Args:
        tool_context (ToolContext): ADK工具上下文，用于访问会话状态。

    Returns:
        str: 一段人类可读的文本，总结了数据内容和反应物来源统计，
             并提示用户下一步操作。
    """
    logging.info("Verification Agent: 开始执行数据验证和摘要...")

    try:
        # 1. 从会话状态中获取初始数据路径
        initial_data_path_str = tool_context.state.get("initial_data_path")
        if not initial_data_path_str:
            return "错误：在会话状态中未找到 'initial_data_path'。无法继续。"
        
        initial_data_path = Path(initial_data_path_str)
        if not initial_data_path.exists():
            return f"错误：找不到指定的文件：{initial_data_path}"

        # 2. 加载CSV数据
        df = pd.read_csv(initial_data_path)
        logging.info(f"成功从 {initial_data_path} 加载了 {len(df)} 行数据。")

        # 3. 生成摘要和来源统计
        summary_parts = [
            f"数据加载成功！共发现 {len(df)} 行实验数据。",
            "我们分析了反应物的来源，统计如下:",
        ]

        reactant_columns = [
            col for col in df.columns 
            if df[col].dtype == 'object' and not any(keyword in col.lower() for keyword in TARGET_KEYWORDS)
        ]

        if not reactant_columns:
            summary_parts.append("- 未找到可分析的反应物来源列。")
        else:
            for col in reactant_columns:
                summary_parts.append(f"\n**{col}**:")
                value_counts = df[col].value_counts()
                for item, count in value_counts.items():
                    summary_parts.append(f"- {item}: {count} 次")

        # 4. 保存已验证的数据
        session_id = tool_context.state.get("session_id", "unknown_session")
        verified_data_dir = initial_data_path.parent
        verified_data_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
        
        verified_data_filename = f"verified_data_{session_id}.csv"
        verified_data_path = verified_data_dir / verified_data_filename
        
        df.to_csv(verified_data_path, index=False)
        logging.info(f"已验证的数据已保存至：{verified_data_path}")

        # 5. 更新会话状态
        tool_context.state["verified_data_path"] = str(verified_data_path)
        tool_context.state["status"] = "Verification_Complete"
        logging.info("会话状态已更新。")
        
        # 6. 准备最终输出给用户
        summary_parts.append("\n请确认以上信息是否正确。如果正确，我们将继续进行下一步的描述符生成。")
        final_summary = "\n".join(summary_parts)

        return final_summary

    except FileNotFoundError:
        logging.error(f"文件未找到: {initial_data_path_str}")
        return f"错误: 无法找到文件 '{initial_data_path_str}'。"
    except pd.errors.ParserError:
        logging.error(f"无法解析CSV文件: {initial_data_path_str}")
        return f"错误: 无法解析CSV文件 '{initial_data_path_str}'。请检查文件格式是否正确。"
    except Exception as e:
        logging.error(f"在验证过程中发生未知错误: {e}")
        return f"在处理您的文件时发生了一个意外错误: {e}" 