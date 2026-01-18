# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recommender Agent Tools - BayBE Campaign构建、贝叶斯优化推荐和迭代管理

包含 Campaign 构建功能，实现完整的优化工作流：
1. Campaign构建（与推荐流程整合）
2. 实验推荐生成
3. 实验结果处理
4. 迭代优化管理
"""

import os
import re
import pandas as pd
import numpy as np
import json
import tempfile
from datetime import datetime
from typing import Dict, Optional
from google.adk.tools import ToolContext

# 导入高级收敛分析
try:
    from .adaptive_strategy import AdaptiveRecommendationStrategy
    ADVANCED_CONVERGENCE_AVAILABLE = True
except ImportError:
    ADVANCED_CONVERGENCE_AVAILABLE = False
    print("[WARN] Advanced convergence analysis not available. Using basic analysis.")

# BayBE导入
try:
    from baybe import Campaign
    from baybe.parameters import (
        CategoricalParameter, 
        NumericalContinuousParameter, 
        NumericalDiscreteParameter
    )
    from baybe.searchspace import SearchSpace
    from baybe.targets import NumericalTarget
    from baybe.objectives import DesirabilityObjective, ParetoObjective
    from baybe.constraints import (
        DiscreteSumConstraint,
        ContinuousLinearConstraint
    )
    from baybe.constraints.conditions import ThresholdCondition
    from baybe.recommenders import BotorchRecommender
    from baybe.utils.dataframe import add_fake_measurements
    BAYBE_AVAILABLE = True
except ImportError:
    print("Warning: BayBE not installed. Recommender Agent will not function.")
    BAYBE_AVAILABLE = False


# =============================================================================
# CSV 安全读取辅助函数（处理 UTF-8 BOM 并清洗列名）
# =============================================================================
def _is_valid_smiles_format(smiles: str) -> bool:
    """
    基本SMILES格式验证：检查括号是否匹配
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        bool: 格式是否有效
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    # 检查括号匹配
    stack = []
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in smiles:
        if char in bracket_pairs:
            stack.append(char)
        elif char in bracket_pairs.values():
            if not stack:
                return False  # 多余的右括号
            if bracket_pairs[stack.pop()] != char:
                return False  # 括号不匹配
    
    return len(stack) == 0  # 所有括号都应该匹配


def _build_baybe_recommender(optimization_config: dict):
    """根据 acquisition_function 构建推荐器（如未指定则返回 None）"""
    if not BAYBE_AVAILABLE:
        return None
    acquisition_function = optimization_config.get("acquisition_function", "default")
    if not acquisition_function or acquisition_function == "default":
        return None
    try:
        return BotorchRecommender(acquisition_function=acquisition_function)
    except Exception as exc:
        print(f"[WARN] 无法创建推荐器 (acquisition_function={acquisition_function}): {exc}")
        return None


def _read_csv_clean(path: str) -> pd.DataFrame:
    """
    读取 CSV 时处理 UTF-8 BOM，清理列名中的 BOM 与首尾空白。
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    # 丢弃常见的索引列（如 Pandas 导出的 "Unnamed: 0"）
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$")]
    return df


# =============================================================================
# Campaign 缓存机制（避免序列化 Campaign 对象到 state）
# =============================================================================
_campaign_cache: Dict[str, Campaign] = {}


def _get_campaign_from_cache(session_id: str) -> Optional[Campaign]:
    return _campaign_cache.get(session_id)


def _save_campaign_to_cache(session_id: str, campaign: Campaign) -> None:
    _campaign_cache[session_id] = campaign
    print(f"[DEBUG] Campaign cached for session: {session_id}")


def _clear_campaign_cache(session_id: str) -> None:
    if session_id in _campaign_cache:
        del _campaign_cache[session_id]
        print(f"[DEBUG] Campaign cache cleared for session: {session_id}")


def _format_recommendations_like_original(
    recommendations: pd.DataFrame, 
    state: dict, 
    campaign: Campaign
) -> pd.DataFrame:
    """
    将推荐格式化为原始数据表格格式
    
    目标：
    1. 保持原始表格的列顺序
    2. 保持原始表格的列名格式
    3. 包含所有原始列（即使不是参数）
    4. 添加化合物名称列（如果有）
    
    Returns:
        pd.DataFrame: 格式化后的推荐表格
    """
    original_format = state.get("original_data_format", {})
    original_column_order = original_format.get("column_order", [])

    if not original_column_order:
        # 回退1：使用已存在的 experiment_log.csv 列顺序
        unified_log = state.get("unified_experiment_log_path")
        if unified_log and os.path.exists(unified_log):
            try:
                existing_log = _read_csv_clean(unified_log)
                original_column_order = list(existing_log.columns)
                print(f"[WARN] No original_data_format; fallback to experiment_log columns ({len(original_column_order)})")
            except Exception as exc:
                print(f"[WARN] Failed to read experiment_log for column order: {exc}")

    if not original_column_order:
        # 回退2：使用标准化数据列顺序
        standardized_path = state.get("verification_results", {}).get("standardized_data_path")
        if standardized_path and os.path.exists(standardized_path):
            try:
                standardized_df = _read_csv_clean(standardized_path)
                original_column_order = list(standardized_df.columns)
                print(f"[WARN] No original_data_format; fallback to standardized_data columns ({len(original_column_order)})")
            except Exception as exc:
                print(f"[WARN] Failed to read standardized_data for column order: {exc}")

    if not original_column_order:
        # 如果没有原始格式信息，使用默认格式
        print("[WARN] No original data format found, using default format")
        return recommendations
    
    print(f"[DEBUG] Formatting recommendations to match original format ({len(original_column_order)} columns)")
    
    # 创建新 DataFrame，按照原始列顺序
    formatted_df = pd.DataFrame()
    
    # 获取 SMILES → 名称映射
    smiles_to_name_map = state.get("smiles_to_name_map", {})
    
    # 1. 添加参数列（按照原始顺序）
    for col in original_column_order:
        col_lower = col.lower()
        
        # 检查是否是参数列
        is_param_col = False
        param_name = None
        
        # 检查是否是分子列
        if 'smile' in col_lower or col.endswith("_molecule"):
            # 查找对应的参数名
            for param_name_candidate in campaign.searchspace.parameter_names:
                if param_name_candidate.endswith("_molecule"):
                    # 尝试匹配列名
                    if col.replace('_SMILE', '').replace('_SMILES', '').replace('_molecule', '') in param_name_candidate:
                        param_name = param_name_candidate
                        is_param_col = True
                        break
        
        # 检查是否是比例列
        if not is_param_col and 'ratio' in col_lower:
            for param_name_candidate in campaign.searchspace.parameter_names:
                if param_name_candidate == col or param_name_candidate.replace('_ratio', '') in col:
                    param_name = param_name_candidate
                    is_param_col = True
                    break
        
        # 检查是否是其他数值参数
        if not is_param_col:
            for param_name_candidate in campaign.searchspace.parameter_names:
                if param_name_candidate == col:
                    param_name = param_name_candidate
                    is_param_col = True
                    break
        
        if is_param_col and param_name in recommendations.columns:
            # 添加参数值
            formatted_df[col] = recommendations[param_name].values
            print(f"[DEBUG] _format_recommendations_like_original: 添加参数列 '{col}' <- '{param_name}' ({len(recommendations[param_name])} 个值)")
            
            # 调试：检查molecule列的值
            if param_name.endswith("_molecule"):
                non_null_count = recommendations[param_name].notna().sum()
                print(f"[DEBUG]   Molecule列 '{col}' 有 {non_null_count}/{len(recommendations)} 个非空值")
                if non_null_count > 0:
                    print(f"[DEBUG]     第一个值: {recommendations[param_name].iloc[0]}")
            
            # 如果是分子列，添加对应的名称列（无论原始表格是否有）
            if param_name.endswith("_molecule"):
                prefix = param_name.rsplit("_molecule", 1)[0]
                name_col_original = f"{prefix}_name"
                
                # 从推荐中获取名称或从映射中查找
                name_col_recommendation = f"{prefix}_name"
                if name_col_recommendation in recommendations.columns:
                    # 推荐中已有名称列，直接使用
                    formatted_df[name_col_original] = recommendations[name_col_recommendation].values
                    print(f"[DEBUG] Added name column {name_col_original} from recommendations")
                elif smiles_to_name_map:
                    # 从映射中查找名称
                    names = []
                    for smiles in recommendations[param_name]:
                        friendly_name = smiles_to_name_map.get(str(smiles).strip(), smiles)
                        names.append(friendly_name)
                    formatted_df[name_col_original] = names
                    print(f"[DEBUG] Added name column {name_col_original} from smiles_to_name_map")
                
                # 确保名称列在正确的位置（紧跟在分子列后面）
                # 如果原始表格中有名称列，保持原始顺序；如果没有，插入到分子列后面
                if name_col_original not in original_column_order:
                    # 找到分子列的位置，在其后插入名称列
                    col_index = original_column_order.index(col) if col in original_column_order else len(formatted_df.columns) - 1
                    # 重新排列列，将名称列放在分子列后面
                    cols = list(formatted_df.columns)
                    if name_col_original in cols:
                        cols.remove(name_col_original)
                        # 找到分子列的位置
                        try:
                            mol_idx = cols.index(col)
                            cols.insert(mol_idx + 1, name_col_original)
                            formatted_df = formatted_df[cols]
                        except ValueError:
                            pass
        
        elif col.startswith("Target_"):
            # 目标列：添加占位符
            if col in recommendations.columns:
                formatted_df[col] = recommendations[col].values
            else:
                formatted_df[col] = ["<请填写测量值>"] * len(recommendations)
        
        else:
            # 其他列（如元数据列）：添加默认值或空值
            if col in recommendations.columns:
                formatted_df[col] = recommendations[col].values
            else:
                # 根据列名推断默认值
                if "date" in col_lower:
                    formatted_df[col] = ["<YYYY-MM-DD>"] * len(recommendations)
                elif "id" in col_lower or "编号" in col:
                    formatted_df[col] = [f"EXP_{i+1:03d}" for i in range(len(recommendations))]
                elif "operator" in col_lower or "操作" in col:
                    formatted_df[col] = ["<操作员>"] * len(recommendations)
                elif "note" in col_lower or "备注" in col:
                    formatted_df[col] = [""] * len(recommendations)
                else:
                    formatted_df[col] = [""] * len(recommendations)
    
    # 2. 确保所有分子参数都有对应的名称列（即使原始表格没有）
    # 遍历推荐中的分子参数，确保名称列存在
    for param_name in campaign.searchspace.parameter_names:
        if param_name.endswith("_molecule") and param_name in recommendations.columns:
            prefix = param_name.rsplit("_molecule", 1)[0]
            name_col = f"{prefix}_name"
            
            # 如果名称列还没有被添加，现在添加它
            if name_col not in formatted_df.columns:
                name_col_recommendation = f"{prefix}_name"
                if name_col_recommendation in recommendations.columns:
                    # 从推荐中获取名称
                    formatted_df[name_col] = recommendations[name_col_recommendation].values
                    print(f"[DEBUG] Added missing name column {name_col} from recommendations")
                elif smiles_to_name_map:
                    # 从映射中查找名称
                    names = []
                    for smiles in recommendations[param_name]:
                        friendly_name = smiles_to_name_map.get(str(smiles).strip(), smiles)
                        names.append(friendly_name)
                    formatted_df[name_col] = names
                    print(f"[DEBUG] Added missing name column {name_col} from smiles_to_name_map")
                
                # 将名称列插入到分子列后面（如果可能）
                if name_col in formatted_df.columns:
                    cols = list(formatted_df.columns)
                    # 查找对应的分子列
                    mol_col = None
                    for orig_col in original_column_order:
                        if 'smile' in orig_col.lower() or orig_col.endswith("_molecule"):
                            if prefix in orig_col:
                                mol_col = orig_col
                                break
                    
                    if mol_col and mol_col in cols:
                        # 将名称列移到分子列后面
                        cols.remove(name_col)
                        try:
                            mol_idx = cols.index(mol_col)
                            cols.insert(mol_idx + 1, name_col)
                            formatted_df = formatted_df[cols]
                        except ValueError:
                            pass
    
    print(f"[DEBUG] Formatted recommendations: {len(formatted_df.columns)} columns, {len(formatted_df)} rows")
    print(f"[DEBUG] Columns with names: {[col for col in formatted_df.columns if col.endswith('_name')]}")
    return formatted_df


def _extract_smiles_name_map_from_state(state: dict) -> dict:
    """
    从 state 中的数据文件提取 SMILES → 名称映射
    
    这是一个备份方案，当 enhanced_verification 阶段的映射丢失时使用。
    """
    mapping = {}
    
    # 尝试从多个可能的数据文件路径读取
    possible_paths = [
        state.get("current_data_path"),
        state.get("verification_results", {}).get("standardized_data_path"),
    ]
    
    for path in possible_paths:
        if not path or not os.path.exists(path):
            continue
        
        try:
            df = _read_csv_clean(path)
            print(f"[DEBUG] Reading data from {path}, columns: {list(df.columns)}")
            
            # 遍历列查找 SMILES-名称配对
            for col in df.columns:
                col_upper = col.upper()
                is_smiles_col = 'SMILE' in col_upper or col.endswith("_molecule")
                if not is_smiles_col:
                    continue
                
                # 确定前缀
                if col.endswith("_molecule"):
                    prefix = col.rsplit("_molecule", 1)[0]
                elif '_SMILE' in col_upper:
                    idx = col_upper.find('_SMILE')
                    prefix = col[:idx] if idx > 0 else col.split('_')[0]
                else:
                    continue
                
                # 查找对应的名称列（支持大小写变体）
                name_col = None
                for df_col in df.columns:
                    df_col_upper = df_col.upper()
                    if df_col_upper == f"{prefix.upper()}_NAME" and df_col != col:
                        name_col = df_col
                        break
                
                if not name_col:
                    print(f"[DEBUG] No name column found for {col}, prefix={prefix}")
                    continue
                
                print(f"[DEBUG] Found SMILES-name pair: {col} -> {name_col}")
                
                # 提取映射
                for idx, row in df.iterrows():
                    smiles = row[col]
                    name = row[name_col]
                    
                    if pd.isna(smiles) or pd.isna(name):
                        continue
                    
                    smiles_str = str(smiles).strip()
                    name_str = str(name).strip()
                    
                    if smiles_str and name_str and smiles_str not in mapping:
                        mapping[smiles_str] = name_str
            
            if mapping:
                print(f"[DEBUG] Extracted {len(mapping)} SMILES-name mappings from {path}")
                break
                
        except Exception as e:
            print(f"[DEBUG] Error reading {path}: {e}")
            continue
    
    return mapping


def _add_name_columns(df: pd.DataFrame, smiles_map: dict) -> pd.DataFrame:
    """
    为每个 *_molecule 列添加对应的 *_name 列
    
    Args:
        df: 包含分子列的 DataFrame
        smiles_map: SMILES -> 名称 的映射字典
        
    Returns:
        添加了名称列的新 DataFrame
    """
    print(f"[DEBUG] _add_name_columns called with smiles_map size: {len(smiles_map) if smiles_map else 0}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    
    if not smiles_map:
        print("[DEBUG] No smiles_map provided, returning original DataFrame")
        return df
    
    result = df.copy()
    new_cols = {}
    insert_positions = []
    
    for col in df.columns:
        if col.endswith("_molecule"):
            prefix = col.rsplit("_molecule", 1)[0]
            name_col = f"{prefix}_name"
            
            print(f"[DEBUG] Processing molecule column: {col} -> {name_col}")
            
            # 映射 SMILES -> 名称，找不到则保留 SMILES
            def lookup_name(x):
                if pd.isna(x):
                    return x
                smiles_str = str(x).strip()
                name = smiles_map.get(smiles_str, None)
                if name is None:
                    # 尝试部分匹配（SMILES 可能有轻微差异）
                    for k, v in smiles_map.items():
                        if k in smiles_str or smiles_str in k:
                            name = v
                            break
                return name if name else smiles_str
            
            names = df[col].apply(lookup_name)
            new_cols[name_col] = names
            # 记录插入位置（在 molecule 列之后）
            insert_positions.append((df.columns.get_loc(col) + 1, name_col))
            
            # 调试：显示几个映射结果
            for i, (smiles, name) in enumerate(zip(df[col].head(2), names.head(2))):
                print(f"[DEBUG]   Row {i}: '{str(smiles)[:30]}...' -> '{name}'")
    
    # 按位置从后往前插入，避免索引偏移
    for pos, name_col in sorted(insert_positions, reverse=True):
        if name_col in new_cols:
            result.insert(pos, name_col, new_cols[name_col])
    
    print(f"[DEBUG] Result DataFrame columns: {list(result.columns)}")
    return result


# =============================================================================
# Campaign构建工具（与推荐流程整合）
# =============================================================================

def build_campaign_and_recommend(batch_size: str, tool_context: ToolContext) -> str:
    """
    一体化工具：构建BayBE Campaign并立即生成第一批实验推荐
    
    这个工具整合了 Campaign 构建与首轮推荐的流程。
    
    Args:
        batch_size: 推荐的实验数量（默认5）
        tool_context: ADK工具上下文
        
    Returns:
        str: Campaign构建结果和第一批实验推荐
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法构建Campaign。请运行: pip install baybe"
    
    try:
        # 验证batch_size
        try:
            batch_size = int(batch_size)
            if batch_size <= 0 or batch_size > 20:
                batch_size = 5
        except ValueError:
            batch_size = 5
        
        # 检查是否已有Campaign（从缓存）
        existing_campaign = _get_campaign_from_cache(session_id)
        if existing_campaign is not None:
            # 如果已有Campaign，直接生成推荐
            return _generate_recommendations_internal(existing_campaign, batch_size, state, session_id)
        else:
            # 确保构建前清理旧缓存
            _clear_campaign_cache(session_id)
        
        # 获取Enhanced Verification Agent的输出
        verification_results = state.get("verification_results", {})
        baybe_campaign_config = state.get("baybe_campaign_config", {})
        optimization_config = state.get("optimization_config", {})
        
        if not verification_results:
            return "❌ 未找到验证结果。请先运行Enhanced Verification Agent完成数据验证和优化目标配置。"
        
        # 步骤1：构建BayBE Campaign
        campaign_result = _build_baybe_campaign(
            verification_results, 
            baybe_campaign_config,
            optimization_config
        )
        
        if not campaign_result["success"]:
            return f"❌ Campaign构建失败: {campaign_result['error']}"
        
        campaign = campaign_result["campaign"]
        
        # 保存Campaign到缓存（避免序列化到state）
        _save_campaign_to_cache(session_id, campaign)

        # 保存可序列化的信息到state
        state["searchspace_info"] = campaign_result["searchspace_info"]
        state["ready_for_optimization"] = True
        state["construction_timestamp"] = datetime.now().isoformat()
        state["campaign_built"] = True
        
        # 步骤2：立即生成第一批推荐
        recommendation_result = _generate_recommendations_internal(campaign, batch_size, state, session_id)
        
        # 生成综合输出
        construction_summary = _generate_construction_summary(campaign_result, verification_results)
        
        return f"""
{construction_summary}

---

{recommendation_result}
"""
        
    except Exception as e:
        import traceback
        return f"❌ Campaign构建和推荐生成失败: {str(e)}\n{traceback.format_exc()}"


def _build_baybe_campaign(verification_results: dict, 
                         campaign_config: dict,
                         optimization_config: dict) -> dict:
    """
    构建完整的BayBE Campaign
    """
    try:
        # 调试：打印 optimization_config 中的边界信息
        print(f"[DEBUG] _build_baybe_campaign: optimization_config keys: {list(optimization_config.keys())}")
        if "custom_parameter_bounds" in optimization_config:
            bounds = optimization_config["custom_parameter_bounds"]
            print(f"[DEBUG] _build_baybe_campaign: custom_parameter_bounds = {bounds} (type: {type(bounds)})")
            if isinstance(bounds, dict):
                print(f"[DEBUG] _build_baybe_campaign: custom_parameter_bounds keys: {list(bounds.keys())}")
                for key, value in bounds.items():
                    print(f"[DEBUG] _build_baybe_campaign:   {key}: {value} (type: {type(value)})")
        if "parameter_boundaries" in optimization_config:
            bounds = optimization_config["parameter_boundaries"]
            print(f"[DEBUG] _build_baybe_campaign: parameter_boundaries = {bounds} (type: {type(bounds)})")
        
        # 1. 读取标准化数据
        standardized_data_path = verification_results.get("standardized_data_path")
        if not standardized_data_path or not os.path.exists(standardized_data_path):
            return {"success": False, "error": "标准化数据文件不存在"}
        
        df = _read_csv_clean(standardized_data_path)
        
        # 2. 创建BayBE参数
        parameters = _create_baybe_parameters(df, verification_results, optimization_config)
        
        if not parameters:
            return {"success": False, "error": "无法创建BayBE参数"}
        
        # 3. 创建约束（需要在创建搜索空间之前）
        constraints = _create_baybe_constraints(df, optimization_config)
        
        # 4. 创建搜索空间（传递约束）
        if constraints:
            print(f"[DEBUG] Creating SearchSpace with {len(constraints)} constraints")
            searchspace = SearchSpace.from_product(parameters=parameters, constraints=constraints)
        else:
            print("[DEBUG] Creating SearchSpace without constraints")
            searchspace = SearchSpace.from_product(parameters=parameters)
        
        # 5. 创建目标
        targets = _create_baybe_targets(df, optimization_config)
        
        if not targets:
            return {"success": False, "error": "无法创建目标函数"}
        
        # 6. 创建目标函数
        objective = _create_baybe_objective(targets, optimization_config)
        
        # 7. 创建Campaign（可选指定获取函数）
        recommender = _build_baybe_recommender(optimization_config)
        if recommender:
            campaign = Campaign(
                searchspace=searchspace,
                objective=objective,
                recommender=recommender
            )
        else:
            campaign = Campaign(
                searchspace=searchspace,
                objective=objective
            )
        
        # 8. 准备返回信息
        # 注意：SubspaceDiscrete 不支持 len()，需要用 exp_rep 获取大小
        searchspace_size = "continuous"
        if hasattr(searchspace, 'discrete') and searchspace.discrete is not None:
            try:
                searchspace_size = len(searchspace.discrete.exp_rep)
            except Exception:
                searchspace_size = "hybrid/unknown"

        searchspace_info = {
            "total_parameters": len(parameters),
            "molecule_parameters": len([p for p in parameters if isinstance(p, CategoricalParameter)]),
            "numerical_parameters": len([p for p in parameters if isinstance(p, (NumericalContinuousParameter, NumericalDiscreteParameter))]),
            "constraint_count": len(constraints),
            "searchspace_size": searchspace_size
        }
        
        return {
            "success": True,
            "campaign": campaign,
            "searchspace_info": searchspace_info,
            "parameters": parameters,
            "targets": targets,
            "constraints": constraints
        }
        
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


def _create_baybe_parameters(df: pd.DataFrame, verification_results: dict, optimization_config: dict) -> list:
    """
    创建BayBE参数定义
    
    参数边界来源优先级：
    1. optimization_config["parameter_boundaries"] - 用户自定义边界
    2. optimization_config["parameters"][col]["suggested_bounds"] - Enhanced Verification建议的边界
    3. 数据的 min/max 值（但需要检测退化区间）
    """
    parameters = []
    smiles_validation = verification_results.get("smiles_validation", {})
    
    # 从多个来源获取参数边界设置
    # 1. 用户自定义边界（最高优先级）
    # 注意：字段名可能是 "custom_parameter_bounds" 或 "parameter_boundaries"
    user_boundaries = optimization_config.get("custom_parameter_bounds", {}) or optimization_config.get("parameter_boundaries", {})
    if user_boundaries:
        print(f"[DEBUG] _create_baybe_parameters: 找到用户自定义边界，参数: {list(user_boundaries.keys())}")
        print(f"[DEBUG] _create_baybe_parameters: 边界内容: {user_boundaries}")
    else:
        print(f"[DEBUG] _create_baybe_parameters: 未找到用户自定义边界")
        print(f"[DEBUG] _create_baybe_parameters: optimization_config keys: {list(optimization_config.keys())}")
        if "custom_parameter_bounds" in optimization_config:
            print(f"[DEBUG] custom_parameter_bounds 值: {optimization_config.get('custom_parameter_bounds')}")
        if "parameter_boundaries" in optimization_config:
            print(f"[DEBUG] parameter_boundaries 值: {optimization_config.get('parameter_boundaries')}")
    # 2. Enhanced Verification 建议的边界
    parameter_suggestions = optimization_config.get("parameters", {})
    # 3. 是否接受建议的参数边界
    accept_suggested = optimization_config.get("accept_suggested_parameters", True)
    
    # 1. 分子参数 - 直接使用已验证的SMILES
    smiles_columns = [col for col in df.columns if 'SMILE' in col.upper()]
    for col in smiles_columns:
        substance_name = col.split('_')[0] if '_' in col else col
        
        # 获取有效的规范化SMILES
        valid_smiles = []
        canonical_mapping = smiles_validation.get("canonical_smiles_mapping", {})
        invalid_smiles_list = smiles_validation.get("invalid_smiles", [])
        
        # 收集无效SMILES的集合，用于过滤
        invalid_smiles_set = set()
        for invalid_item in invalid_smiles_list:
            if isinstance(invalid_item, dict) and "smiles" in invalid_item:
                invalid_smiles_set.add(str(invalid_item["smiles"]))
            elif isinstance(invalid_item, str):
                invalid_smiles_set.add(invalid_item)
        
        for smiles in df[col].dropna().unique():
            smiles_str = str(smiles).strip()
            
            # 跳过无效的SMILES
            if smiles_str in invalid_smiles_set:
                print(f"[WARN] 跳过无效SMILES: {smiles_str[:50]}...")
                continue
            
            # 优先使用规范化映射
            if smiles_str in canonical_mapping:
                canonical_smiles = canonical_mapping[smiles_str]
                # 验证规范化后的SMILES不为空且格式正确
                if canonical_smiles and canonical_smiles.strip():
                    # 基本格式检查：检查括号是否匹配
                    if _is_valid_smiles_format(canonical_smiles):
                        valid_smiles.append(canonical_smiles)
                    else:
                        print(f"[WARN] 规范化SMILES格式可能有问题: {canonical_smiles[:50]}...")
                else:
                    print(f"[WARN] 规范化SMILES为空: 原始值={smiles_str[:50]}...")
            else:
                # 如果没有规范化映射，尝试直接验证格式
                if _is_valid_smiles_format(smiles_str):
                    # 尝试使用BayBE的规范化函数
                    try:
                        from baybe.utils.chemistry import get_canonical_smiles
                        canonical = get_canonical_smiles(smiles_str)
                        if canonical:
                            valid_smiles.append(canonical)
                        else:
                            print(f"[WARN] 无法规范化SMILES: {smiles_str[:50]}...")
                    except:
                        print(f"[WARN] 跳过未验证的SMILES: {smiles_str[:50]}...")
                else:
                    print(f"[WARN] SMILES格式无效: {smiles_str[:50]}...")
        
        # 去重并排序（便于调试和显示）
        valid_smiles = sorted(list(set(valid_smiles)))
        
        if len(valid_smiles) >= 2:  # BayBE要求至少2个值
            param = CategoricalParameter(
                name=f"{substance_name}_molecule",
                values=valid_smiles,
                encoding="OHE"
            )
            parameters.append(param)
        elif len(valid_smiles) == 1:
            print(f"⚠️ {substance_name} 只有1个SMILES值，跳过分子参数创建")
        else:
            print(f"⚠️ {substance_name} 没有有效SMILES，跳过参数创建")
    
    # 2. 数值参数（比例、温度等）
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_columns = [col for col in df.columns if col.startswith('Target_')]
    
    for col in numeric_columns:
        if re.match(r"^Unnamed:\s*\d+$", str(col)):
            print(f"[DEBUG] 跳过索引列: {col}")
            continue
        if col not in target_columns:  # 排除目标变量
            # ===== 检查是否配置为离散参数 =====
            # ===== 仅创建连续参数（禁用离散参数）=====
            # 获取参数边界（按优先级）
            min_val, max_val = _get_parameter_bounds(
                col, df, user_boundaries, parameter_suggestions, accept_suggested
            )
            
            # 检测并处理退化区间（BayBE要求 min < max）
            if min_val >= max_val:
                print(f"⚠️ 检测到退化区间: {col} = [{min_val}, {max_val}]")
                min_val, max_val = _fix_degenerate_bounds(col, min_val, max_val, parameter_suggestions)
                print(f"   已修复为: [{min_val}, {max_val}]")
            
            # 根据列名选择参数类型
            if 'ratio' in col.lower():
                # 比例参数使用连续参数
                # 优先使用用户指定的边界，如果没有指定则使用获取到的边界
                # 注意：不再强制限制在[0,1]范围内，因为用户可能设置其他范围（如[0.5, 2.5]）
                final_min = min_val
                final_max = max_val
                
                # 检查是否用户明确指定了边界（如果是，应该完全信任用户设置）
                user_specified = False
                if col in user_boundaries:
                    user_specified = True
                    print(f"[DEBUG] {col} 使用用户自定义边界: [{final_min}, {final_max}]")
                elif accept_suggested and col in parameter_suggestions:
                    suggestion = parameter_suggestions[col]
                    if isinstance(suggestion, dict) and suggestion.get("suggested_bounds"):
                        user_specified = True
                        print(f"[DEBUG] {col} 使用建议边界: [{final_min}, {final_max}]")
                
                # 如果用户没有明确指定，且范围看起来不合理，才应用默认约束
                if not user_specified:
                    # 如果范围超出 [0, 1]，给出警告但不强制限制
                    if min_val < 0.0 or max_val > 1.0:
                        print(f"[WARN] {col} 的范围 [{min_val}, {max_val}] 超出典型的比例范围 [0, 1]")
                        print(f"      如果这是预期的，请忽略此警告")
                
                # 检查退化区间
                if final_min >= final_max:
                    print(f"⚠️ 比例参数 {col} 是退化区间: [{final_min}, {final_max}]")
                    # 尝试修复
                    center = (final_min + final_max) / 2 if final_min == final_max else 0.5
                    final_min = max(0.0, center - 0.1)
                    final_max = min(max(2.0, max_val), center + 0.1)  # 允许超出1.0
                    print(f"   已修复为: [{final_min}, {final_max}]")
                
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(final_min, final_max)
                )
                print(f"[DEBUG] 创建比例参数 {col}: bounds=({final_min}, {final_max})")
            elif 'temperature' in col.lower():
                # 温度参数
                # 检查是否用户明确指定了边界
                user_specified = col in user_boundaries
                if user_specified:
                    # 如果用户指定了边界，完全信任用户设置（不强制限制在[20, 200]）
                    final_min = min_val
                    final_max = max_val
                    print(f"[DEBUG] {col} 使用用户自定义温度边界: [{final_min}, {final_max}]")
                else:
                    # 如果没有用户指定，才应用默认的安全范围
                    final_min = max(20.0, min_val)
                    final_max = min(200.0, max_val)
                    if min_val < 20.0 or max_val > 200.0:
                        print(f"[WARN] {col} 的范围 [{min_val}, {max_val}] 超出典型温度范围 [20, 200]")
                        print(f"      已限制为: [{final_min}, {final_max}]")
                
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(final_min, final_max)
                )
                print(f"[DEBUG] 创建温度参数 {col}: bounds=({final_min}, {final_max})")
            else:
                # 其他数值参数
                # 检查是否用户明确指定了边界
                user_specified = col in user_boundaries
                if user_specified:
                    print(f"[DEBUG] {col} 使用用户自定义边界: [{min_val}, {max_val}]")
                else:
                    print(f"[DEBUG] {col} 使用数据/建议边界: [{min_val}, {max_val}]")
                
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(min_val, max_val)
                )
                print(f"[DEBUG] 创建参数 {col}: bounds=({min_val}, {max_val}), type=NumericalContinuousParameter")
            
            parameters.append(param)
            
            # ===== 最终验证：确保参数边界与用户设置一致 =====
            if col in user_boundaries:
                bounds = user_boundaries[col]
                if isinstance(bounds, dict):
                    # 支持 values 格式
                    if "values" in bounds:
                        values_list = bounds["values"]
                        if isinstance(values_list, (list, tuple)) and len(values_list) > 0:
                            expected_min = float(min(values_list))
                            expected_max = float(max(values_list))
                        else:
                            expected_min = expected_max = None
                    else:
                        # 标准边界格式
                        expected_min = bounds.get("min") or bounds.get("lower") or bounds.get("min_val") or bounds.get("minimum")
                        expected_max = bounds.get("max") or bounds.get("upper") or bounds.get("max_val") or bounds.get("maximum")
                elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    if len(bounds) == 2:
                        expected_min = float(bounds[0])
                        expected_max = float(bounds[1])
                    else:
                        # 多个值，可能是离散值列表
                        expected_min = float(min(bounds))
                        expected_max = float(max(bounds))
                else:
                    expected_min = expected_max = None
                
                if expected_min is not None and expected_max is not None:
                    # 获取实际创建的参数边界
                    actual_min, actual_max = _extract_bounds_from_param(param.bounds)
                    if (expected_min is None or expected_max is None or
                        actual_min is None or actual_max is None or
                        pd.isna(expected_min) or pd.isna(expected_max) or
                        pd.isna(actual_min) or pd.isna(actual_max)):
                        print(f"[WARN] {col} 参数边界包含 None/NaN，跳过一致性比较")
                        print(f"       用户设置: [{expected_min}, {expected_max}]")
                        print(f"       实际创建: [{actual_min}, {actual_max}]")
                    elif abs(actual_min - expected_min) > 1e-6 or abs(actual_max - expected_max) > 1e-6:
                        print(f"[ERROR] {col} 参数边界不一致！")
                        print(f"       用户设置: [{expected_min}, {expected_max}]")
                        print(f"       实际创建: [{actual_min}, {actual_max}]")
                    else:
                        print(f"[DEBUG] {col} 参数边界验证通过: [{actual_min}, {actual_max}]")
    
    return parameters


def _get_parameter_bounds(col: str, df: pd.DataFrame, 
                          user_boundaries: dict, 
                          parameter_suggestions: dict,
                          accept_suggested: bool) -> tuple:
    """
    按优先级获取参数边界
    
    优先级：
    1. 用户自定义边界 (user_boundaries)
    2. Enhanced Verification 建议的边界 (parameter_suggestions) - 如果accept_suggested=True
    3. 数据的 min/max 值
    """
    # 1. 用户自定义边界（支持多种格式）
    if col in user_boundaries:
        bounds = user_boundaries[col]
        
        # 支持多种格式：
        # - 字典格式: {"min": 0.5, "max": 2.5}
        # - 字典格式（离散值）: {"values": [0.5, 0.75, 1.0, ...]} - 从 values 提取 min/max
        # - 列表/元组格式: [0.5, 2.5] 或 (0.5, 2.5)
        if isinstance(bounds, dict):
            # 检查是否是离散值列表格式
            if "values" in bounds:
                values_list = bounds["values"]
                if isinstance(values_list, (list, tuple)) and len(values_list) > 0:
                    # 从离散值列表中提取 min/max
                    min_val = float(min(values_list))
                    max_val = float(max(values_list))
                    print(f"[DEBUG] _get_parameter_bounds: {col} 从 values 列表提取边界: [{min_val}, {max_val}]")
                else:
                    min_val = max_val = None
            else:
                # 标准边界格式
                min_val = bounds.get("min")
                max_val = bounds.get("max")
                # 如果字典中没有 min/max，尝试其他键名
                if min_val is None or max_val is None:
                    # 尝试 "lower"/"upper" 或其他可能的键名
                    min_val = bounds.get("lower") or bounds.get("min_val") or bounds.get("minimum")
                    max_val = bounds.get("upper") or bounds.get("max_val") or bounds.get("maximum")
            
            # 如果还是没有，使用数据范围作为默认值
            if min_val is None or pd.isna(min_val):
                min_val = float(pd.to_numeric(df[col], errors='coerce').min())
            if max_val is None or pd.isna(max_val):
                max_val = float(pd.to_numeric(df[col], errors='coerce').max())
        elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            min_val = float(bounds[0])
            max_val = float(bounds[1])
        else:
            # 如果格式不识别，使用数据范围
            print(f"[WARN] {col} 的用户边界格式不识别: {type(bounds)}, 值: {bounds}, 使用数据范围")
            min_val = float(df[col].min())
            max_val = float(df[col].max())
        
        # 如果仍是 NaN，给出兜底区间（避免后续 None/NaN 参与运算）
        if pd.isna(min_val) or pd.isna(max_val):
            print(f"[WARN] {col} 用户边界或数据范围为 NaN，使用兜底区间")
            if 'ratio' in col.lower():
                min_val, max_val = 0.0, 1.0
            elif 'temperature' in col.lower():
                min_val, max_val = 20.0, 200.0
            else:
                min_val, max_val = 0.0, 1.0

        print(f"[DEBUG] _get_parameter_bounds: {col} 用户自定义边界 = [{min_val}, {max_val}] (来源: {type(bounds)})")
        return min_val, max_val
    
    # 2. Enhanced Verification 建议的边界（如果用户接受建议）
    if accept_suggested and col in parameter_suggestions:
        suggestion = parameter_suggestions[col]
        if isinstance(suggestion, dict) and suggestion.get("suggested_bounds"):
            suggested = suggestion["suggested_bounds"]
            if isinstance(suggested, (list, tuple)) and len(suggested) == 2:
                return float(suggested[0]), float(suggested[1])
    
    # 3. 数据的 min/max 值
    min_val = float(pd.to_numeric(df[col], errors='coerce').min())
    max_val = float(pd.to_numeric(df[col], errors='coerce').max())
    if pd.isna(min_val) or pd.isna(max_val):
        print(f"[WARN] {col} 数据范围为 NaN，使用兜底区间")
        if 'ratio' in col.lower():
            return 0.0, 1.0
        if 'temperature' in col.lower():
            return 20.0, 200.0
        return 0.0, 1.0
    return min_val, max_val


def _fix_degenerate_bounds(col: str, min_val: float, max_val: float, 
                           parameter_suggestions: dict) -> tuple:
    """
    修复退化区间（min >= max）
    
    策略：
    1. 首先尝试使用 Enhanced Verification 建议的边界
    2. 如果没有建议，根据参数类型智能扩展
    """
    # 1. 尝试使用建议的边界
    if col in parameter_suggestions:
        suggestion = parameter_suggestions[col]
        if isinstance(suggestion, dict) and suggestion.get("suggested_bounds"):
            suggested = suggestion["suggested_bounds"]
            if isinstance(suggested, (list, tuple)) and len(suggested) == 2:
                if suggested[0] < suggested[1]:
                    return float(suggested[0]), float(suggested[1])
    
    # 2. 智能扩展
    # 兜底：避免 None/NaN 参与运算
    if min_val is None or max_val is None or pd.isna(min_val) or pd.isna(max_val):
        print(f"[WARN] {col} 退化区间包含 None/NaN，使用兜底区间")
        if 'ratio' in col.lower():
            return 0.0, 1.0
        if 'temperature' in col.lower():
            return 20.0, 200.0
        return 0.0, 1.0

    center = (min_val + max_val) / 2
    
    if 'ratio' in col.lower():
        # 比例参数：扩展到 [0, 0.2] 或 [center-0.1, center+0.1]
        new_min = max(0.0, center - 0.1)
        new_max = min(1.0, center + 0.1)
        # 如果仍然退化，使用更大的范围
        if new_min >= new_max:
            new_min = 0.0
            new_max = 0.2
    elif 'temperature' in col.lower():
        # 温度参数：扩展 ±10 度
        new_min = max(20.0, center - 10)
        new_max = min(200.0, center + 10)
    else:
        # 其他参数：扩展 ±10% 或 ±1.0
        range_expansion = max(abs(center) * 0.1, 1.0)
        new_min = center - range_expansion
        new_max = center + range_expansion
    
    return new_min, new_max


def _create_baybe_targets(df: pd.DataFrame, optimization_config: dict) -> list:
    """
    创建BayBE目标函数
    """
    targets = []
    # 确定使用的优化策略
    strategy = optimization_config.get("optimization_strategy", "pareto")
    use_pareto = (strategy != "desirability")

    # 从optimization_config获取目标配置
    targets_config = optimization_config.get("targets", [])
    
    if targets_config:
        # 使用用户配置的目标
        for target_cfg in targets_config:
            target_name = target_cfg.get("name")
            target_mode = target_cfg.get("mode", "MAX").upper()
            target_bounds = target_cfg.get("bounds")
            
            if target_name and target_name in df.columns:
                if use_pareto and target_mode in ["MIN", "MAX"]:
                    # Pareto 不允许 MIN/MAX 目标带 transforms/bounds
                    target = NumericalTarget(
                        name=target_name,
                        mode=target_mode
                    )
                elif target_mode == "MATCH":
                    # MATCH 必须有 bounds
                    if not target_bounds:
                        min_val = float(df[target_name].min())
                        max_val = float(df[target_name].max())
                        target_bounds = (min_val, max_val)
                    else:
                        target_bounds = tuple(target_bounds)
                    target = NumericalTarget(
                        name=target_name,
                        mode=target_mode,
                        bounds=target_bounds,
                        transformation="TRIANGULAR"
                    )
                else:
                    # Desirability / 非pareto 或 非 MIN/MAX
                    if not target_bounds:
                        min_val = float(df[target_name].min())
                        max_val = float(df[target_name].max())
                        if min_val == max_val:
                            min_val = min_val - 1.0 if min_val > 0 else 0.0
                            max_val = max_val + 1.0
                        target_bounds = (min_val, max_val)
                    else:
                        target_bounds = tuple(target_bounds)
                    target = NumericalTarget(
                        name=target_name,
                        mode=target_mode,
                        bounds=target_bounds,
                        transformation="LINEAR"
                    )
                targets.append(target)
    else:
        # 自动检测Target_开头的列
        target_columns = [col for col in df.columns if col.startswith('Target_')]
        
        for col in target_columns:
            if use_pareto:
                target = NumericalTarget(name=col, mode="MAX")
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                if min_val == max_val:
                    min_val = min_val - 1.0 if min_val > 0 else 0.0
                    max_val = max_val + 1.0
                target = NumericalTarget(
                    name=col,
                    mode="MAX",
                    bounds=(min_val, max_val),
                    transformation="LINEAR"
                )
            targets.append(target)
    
    return targets


def _create_baybe_objective(targets: list, optimization_config: dict):
    """
    创建BayBE目标函数
    
    多目标优化策略选择：
    - pareto (默认): 返回Pareto前沿上的多个非支配解，不需要预先指定权重，
                     用户可以在实验完成后根据实际情况选择最佳权衡方案
    - desirability: 需要预先指定权重，返回单一最优点
    
    BayBE文档推荐：当目标之间存在冲突且用户不确定权重时，优先使用Pareto方法
    """
    if len(targets) == 1:
        # 单目标优化 - 直接返回单一目标的Objective
        return DesirabilityObjective(targets=targets)
    else:
        # 多目标优化 - 默认使用Pareto（更灵活，不需要预先指定权重）
        strategy = optimization_config.get("optimization_strategy", "pareto")
        
        if strategy == "desirability":
            # 用户明确选择Desirability时使用
            # 需要从targets配置中获取权重
            weights = []
            targets_config = optimization_config.get("targets", [])
            
            for target in targets:
                weight = 1.0
                for tc in targets_config:
                    if tc.get("name") == target.name:
                        weight = tc.get("weight", 1.0)
                        break
                weights.append(weight)
            
            # 归一化权重
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            scalarizer = optimization_config.get("scalarizer", "GEOM_MEAN")
            
            return DesirabilityObjective(
                targets=targets,
                weights=weights,
                scalarizer=scalarizer
            )
        else:
            # 默认使用ParetoObjective - 不需要权重，更灵活
            return ParetoObjective(targets=targets)


def _create_baybe_constraints(df: pd.DataFrame, optimization_config: dict) -> list:
    """
    创建BayBE约束条件
    
    支持的约束类型：
    1. 自动检测的比例和约束（ratio 列之和 = 1.0）
    2. 用户定义的约束（从 optimization_config["constraints"] 解析）
    """
    constraints = []
    
    # 从optimization_config获取约束配置
    constraints_config = optimization_config.get("constraints", [])
    
    print(f"[DEBUG] Creating constraints, user-defined constraints: {len(constraints_config)}")
    
    # 1. 自动检测比例约束（可通过开关关闭）
    auto_ratio_sum_constraint = optimization_config.get("auto_ratio_sum_constraint", True)
    ratio_columns = [col for col in df.columns if 'ratio' in col.lower()]
    
    if auto_ratio_sum_constraint and len(ratio_columns) > 1:
        # 检查比例是否应该和为1
        ratio_sum = df[ratio_columns].sum(axis=1)
        if np.allclose(ratio_sum, 1.0, atol=0.1):
            try:
                constraint = ContinuousLinearConstraint(
                    parameters=ratio_columns,
                    coefficients=[1.0] * len(ratio_columns),
                    rhs=1.0,
                    operator="="
                )
                constraints.append(constraint)
                print(f"[DEBUG] Added auto-detected ratio sum constraint: {ratio_columns}")
            except Exception as e:
                print(f"[WARN] 无法创建比例约束: {e}")
    
    # 2. 解析用户定义的约束
    if constraints_config:
        for constraint_cfg in constraints_config:
            try:
                constraint_type = constraint_cfg.get("type", "").lower()
                
                if constraint_type == "sum_equals_one" or constraint_type == "sum_constraint":
                    # 比例和约束: x1 + x2 + ... = 1.0
                    params = constraint_cfg.get("parameters", [])
                    if params:
                        constraint = ContinuousLinearConstraint(
                            parameters=params,
                            coefficients=[1.0] * len(params),
                            rhs=1.0,
                            operator="="
                        )
                        constraints.append(constraint)
                        print(f"[DEBUG] Added user-defined sum constraint: {params} = 1.0")
                
                elif constraint_type == "sum_greater_than" or constraint_type == "sum_gt":
                    # 累加大于约束: x1 + x2 + ... > threshold
                    params = constraint_cfg.get("parameters", [])
                    threshold = constraint_cfg.get("threshold", constraint_cfg.get("rhs", 0.0))
                    if params:
                        constraint = ContinuousLinearConstraint(
                            parameters=params,
                            coefficients=[1.0] * len(params),
                            rhs=threshold,
                            operator=">="
                        )
                        constraints.append(constraint)
                        print(f"[DEBUG] Added sum_greater_than constraint: {params} >= {threshold}")
                
                elif constraint_type == "sum_less_than" or constraint_type == "sum_lt":
                    # 累加小于约束: x1 + x2 + ... < threshold
                    params = constraint_cfg.get("parameters", [])
                    threshold = constraint_cfg.get("threshold", constraint_cfg.get("rhs", 0.0))
                    if params:
                        constraint = ContinuousLinearConstraint(
                            parameters=params,
                            coefficients=[1.0] * len(params),
                            rhs=threshold,
                            operator="<="
                        )
                        constraints.append(constraint)
                        print(f"[DEBUG] Added sum_less_than constraint: {params} <= {threshold}")
                
                elif constraint_type == "linear_equality":
                    # 线性等式约束: a1*x1 + a2*x2 + ... = b
                    params = constraint_cfg.get("parameters", [])
                    coeffs = constraint_cfg.get("coefficients", [1.0] * len(params))
                    rhs = constraint_cfg.get("rhs", 0.0)
                    
                    if len(params) == len(coeffs):
                        constraint = ContinuousLinearConstraint(
                            parameters=params,
                            coefficients=coeffs,
                            rhs=rhs,
                            operator="="
                        )
                        constraints.append(constraint)
                        print(f"[DEBUG] Added user-defined linear equality: {params} = {rhs}")
                
                elif constraint_type == "linear_inequality":
                    # 线性不等式约束: a1*x1 + a2*x2 + ... <= b (或 >=, <, >)
                    params = constraint_cfg.get("parameters", [])
                    coeffs = constraint_cfg.get("coefficients", [1.0] * len(params))
                    rhs = constraint_cfg.get("rhs", 0.0)
                    operator = constraint_cfg.get("operator", "<=")
                    
                    # 标准化操作符（BayBE 支持 >= 和 <=）
                    if operator in [">", "gt"]:
                        operator = ">="
                    elif operator in ["<", "lt"]:
                        operator = "<="
                    
                    if len(params) == len(coeffs):
                        constraint = ContinuousLinearConstraint(
                            parameters=params,
                            coefficients=coeffs,
                            rhs=rhs,
                            operator=operator
                        )
                        constraints.append(constraint)
                        print(f"[DEBUG] Added user-defined linear inequality: {params} {operator} {rhs}")
                
                else:
                    print(f"[WARN] 未知的约束类型: {constraint_type}, 跳过")
                    print(f"[INFO] 支持的约束类型: sum_equals_one, sum_greater_than, sum_less_than, linear_equality, linear_inequality")
                    
            except Exception as e:
                print(f"[WARN] 解析用户约束失败: {constraint_cfg}, 错误: {e}")
    
    print(f"[DEBUG] Total constraints created: {len(constraints)}")
    return constraints


def _generate_construction_summary(campaign_result: dict, verification_results: dict) -> str:
    """
    生成搜索空间构建摘要
    """
    searchspace_info = campaign_result["searchspace_info"]
    
    summary = f"""
🚀 **BayBE Campaign 构建成功**

📊 **搜索空间配置**:
- 参数总数: {searchspace_info['total_parameters']}
  - 分子参数: {searchspace_info['molecule_parameters']} (BayBE自动处理描述符)
  - 数值参数: {searchspace_info['numerical_parameters']}
- 约束条件: {searchspace_info['constraint_count']}
- 搜索空间大小: {searchspace_info['searchspace_size']}

✅ **简化架构优势**:
- Enhanced Verification → Recommender Agent (无需中间步骤)
- BayBE自动处理所有分子描述符计算和缓存
- Campaign对象已创建完成，直接生成推荐
"""
    return summary


# =============================================================================
# 实验推荐工具
# =============================================================================

def generate_recommendations(batch_size: str, tool_context: ToolContext) -> str:
    """
    生成实验推荐 (用于后续轮次的推荐)
    
    Args:
        batch_size: 推荐的实验数量
        tool_context: ADK工具上下文
        
    Returns:
        str: 实验推荐结果
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法生成推荐。请运行: pip install 'baybe[chem]'"
    
    try:
        # 从缓存获取Campaign
        campaign = _get_campaign_from_cache(session_id)
        
        if not campaign:
            # 如果没有Campaign，尝试构建
            return build_campaign_and_recommend(batch_size, tool_context)
        
        # 验证batch_size
        try:
            batch_size = int(batch_size)
            if batch_size <= 0 or batch_size > 20:
                batch_size = 5
        except ValueError:
            batch_size = 5
        
        return _generate_recommendations_internal(campaign, batch_size, state, session_id)
        
    except Exception as e:
        return f"❌ 推荐生成失败: {str(e)}"


def _validate_recommendations_against_constraints(recommendations: pd.DataFrame, campaign: Campaign) -> list:
    """
    验证推荐值是否满足约束条件和参数边界
    
    Returns:
        list: 违反约束或边界的警告列表
    """
    warnings = []
    
    # ===== 新增：验证推荐值是否在参数边界内 =====
    for param_name in campaign.searchspace.parameter_names:
        if param_name not in recommendations.columns:
            continue
        
        # 获取参数对象
        param_obj = None
        if hasattr(campaign.searchspace, 'parameters'):
            for p in campaign.searchspace.parameters:
                if p.name == param_name:
                    param_obj = p
                    break
        
        # 检查连续参数的边界
        if param_obj and hasattr(param_obj, 'bounds'):
            min_val, max_val = _extract_bounds_from_param(param_obj.bounds)
            if min_val is not None and max_val is not None:
                # 使用小的容差（1e-6）来处理浮点数精度问题
                # BayBE 理论上应该严格遵守边界，但可能存在数值精度问题
                tolerance = 1e-6
                out_of_bounds = (recommendations[param_name] < min_val - tolerance) | (recommendations[param_name] > max_val + tolerance)
                if out_of_bounds.any():
                    for idx in recommendations[out_of_bounds].index:
                        val = recommendations.loc[idx, param_name]
                        # 计算超出量
                        if val < min_val:
                            excess = min_val - val
                        else:
                            excess = val - max_val
                        warnings.append(
                            f"实验 {idx + 1}: 参数 '{param_name}' 值 {val} 超出边界 [{min_val}, {max_val}] (超出 {excess:.2e})"
                        )
                    print(f"[ERROR] 检测到 {out_of_bounds.sum()} 个推荐值超出参数 '{param_name}' 的边界 [{min_val}, {max_val}]")
                    print(f"       这可能是 BayBE 的 bug 或数值精度问题，也可能是参数创建时边界设置错误")
        
        # 检查离散参数的值
        if param_obj and hasattr(param_obj, 'values'):
            valid_values = set(param_obj.values)
            invalid_values = recommendations[param_name].apply(lambda x: x not in valid_values)
            if invalid_values.any():
                for idx in recommendations[invalid_values].index:
                    val = recommendations.loc[idx, param_name]
                    warnings.append(
                        f"实验 {idx + 1}: 参数 '{param_name}' 值 {val} 不在允许的离散值列表中 {list(valid_values)}"
                    )
                print(f"[ERROR] 检测到 {invalid_values.sum()} 个推荐值不在参数 '{param_name}' 的离散值列表中")
    
    if not hasattr(campaign.searchspace, 'discrete') or campaign.searchspace.discrete is None:
        return warnings
    
    # 检查离散空间的约束
    discrete = campaign.searchspace.discrete
    if hasattr(discrete, 'constraints') and discrete.constraints is not None:
        # 这里可以添加离散约束验证逻辑
        pass
    
    # 检查连续空间的约束
    if hasattr(campaign.searchspace, 'continuous') and campaign.searchspace.continuous is not None:
        continuous = campaign.searchspace.continuous
        if hasattr(continuous, 'linear_equality_constraints') and continuous.linear_equality_constraints is not None:
            # 验证线性等式约束
            for idx, row in recommendations.iterrows():
                for constraint in continuous.linear_equality_constraints:
                    # 计算约束值
                    constraint_value = sum(
                        row.get(param, 0) * coeff 
                        for param, coeff in zip(constraint.parameters, constraint.coefficients)
                    )
                    # 检查是否满足约束（允许小的浮点误差）
                    if abs(constraint_value - constraint.rhs) > 1e-6:
                        warnings.append(
                            f"实验 {idx + 1} 违反约束: "
                            f"{'+'.join([f'{c}*{p}' for c, p in zip(constraint.coefficients, constraint.parameters)])} = {constraint.rhs}, "
                            f"实际值: {constraint_value:.6f}"
                        )
    
    return warnings


def _generate_recommendations_internal(campaign: Campaign, batch_size: int, state: dict, session_id: str) -> str:
    """
    内部推荐生成函数
    """
    # 生成推荐（可选：使用用户偏好的获取函数）
    optimization_config = state.get("optimization_config", {})
    recommender = _build_baybe_recommender(optimization_config)
    if recommender:
        current_recommender = getattr(campaign, "recommender", None)
        current_acq = getattr(current_recommender, "acquisition_function", None) if current_recommender else None
        if current_recommender is None or current_acq != optimization_config.get("acquisition_function"):
            try:
                campaign.recommender = recommender
                print(f"[DEBUG] 使用获取函数 {optimization_config.get('acquisition_function')} 配置推荐器")
            except Exception as e:
                print(f"[WARN] 无法设置推荐器，回退默认推荐: {e}")
    recommendations = campaign.recommend(batch_size=batch_size)
    
    # 验证推荐值是否满足约束
    constraint_warnings = _validate_recommendations_against_constraints(recommendations, campaign)
    if constraint_warnings:
        print(f"[WARN] 检测到 {len(constraint_warnings)} 个约束违反:")
        for warning in constraint_warnings[:3]:
            print(f"  {warning}")
    
    # 获取 SMILES → 名称映射
    smiles_to_name_map = state.get("smiles_to_name_map", {})
    print(f"[DEBUG] _generate_recommendations_internal: smiles_to_name_map size = {len(smiles_to_name_map)}")
    
    # 如果映射为空，尝试从原始数据文件重新提取
    if not smiles_to_name_map:
        print("[DEBUG] smiles_to_name_map is empty, trying to extract from data file...")
        smiles_to_name_map = _extract_smiles_name_map_from_state(state)
        if smiles_to_name_map:
            state["smiles_to_name_map"] = smiles_to_name_map
            print(f"[DEBUG] Extracted smiles_to_name_map: {len(smiles_to_name_map)} entries")
    
    if smiles_to_name_map:
        print(f"[DEBUG] First few entries: {list(smiles_to_name_map.items())[:3]}")
    
    # 添加化合物名称列（如果有映射）
    recommendations_with_names = _add_name_columns(recommendations, smiles_to_name_map)
    
    # ===== 按照原始数据格式生成推荐表格 =====
    recommendations_formatted = _format_recommendations_like_original(
        recommendations_with_names, 
        state, 
        campaign
    )
    
    # 确定保存路径（统一实验记录表）
    session_dir = state.get("session_dir", ".")
    unified_experiment_log = state.get("unified_experiment_log_path")
    
    if not unified_experiment_log:
        unified_experiment_log = os.path.join(session_dir, "experiment_log.csv")
        state["unified_experiment_log_path"] = unified_experiment_log
    
    # 如果统一记录表不存在，创建它
    if not os.path.exists(unified_experiment_log):
        # 使用格式化后的推荐作为初始结构
        recommendations_formatted["optimization_round"] = state.get("optimization_round", 0) + 1
        recommendations_formatted["experiment_status"] = "pending"  # 待实验
        
        # 调试：检查molecule列是否正确填写
        molecule_cols = [col for col in recommendations_formatted.columns if col.endswith('_molecule') or 'SMILE' in col.upper()]
        if molecule_cols:
            print(f"[DEBUG] 创建experiment_log时，molecule列检查: {molecule_cols}")
            for mol_col in molecule_cols:
                non_null_count = recommendations_formatted[mol_col].notna().sum()
                print(f"[DEBUG]   {mol_col}: {non_null_count}/{len(recommendations_formatted)} 行有值")
                if non_null_count > 0:
                    print(f"[DEBUG]     示例值: {recommendations_formatted[mol_col].iloc[0]}")
        
        recommendations_formatted.to_csv(unified_experiment_log, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Created unified experiment log with recommendations")
    else:
        # 追加到现有表格
        existing_log = _read_csv_clean(unified_experiment_log)
        current_round = state.get("optimization_round", 0) + 1
        
        # 添加轮次和状态标记
        recommendations_formatted["optimization_round"] = current_round
        recommendations_formatted["experiment_status"] = "pending"

        # 对齐列顺序，确保化合物名称等列写入正确单元格而不是追加到表尾
        # ===== 关键修复：处理列名不匹配问题 =====
        # 如果existing_log使用_molecule列名，而recommendations_formatted使用_SMILE列名，需要映射
        column_mapping = {}
        for existing_col in existing_log.columns:
            if existing_col in recommendations_formatted.columns:
                # 列名已匹配，无需映射
                continue
            
            # 检查是否是分子列的不匹配
            if existing_col.endswith('_molecule'):
                base_name = existing_col.replace('_molecule', '')
                # 尝试在recommendations_formatted中查找对应的_SMILE列
                possible_cols = [
                    f"{base_name}_SMILE",
                    f"{base_name}_SMILES",
                    f"{base_name}_smile",
                    f"{base_name}_smiles",
                ]
                for possible_col in possible_cols:
                    if possible_col in recommendations_formatted.columns:
                        column_mapping[possible_col] = existing_col
                        print(f"[DEBUG] 列名映射（追加到experiment_log）: {possible_col} -> {existing_col}")
                        break
                    # 大小写不敏感匹配
                    matching_cols = [col for col in recommendations_formatted.columns if col.upper() == possible_col.upper()]
                    if matching_cols:
                        column_mapping[matching_cols[0]] = existing_col
                        print(f"[DEBUG] 列名映射（追加到experiment_log，大小写不敏感）: {matching_cols[0]} -> {existing_col}")
                        break
            
            # ===== 关键修复：处理name列的不匹配 =====
            # 检查是否是name列的不匹配（如 SubstanceA_NAME vs SubstanceA_name）
            if existing_col.endswith('_name') or existing_col.endswith('_NAME'):
                base_name = existing_col.replace('_name', '').replace('_NAME', '')
                # 尝试在recommendations_formatted中查找对应的name列（大小写不敏感）
                possible_name_cols = [
                    f"{base_name}_name",
                    f"{base_name}_NAME",
                    f"{base_name}_Name",
                ]
                for possible_name_col in possible_name_cols:
                    if possible_name_col in recommendations_formatted.columns:
                        if possible_name_col != existing_col:
                            column_mapping[possible_name_col] = existing_col
                            print(f"[DEBUG] Name列映射（追加到experiment_log）: {possible_name_col} -> {existing_col}")
                        break
                    # 大小写不敏感匹配
                    matching_cols = [col for col in recommendations_formatted.columns 
                                    if col.upper() == possible_name_col.upper() and col != existing_col]
                    if matching_cols:
                        column_mapping[matching_cols[0]] = existing_col
                        print(f"[DEBUG] Name列映射（追加到experiment_log，大小写不敏感）: {matching_cols[0]} -> {existing_col}")
                        break
        
        # 应用列名映射
        if column_mapping:
            print(f"[DEBUG] 应用列名映射前，recommendations_formatted的列: {list(recommendations_formatted.columns)}")
            recommendations_formatted = recommendations_formatted.rename(columns=column_mapping)
            print(f"[DEBUG] 已应用 {len(column_mapping)} 个列名映射以匹配existing_log")
            print(f"[DEBUG] 应用列名映射后，recommendations_formatted的列: {list(recommendations_formatted.columns)}")
            
            # 调试：检查映射后的name列是否有值
            for old_col, new_col in column_mapping.items():
                if old_col.endswith('_name') or old_col.endswith('_NAME'):
                    if new_col in recommendations_formatted.columns:
                        non_null_count = recommendations_formatted[new_col].notna().sum()
                        print(f"[DEBUG] 映射后的name列 '{new_col}' (原 '{old_col}'): {non_null_count}/{len(recommendations_formatted)} 行有值")
                        if non_null_count > 0:
                            print(f"[DEBUG]   示例值: {recommendations_formatted[new_col].iloc[0]}")
        
        # 1. 确保推荐表包含所有现有列
        for col in existing_log.columns:
            if col not in recommendations_formatted.columns:
                # 对于name列，尝试从recommendations_formatted中查找对应的大小写变体
                if col.endswith('_name') or col.endswith('_NAME'):
                    base_name = col.replace('_name', '').replace('_NAME', '')
                    # 尝试查找各种大小写变体的name列
                    possible_name_cols = [f"{base_name}_name", f"{base_name}_NAME", f"{base_name}_Name"]
                    found = False
                    for possible_name_col in possible_name_cols:
                        if possible_name_col in recommendations_formatted.columns:
                            # 检查源列是否有值
                            source_non_null = recommendations_formatted[possible_name_col].notna().sum()
                            if source_non_null > 0:
                                recommendations_formatted[col] = recommendations_formatted[possible_name_col].values
                                print(f"[DEBUG] 从 {possible_name_col} 复制值到缺失的name列 '{col}' ({source_non_null} 个非空值)")
                                found = True
                                break
                            else:
                                print(f"[DEBUG] 找到 {possible_name_col} 但所有值都是NaN，继续查找")
                    if found:
                        continue
                    else:
                        print(f"[DEBUG] 未找到name列 '{col}' 的对应列，将填充NaN")
                
                # 对于不存在的列，填充为空值（由用户后续填写或保持空白）
                recommendations_formatted[col] = np.nan
                print(f"[DEBUG] 添加缺失列 '{col}' 到recommendations_formatted（填充NaN）")

        # 2. 丢弃在现有日志中不存在的多余列，避免在表尾新增意外列
        recommendations_aligned = recommendations_formatted[existing_log.columns]
        
        # 调试：检查molecule列和name列是否正确填写
        molecule_cols = [col for col in existing_log.columns if col.endswith('_molecule') or 'SMILE' in col.upper()]
        name_cols = [col for col in existing_log.columns if col.endswith('_name') or col.endswith('_NAME')]
        
        if molecule_cols:
            print(f"[DEBUG] Molecule列检查: {molecule_cols}")
            for mol_col in molecule_cols:
                if mol_col in recommendations_aligned.columns:
                    non_null_count = recommendations_aligned[mol_col].notna().sum()
                    print(f"[DEBUG]   {mol_col}: {non_null_count}/{len(recommendations_aligned)} 行有值")
                    if non_null_count > 0:
                        print(f"[DEBUG]     示例值: {recommendations_aligned[mol_col].iloc[0]}")
                else:
                    print(f"[DEBUG]   {mol_col}: 列不存在于recommendations_aligned中")
        
        if name_cols:
            print(f"[DEBUG] Name列检查: {name_cols}")
            for name_col in name_cols:
                if name_col in recommendations_aligned.columns:
                    # 检查非空值（包括非空字符串）
                    non_null_mask = recommendations_aligned[name_col].notna() & (recommendations_aligned[name_col].astype(str).str.strip() != '')
                    non_null_count = non_null_mask.sum()
                    print(f"[DEBUG]   {name_col}: {non_null_count}/{len(recommendations_aligned)} 行有值")
                    if non_null_count > 0:
                        first_value = recommendations_aligned[name_col][non_null_mask].iloc[0] if non_null_mask.any() else recommendations_aligned[name_col].iloc[0]
                        print(f"[DEBUG]     示例值: '{first_value}' (类型: {type(first_value).__name__})")
                    else:
                        print(f"[DEBUG]   {name_col}: 列存在但所有值都是NaN或空字符串")
                        # 尝试从recommendations_formatted中查找对应的name列并复制值
                        base_name = name_col.replace('_name', '').replace('_NAME', '')
                        possible_name_cols = [f"{base_name}_name", f"{base_name}_NAME", f"{base_name}_Name"]
                        for possible_name_col in possible_name_cols:
                            if possible_name_col in recommendations_formatted.columns:
                                source_non_null = (recommendations_formatted[possible_name_col].notna() & 
                                                  (recommendations_formatted[possible_name_col].astype(str).str.strip() != '')).sum()
                                if source_non_null > 0:
                                    recommendations_aligned[name_col] = recommendations_formatted[possible_name_col].values
                                    print(f"[DEBUG]     已从 {possible_name_col} 复制值到 {name_col} ({source_non_null} 个非空值)")
                                    break
                else:
                    print(f"[DEBUG]   {name_col}: 列不存在于recommendations_aligned中")
                    # 尝试从recommendations_formatted中查找并添加
                    base_name = name_col.replace('_name', '').replace('_NAME', '')
                    possible_name_cols = [f"{base_name}_name", f"{base_name}_NAME", f"{base_name}_Name"]
                    for possible_name_col in possible_name_cols:
                        if possible_name_col in recommendations_formatted.columns:
                            source_non_null = (recommendations_formatted[possible_name_col].notna() & 
                                              (recommendations_formatted[possible_name_col].astype(str).str.strip() != '')).sum()
                            if source_non_null > 0:
                                recommendations_aligned[name_col] = recommendations_formatted[possible_name_col].values
                                print(f"[DEBUG]     已从 {possible_name_col} 添加列 {name_col} ({source_non_null} 个非空值)")
                                break

        # 3. 追加新行（列顺序与原表完全一致）
        combined_log = pd.concat([existing_log, recommendations_aligned], ignore_index=True)
        combined_log.to_csv(unified_experiment_log, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Appended {len(recommendations_aligned)} recommendations to unified log (columns aligned)")
    
    # 更新状态（保存带名称的版本）
    state["latest_recommendations"] = recommendations_formatted.to_dict('records')
    state["recommendation_file"] = unified_experiment_log
    state["recommendations_generated"] = True
    state["awaiting_experimental_results"] = True
    state["last_recommendation_time"] = datetime.now().isoformat()
    
    # 生成用户友好的推荐显示（传入带名称的版本和映射）
    return _format_recommendations_output(
        recommendations_formatted, 
        campaign, 
        unified_experiment_log, 
        smiles_to_name_map,
        constraint_warnings
    )


def _format_recommendations_output(
    recommendations: pd.DataFrame, 
    campaign: Campaign, 
    file_path: str, 
    smiles_map: dict = None,
    constraint_warnings: list = None
) -> str:
    """
    格式化推荐输出
    
    Args:
        recommendations: 推荐结果 DataFrame（可能已包含 *_name 列）
        campaign: BayBE Campaign 对象
        file_path: 保存的文件路径
        smiles_map: SMILES → 名称映射（用于在输出中显示友好名称）
        constraint_warnings: 约束违反警告列表
    """
    if smiles_map is None:
        smiles_map = {}
    if constraint_warnings is None:
        constraint_warnings = []
    
    # 检查搜索空间中的约束数量
    constraint_count = 0
    if hasattr(campaign.searchspace, 'continuous') and campaign.searchspace.continuous is not None:
        continuous = campaign.searchspace.continuous
        if hasattr(continuous, 'linear_equality_constraints') and continuous.linear_equality_constraints is not None:
            constraint_count += len(continuous.linear_equality_constraints)
        if hasattr(continuous, 'linear_inequality_constraints') and continuous.linear_inequality_constraints is not None:
            constraint_count += len(continuous.linear_inequality_constraints)
    
    output = f"""
🎯 **实验推荐已生成**

📊 **推荐概览**:
- 推荐实验数: {len(recommendations)}
- 参数数量: {len(campaign.searchspace.parameter_names)}
- 目标数量: {len(campaign.objective.targets)}
- 约束条件: {constraint_count} 个

🧪 **推荐的实验条件**:
"""
    
    # 显示推荐的实验条件
    for idx, row in recommendations.iterrows():
        output += f"\n**实验 {idx + 1}**:\n"
        for param_name in campaign.searchspace.parameter_names:
            if param_name in row:
                value = row[param_name]
                
                # 如果是分子参数，尝试显示名称
                if param_name.endswith("_molecule"):
                    prefix = param_name.rsplit("_molecule", 1)[0]
                    name_col = f"{prefix}_name"
                    
                    # 优先从 DataFrame 中获取名称列
                    if name_col in row:
                        friendly_name = row[name_col]
                    else:
                        # 否则从映射中查找
                        friendly_name = smiles_map.get(str(value).strip(), None)
                    
                    if friendly_name and friendly_name != value:
                        output += f"   - {param_name}: {friendly_name} ({value})\n"
                    else:
                        output += f"   - {param_name}: {value}\n"
                elif isinstance(value, float):
                    output += f"   - {param_name}: {value:.4f}\n"
                else:
                    output += f"   - {param_name}: {value}\n"
    
    # 添加约束验证信息
    if constraint_warnings:
        output += f"""

⚠️ **约束验证警告**:
检测到 {len(constraint_warnings)} 个推荐值违反约束条件:
"""
        for warning in constraint_warnings[:5]:  # 只显示前5个
            output += f"  • {warning}\n"
        
        if len(constraint_warnings) > 5:
            output += f"  ... 还有 {len(constraint_warnings) - 5} 个违反未显示\n"
        
        output += """
💡 **可能的原因**:
- 约束定义不正确
- 搜索空间与约束不兼容
- BayBE 推荐算法在数值精度范围内处理约束

🔧 **建议**: 检查约束定义和参数边界设置
"""
    elif constraint_count > 0:
        output += f"""

✅ **约束验证**: 所有推荐值均满足 {constraint_count} 个约束条件
"""
    
    output += f"""

📄 **文件保存**: {file_path}
   (CSV 文件中已包含化合物名称列，便于阅读)

🔄 **下一步**:
1. 使用 `generate_result_template` 工具生成结果上传模板
2. 按照推荐条件进行实验
3. 测量目标变量: {', '.join([t.name for t in campaign.objective.targets])}
4. 使用 `upload_experimental_results` 工具上传结果

💡 **实验提示**:
- 请确保实验条件严格按照推荐值执行
- 记录任何异常情况或偏差
- 测量所有目标变量以获得最佳优化效果
    """
    
    return output


# =============================================================================
# 实验结果处理工具
# =============================================================================

def upload_experimental_results(results_file_path: str, tool_context: ToolContext) -> str:
    """
    处理用户上传的实验结果并更新Campaign
    
    Args:
        results_file_path: 实验结果CSV文件路径或内容
        tool_context: ADK工具上下文
        
    Returns:
        str: 结果处理状态
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法处理实验结果。"
    
    try:
        # 从缓存获取当前Campaign
        campaign = _get_campaign_from_cache(session_id)
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。请先使用 build_campaign_and_recommend 工具。"
        
        # 处理文件路径 vs 文件内容
        if ',' in results_file_path and '\n' in results_file_path and not os.path.exists(results_file_path):
            # 是CSV内容，写入临时文件
            session_dir = state.get("session_dir", ".")
            temp_file_path = os.path.join(
                session_dir,
                f"temp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(results_file_path)
            results_file_path = temp_file_path
            print(f"接收到CSV内容，已写入临时文件: {results_file_path}")
        
        # ===== 统一表格模式：优先从统一实验记录表读取 =====
        unified_experiment_log = state.get("unified_experiment_log_path")
        use_unified_log = False
        
        # 检查是否是统一实验记录表
        if unified_experiment_log and os.path.exists(unified_experiment_log):
            if results_file_path == unified_experiment_log or os.path.abspath(results_file_path) == os.path.abspath(unified_experiment_log):
                use_unified_log = True
                print("[DEBUG] Using unified experiment log for upload")
        
        if use_unified_log:
            # 从统一表格中读取
            experiment_log_df = _read_csv_clean(unified_experiment_log)
            
            # 提取待完成的实验（pending 或目标列已填写但状态仍为 pending）
            expected_targets = [target.name for target in campaign.objective.targets]
            
            # 查找已填写目标值的待实验行
            completed_experiments = experiment_log_df[
                (experiment_log_df.get("experiment_status", pd.Series(["completed"] * len(experiment_log_df))) == "pending") &
                (experiment_log_df[expected_targets[0]].notna() if expected_targets else pd.Series([True] * len(experiment_log_df)))
            ]
            
            if len(completed_experiments) == 0:
                return """
⚠️ **没有可上传的实验结果**

统一实验记录表中没有已填写目标值的待实验记录。

💡 **建议**:
1. 在统一实验记录表中填写目标列的测量值
2. 确保实验状态为 "pending" 或已填写目标值
3. 然后重新上传统一实验记录表
"""
            
            # 使用已完成的实验作为结果
            results_df = completed_experiments.copy()
            print(f"[DEBUG] Extracted {len(results_df)} completed experiments from unified log")
        else:
            # 传统模式：从指定文件读取
            # 验证文件存在
            if not os.path.exists(results_file_path):
                return f"❌ 实验结果文件不存在: {results_file_path}"
            
            # 读取实验结果
            results_df = _read_csv_clean(results_file_path)
        
        # 简化的格式验证
        expected_targets = [target.name for target in campaign.objective.targets]
        missing_targets = [col for col in expected_targets if col not in results_df.columns]
        
        if missing_targets:
            return f"❌ 实验结果缺少目标列: {', '.join(missing_targets)}"
        
        # 检查必需的参数列（在预处理之前，提供更友好的错误信息）
        required_param_names = list(campaign.searchspace.parameter_names)
        missing_params = []
        for param_name in required_param_names:
            if param_name not in results_df.columns:
                # 如果是分子列，检查是否有对应的 _SMILE 列
                if param_name.endswith('_molecule'):
                    base_name = param_name.replace('_molecule', '')
                    possible_smile_cols = [
                        f"{base_name}_SMILE",
                        f"{base_name}_SMILEs",
                        f"{base_name}_smile",
                        f"{base_name}_smiles",
                    ]
                    # 检查是否存在这些列（大小写不敏感）
                    found = False
                    for possible_col in possible_smile_cols:
                        if possible_col in results_df.columns:
                            found = True
                            break
                        # 大小写不敏感匹配
                        matching_cols = [col for col in results_df.columns if col.upper() == possible_col.upper()]
                        if matching_cols:
                            found = True
                            break
                    if not found:
                        missing_params.append(f"{param_name} (或对应的 {base_name}_SMILE 列)")
                else:
                    missing_params.append(param_name)
        
        if missing_params:
            return f"""
❌ **实验结果缺少必需的参数列**

缺少的列: {', '.join(missing_params)}

📋 **期望的参数列**:
{chr(10).join(f"- {name}" for name in required_param_names)}

📊 **您提供的列**:
{chr(10).join(f"- {col}" for col in results_df.columns)}

💡 **建议**:
1. 如果缺少分子列（如 SubstanceA_molecule），请确保 CSV 文件中有对应的 _SMILE 列（如 SubstanceA_SMILE）
2. 使用 generate_result_template 工具生成正确的模板
3. 检查列名拼写是否正确（注意大小写）
"""
        
        # 数据预处理（会自动处理 _SMILE -> _molecule 的映射）
        # 传递state以便保存SMILES匹配错误信息
        processed_results = _preprocess_experimental_results(results_df, campaign, state)
        
        # 验证参数值是否与推荐值不同（警告但不阻止）
        # 注意：在预处理之后比较，确保使用规范化后的值
        parameter_warnings = _validate_parameter_values(processed_results, state, campaign)
        
        if processed_results.empty:
            # 提供更详细的诊断信息
            diagnosis = []
            
            # 检查原始数据
            diagnosis.append(f"📊 **原始数据**: {len(results_df)} 行, {len(results_df.columns)} 列")
            diagnosis.append(f"📋 **原始列名**: {', '.join(results_df.columns.tolist())}")
            
            # 检查列映射
            required_param_names = list(campaign.searchspace.parameter_names)
            expected_targets = [target.name for target in campaign.objective.targets]
            diagnosis.append(f"📋 **需要的参数列**: {', '.join(required_param_names)}")
            diagnosis.append(f"📋 **需要的目标列**: {', '.join(expected_targets)}")
            
            # 检查哪些列缺失
            missing_params = [col for col in required_param_names if col not in results_df.columns]
            missing_targets = [col for col in expected_targets if col not in results_df.columns]
            
            if missing_params:
                diagnosis.append(f"⚠️ **缺失的参数列**: {', '.join(missing_params)}")
            if missing_targets:
                diagnosis.append(f"⚠️ **缺失的目标列**: {', '.join(missing_targets)}")
            
            # 检查数据样本（前3行）
            if len(results_df) > 0:
                diagnosis.append(f"\n📝 **数据样本（前3行）**:")
                for idx in range(min(3, len(results_df))):
                    row_data = results_df.iloc[idx].to_dict()
                    diagnosis.append(f"  行 {idx}: {row_data}")
            
            return f"""
❌ **处理后的实验结果为空**

{chr(10).join(diagnosis)}

🔍 **可能的原因**:
1. **数据类型问题**: 数值列包含文本、占位符（如 "<请填写测量值>"）或特殊字符
2. **列名不匹配**: 缺少必需的参数列或目标列（检查上面的"缺失的列"）
3. **所有值都是 NaN**: 数据类型转换失败，导致所有值变成 NaN 后被移除
4. **参数值超出范围**: 所有参数值都超出搜索空间范围（但通常不会导致数据被清空）

💡 **建议**:
1. **检查目标列**: 确保 `{', '.join(expected_targets)}` 列中的所有值都是有效数字
   - 不能有文本、占位符或空值
   - 示例: `85.5` ✅ 而不是 `"<请填写测量值>"` ❌ 或 `""` ❌
   
2. **检查参数列**: 确保所有参数列的值都是数字且在有效范围内
   - 参数列: `{', '.join(required_param_names)}`
   
3. **检查列名**: 如果缺少分子列（如 `SubstanceA_molecule`），确保有对应的 `_SMILE` 列（如 `SubstanceA_SMILE`）

4. **使用模板**: 使用 `generate_result_template` 工具生成正确的模板，然后填写数据

📋 **调试信息**: 请查看服务器日志中的 `[DEBUG]` 和 `[WARN]` 消息，了解具体哪些值转换失败或超出范围
"""
        
        # 更新BayBE Campaign（添加详细的错误处理）
        try:
            print(f"[DEBUG] Adding {len(processed_results)} measurements to Campaign")
            print(f"[DEBUG] Measurement columns: {list(processed_results.columns)}")
            print(f"[DEBUG] First row sample: {processed_results.iloc[0].to_dict()}")
            
            campaign.add_measurements(processed_results)
            print("[DEBUG] Measurements added successfully")
            
            # ===== 如果使用统一表格，更新实验状态 =====
            if use_unified_log:
                # 更新统一表格中的实验状态
                experiment_log_df = _read_csv_clean(unified_experiment_log)
                
                # 标记已上传的实验为 "completed"
                # 通过匹配参数值来识别对应的行
                for idx, processed_row in processed_results.iterrows():
                    # 在统一表格中查找匹配的行
                    for log_idx, log_row in experiment_log_df.iterrows():
                        # 检查参数是否匹配（允许小的数值误差）
                        match = True
                        for param_name in campaign.searchspace.parameter_names:
                            if param_name in processed_row and param_name in log_row:
                                processed_val = processed_row[param_name]
                                log_val = log_row[param_name]
                                
                                # 对于数值，允许小的误差
                                if isinstance(processed_val, (int, float)) and isinstance(log_val, (int, float)):
                                    if abs(processed_val - log_val) > 1e-6:
                                        match = False
                                        break
                                elif str(processed_val).strip() != str(log_val).strip():
                                    match = False
                                    break
                        
                        if match:
                            experiment_log_df.at[log_idx, "experiment_status"] = "completed"
                            print(f"[DEBUG] Marked experiment at row {log_idx} as completed")
                
                # 保存更新后的统一表格
                experiment_log_df.to_csv(unified_experiment_log, index=False, encoding="utf-8-sig")
                print(f"[DEBUG] Updated unified experiment log: {len(experiment_log_df)} total experiments")
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"[ERROR] Failed to add measurements: {error_msg}")
            print(f"[ERROR] Traceback: {traceback_str}")
            
            # 检查是否是SMILES匹配错误
            smiles_errors = state.get('smiles_matching_errors', {})
            if smiles_errors and ("invalid" in error_msg.lower() or "value" in error_msg.lower() or "molecule" in error_msg.lower()):
                # 构建详细的SMILES错误信息
                error_parts = [
                    "❌ **SMILES值匹配失败**",
                    "",
                    "🔍 **问题分析**:",
                    "上传的SMILES值无法匹配到Campaign中定义的有效SMILES值。",
                    "系统已尝试自动规范化SMILES，但规范化后的值仍不在Campaign的有效值列表中。",
                    "这通常是因为：",
                    "1. RDKit版本或配置差异导致规范化结果不同",
                    "2. SMILES字符串本身存在格式问题",
                    "3. Campaign创建时使用的SMILES与上传的SMILES来源不同",
                    ""
                ]
                
                for param_name, error_info in smiles_errors.items():
                    valid_smiles = error_info.get('valid_smiles', [])
                    errors = error_info.get('errors', [])
                    
                    error_parts.append(f"📋 **{param_name} 参数**:")
                    error_parts.append(f"- Campaign期望的有效SMILES数量: {len(valid_smiles)}")
                    error_parts.append(f"- 无法匹配的SMILES数量: {len(errors)}")
                    error_parts.append("")
                    error_parts.append("✅ **Campaign期望的精确SMILES值（请使用这些值）**:")
                    for i, smiles in enumerate(valid_smiles, 1):
                        error_parts.append(f"   {i}. `{smiles}`")
                    error_parts.append("")
                    
                    if errors:
                        error_parts.append("❌ **无法匹配的SMILES（前5个）**:")
                        for err in errors[:5]:
                            error_parts.append(f"   - 行 {err['row']}: `{err['original']}`")
                            if err['canonical']:
                                error_parts.append(f"     规范化后: `{err['canonical']}` (仍不在有效值列表中)")
                            error_parts.append(f"     错误: {err['error']}")
                        error_parts.append("")
                
                error_parts.extend([
                    "💡 **解决方案**:",
                    "1. 请使用上面列出的**Campaign期望的精确SMILES值**替换您CSV文件中的SMILES",
                    "2. 这些SMILES值是从Campaign创建时使用的规范化SMILES，必须完全匹配",
                    "3. 您可以直接复制上面的SMILES值到您的CSV文件中",
                    "4. 确保SMILES字符串完全一致（包括大小写、括号等）",
                    "",
                    "📝 **操作步骤**:",
                    "1. 打开您的实验结果CSV文件",
                    "2. 找到对应的SMILES列（如 SubstanceA_SMILE）",
                    "3. 将值替换为上面列出的Campaign期望的精确SMILES值",
                    "4. 保存文件后重新上传",
                    "",
                    "⚠️ **注意**: SMILES字符串必须完全匹配，即使是微小的差异（如空格、大小写）也会导致匹配失败。"
                ])
                
                return "\n".join(error_parts)
            
            # 尝试提供更友好的错误信息
            if "invalid" in error_msg.lower() or "value" in error_msg.lower():
                # 检查第一行数据
                first_row = processed_results.iloc[0] if len(processed_results) > 0 else None
                if first_row is not None:
                    param_info = []
                    for param_name in campaign.searchspace.parameter_names:
                        if param_name in first_row:
                            value = first_row[param_name]
                            # 获取参数边界（使用与预处理相同的逻辑）
                            param_bounds = None
                            if hasattr(campaign.searchspace, 'parameters'):
                                for param in campaign.searchspace.parameters:
                                    if param.name == param_name and hasattr(param, 'bounds'):
                                        param_bounds = param.bounds
                                        break
                            
                            if param_bounds is None and hasattr(campaign.searchspace, 'continuous'):
                                continuous = campaign.searchspace.continuous
                                if continuous is not None:
                                    if hasattr(continuous, 'parameters'):
                                        for param in continuous.parameters:
                                            if param.name == param_name and hasattr(param, 'bounds'):
                                                param_bounds = param.bounds
                                                break
                            
                            # 获取参数类型和有效值信息
                            param_type_info = "类型=未知"
                            valid_values_info = ""
                            
                            # 尝试从 searchspace 获取参数对象
                            param_obj = None
                            if hasattr(campaign.searchspace, 'parameters'):
                                for p in campaign.searchspace.parameters:
                                    if p.name == param_name:
                                        param_obj = p
                                        break
                            
                            if param_obj:
                                if isinstance(param_obj, CategoricalParameter):
                                    param_type_info = "类型=分类参数(CategoricalParameter)"
                                    # 对于SMILES参数，显示所有有效值（完整列表）
                                    valid_values = list(param_obj.values)
                                    if len(valid_values) <= 10:
                                        valid_values_info = f", 有效值={valid_values}"
                                    else:
                                        valid_values_info = f", 有效值数量={len(valid_values)}, 前10个={valid_values[:10]}"
                                    # 如果是分子参数，提供更详细的SMILES列表
                                    if param_name.endswith('_molecule'):
                                        valid_values_info += f"\n   完整SMILES列表（共{len(valid_values)}个）:"
                                        for i, smiles in enumerate(valid_values, 1):
                                            valid_values_info += f"\n     {i}. {smiles}"
                                elif isinstance(param_obj, NumericalDiscreteParameter):
                                    param_type_info = "类型=离散数值参数(NumericalDiscreteParameter)"
                                    valid_values_info = f", 有效离散值={sorted(list(param_obj.values))}"
                                elif isinstance(param_obj, NumericalContinuousParameter):
                                    param_type_info = "类型=连续数值参数(NumericalContinuousParameter)"
                                    if hasattr(param_obj, 'bounds'):
                                        min_val, max_val = _extract_bounds_from_param(param_obj.bounds)
                                        if min_val is not None and max_val is not None:
                                            valid_values_info = f", 范围=[{min_val}, {max_val}]"
                            
                            # 如果没有从参数对象获取到边界，尝试从 bounds 获取
                            if not valid_values_info and param_bounds:
                                min_val, max_val = _extract_bounds_from_param(param_bounds)
                                if min_val is not None and max_val is not None:
                                    valid_values_info = f", 范围=[{min_val}, {max_val}]"
                            
                            param_info.append(
                                f"  - {param_name}: 值={value}, 类型={type(value).__name__}, {param_type_info}{valid_values_info}"
                            )
                    
                    return f"""
❌ **实验结果处理失败**

错误信息: {error_msg}

📋 **第一行数据详情**:
{chr(10).join(param_info)}

💡 **可能的原因**:
1. 参数值类型不匹配（如字符串而非数字）
2. 参数值超出搜索空间范围
3. CSV 文件格式问题（编码、分隔符等）

🔧 **建议**:
1. 检查 CSV 文件，确保所有数值列都是纯数字
2. 确认参数值在推荐范围内
3. 尝试重新生成结果模板并填写
"""
            
            return f"""
❌ **实验结果处理失败**

错误信息: {error_msg}

💡 **建议**:
1. 检查数据格式是否正确
2. 确认所有必需的参数和目标列都存在
3. 查看终端日志获取更详细的错误信息
"""
        
        # 更新状态
        current_round = state.get("optimization_round", 0) + 1
        state["optimization_round"] = current_round
        state["campaign_updated"] = True
        state["awaiting_experimental_results"] = False
        state["ready_for_next_recommendations"] = True
        state["last_result_upload_time"] = datetime.now().isoformat()
        
        # 构建返回消息
        result_message = f"""
✅ **实验结果已成功添加到Campaign**

📊 **本轮实验摘要**:
- 轮次: {current_round}
- 新增实验: {len(processed_results)}
- Campaign总实验数: {len(campaign.measurements)}
"""
        
        # 如果有参数修改警告，添加警告信息
        if parameter_warnings:
            # 检查是否是因为数据行数不匹配导致的误报
            num_uploaded = len(processed_results)
            num_recommended = len(state.get("latest_recommendations", []))
            
            if num_uploaded > num_recommended:
                # 用户上传的数据行数多于推荐，可能是添加了新实验
                result_message += f"""
ℹ️ **数据行数说明**:
- 推荐实验数: {num_recommended}
- 实际上传数: {num_uploaded}
- 检测到 {len(parameter_warnings)} 个参数值差异

💡 **说明**: 如果您只是添加了测量值（目标列），参数值应该与推荐值一致。
如果看到此提示，可能是因为：
1. 您添加了新的实验（不是推荐的实验）
2. 参数值在预处理过程中被规范化（如SMILES），这是正常的
3. 数据格式或精度差异导致的微小差异

✅ **数据已接受**: 系统会使用您提供的所有数据更新模型
"""
            else:
                # 正常情况下的参数修改警告
                result_message += f"""
⚠️ **参数修改检测**:
检测到 {len(parameter_warnings)} 个参数值被修改（与推荐值不同）:

"""
                for warning in parameter_warnings[:5]:  # 只显示前5个
                    result_message += f"  • {warning}\n"
                
                if len(parameter_warnings) > 5:
                    result_message += f"  ... 还有 {len(parameter_warnings) - 5} 个修改未显示\n"
                
                result_message += """
💡 **影响说明**:
- 修改参数值会偏离 BayBE 的最优探索路径
- 可能影响代理模型的更新和后续推荐质量
- 如果修改后的值超出搜索空间边界，可能导致优化效率下降
- 建议仅在特殊情况下（如实验条件限制）才修改参数值

✅ **数据已接受**: 系统仍会使用您提供的参数值更新模型
"""
        
        result_message += f"""
🔄 **状态更新**:
- Campaign已更新 ✅
- 可以生成下一轮推荐 ✅

🚀 **下一步**: 
- 使用 `generate_recommendations` 获取新的实验推荐
- 或使用 `check_convergence` 检查优化是否收敛
        """
        
        return result_message
        
    except Exception as e:
        return f"❌ 实验结果处理失败: {str(e)}"


def _validate_parameter_values(processed_df: pd.DataFrame, state: dict, campaign: Campaign) -> list:
    """
    验证上传的参数值是否与推荐值不同
    
    注意：此函数接收的是预处理后的数据（已规范化SMILES等），确保比较的准确性
    
    Args:
        processed_df: 预处理后的实验结果DataFrame（已规范化SMILES等）
        state: 工具上下文状态
        campaign: BayBE Campaign对象
    
    Returns:
        list: 警告消息列表（如果有参数被修改）
    """
    warnings = []
    latest_recommendations = state.get("latest_recommendations", [])
    
    if not latest_recommendations:
        return warnings
    
    # 将推荐转换为 DataFrame 以便比较
    try:
        recommendations_df = pd.DataFrame(latest_recommendations)
    except Exception as e:
        print(f"[WARN] 无法将推荐值转换为DataFrame: {e}")
        return warnings
    
    # ===== 关键修复：将推荐值的列名映射为Campaign参数名 =====
    # 推荐值可能使用原始列名（如 SubstanceA_SMILE），需要映射为Campaign参数名（如 SubstanceA_molecule）
    # 使用与 _preprocess_experimental_results 相同的映射逻辑
    column_mapping = {}
    for param_name in campaign.searchspace.parameter_names:
        if param_name in recommendations_df.columns:
            # 推荐值已经使用Campaign参数名，无需映射
            continue
        else:
            # 尝试查找对应的原始列名
            if param_name.endswith('_molecule'):
                base_name = param_name.replace('_molecule', '')
                # 尝试多种可能的列名格式（与预处理逻辑一致）
                possible_cols = [
                    f"{base_name}_SMILE",
                    f"{base_name}_SMILES",
                    f"{base_name}_smile",
                    f"{base_name}_smiles",
                ]
                for possible_col in possible_cols:
                    # 精确匹配
                    if possible_col in recommendations_df.columns:
                        column_mapping[possible_col] = param_name
                        print(f"[DEBUG] 推荐值列名映射: {possible_col} -> {param_name}")
                        break
                    # 大小写不敏感匹配
                    matching_cols = [col for col in recommendations_df.columns if col.upper() == possible_col.upper()]
                    if matching_cols:
                        column_mapping[matching_cols[0]] = param_name
                        print(f"[DEBUG] 推荐值列名映射（大小写不敏感）: {matching_cols[0]} -> {param_name}")
                        break
    
    # 应用列名映射
    if column_mapping:
        recommendations_df = recommendations_df.rename(columns=column_mapping)
        print(f"[DEBUG] 已应用 {len(column_mapping)} 个推荐值列名映射")
    
    # 只比较实际对应的行数（如果上传的数据行数多于推荐，只比较前N行）
    num_rows_to_compare = min(len(processed_df), len(recommendations_df))
    
    if num_rows_to_compare == 0:
        return warnings
    
    print(f"[DEBUG] _validate_parameter_values: 比较 {num_rows_to_compare} 行数据")
    print(f"[DEBUG] 推荐数据列: {list(recommendations_df.columns)}")
    print(f"[DEBUG] 处理数据列: {list(processed_df.columns)}")
    
    # 比较每个参数列
    for param_name in campaign.searchspace.parameter_names:
        if param_name not in processed_df.columns:
            continue
        
        if param_name not in recommendations_df.columns:
            print(f"[DEBUG] 参数 '{param_name}' 不在推荐数据中，跳过比较")
            continue
        
        # 获取参数类型，以便正确处理比较
        param_obj = None
        if hasattr(campaign.searchspace, 'parameters'):
            for p in campaign.searchspace.parameters:
                if p.name == param_name:
                    param_obj = p
                    break
        
        # 比较每一行的值
        for idx in range(num_rows_to_compare):
            try:
                uploaded_value = processed_df.iloc[idx][param_name]
                recommended_value = recommendations_df.iloc[idx][param_name]
                
                # 跳过NaN值（可能是预处理时被标记为无效的值）
                if pd.isna(uploaded_value) or pd.isna(recommended_value):
                    continue
                
                # 对于数值参数，允许小的浮点误差
                if isinstance(uploaded_value, (int, float)) and isinstance(recommended_value, (int, float)):
                    if abs(uploaded_value - recommended_value) > 1e-6:
                        warnings.append(
                            f"实验 {idx + 1} 的参数 '{param_name}' 被修改: "
                            f"推荐值={recommended_value}, 实际值={uploaded_value}"
                        )
                        print(f"[DEBUG] 检测到参数修改: 行{idx+1}, {param_name}, 推荐={recommended_value}, 实际上传={uploaded_value}")
                
                # 对于分类参数（如分子SMILES），比较规范化后的值
                elif isinstance(param_obj, CategoricalParameter):
                    # 对于SMILES，应该已经规范化，直接比较字符串
                    uploaded_str = str(uploaded_value).strip()
                    recommended_str = str(recommended_value).strip()
                    
                    # 如果推荐值看起来像是化合物名称而不是SMILES，尝试从映射中查找SMILES
                    if param_name.endswith('_molecule') and recommended_str and not recommended_str.startswith(('C', 'c', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', '[')):
                        # 可能是化合物名称，尝试从映射中查找对应的SMILES
                        smiles_to_name_map = state.get("smiles_to_name_map", {})
                        name_to_smiles_map = {v: k for k, v in smiles_to_name_map.items()}
                        if recommended_str in name_to_smiles_map:
                            recommended_str = name_to_smiles_map[recommended_str].strip()
                            print(f"[DEBUG] 推荐值 '{recommended_str[:50]}...' 是化合物名称，已映射为SMILES")
                    
                    # 如果推荐值不是规范化后的SMILES，尝试规范化
                    if param_name.endswith('_molecule') and recommended_str and recommended_str not in param_obj.values:
                        try:
                            from baybe.utils.chemistry import get_canonical_smiles
                            canonical_recommended = get_canonical_smiles(recommended_str)
                            if canonical_recommended and canonical_recommended.strip():
                                recommended_str = canonical_recommended.strip()
                                print(f"[DEBUG] 推荐值已规范化: {recommended_str[:50]}...")
                        except Exception as e:
                            print(f"[DEBUG] 无法规范化推荐值 '{recommended_str[:50]}...': {e}")
                    
                    if uploaded_str != recommended_str:
                        warnings.append(
                            f"实验 {idx + 1} 的参数 '{param_name}' 被修改: "
                            f"推荐值={recommended_str}, 实际值={uploaded_str}"
                        )
                        print(f"[DEBUG] 检测到SMILES参数修改: 行{idx+1}, {param_name}")
                        print(f"[DEBUG]   推荐值: {recommended_str[:80]}...")
                        print(f"[DEBUG]   实际上传: {uploaded_str[:80]}...")
                        print(f"[DEBUG]   是否相等: {uploaded_str == recommended_str}")
                
                # 对于其他分类参数，字符串比较
                else:
                    uploaded_str = str(uploaded_value).strip()
                    recommended_str = str(recommended_value).strip()
                    
                    if uploaded_str != recommended_str:
                        warnings.append(
                            f"实验 {idx + 1} 的参数 '{param_name}' 被修改: "
                            f"推荐值={recommended_str}, 实际值={uploaded_str}"
                        )
                        
            except Exception as e:
                print(f"[WARN] 比较参数 '{param_name}' 行 {idx + 1} 时出错: {e}")
                continue
    
    if warnings:
        print(f"[DEBUG] _validate_parameter_values: 检测到 {len(warnings)} 个参数修改警告")
    else:
        print(f"[DEBUG] _validate_parameter_values: 所有参数值都与推荐值匹配")
    
    return warnings


def _extract_bounds_from_param(param_bounds) -> tuple:
    """
    从参数边界中安全提取 min 和 max 值
    
    支持多种格式：
    - 元组/列表: (min, max)
    - Interval 对象: Interval(left, right)
    - 其他可迭代对象
    
    Returns:
        tuple: (min_val, max_val) 或 (None, None) 如果无法提取
    """
    if param_bounds is None:
        return None, None
    
    try:
        # 如果是元组或列表
        if isinstance(param_bounds, (tuple, list)) and len(param_bounds) >= 2:
            return float(param_bounds[0]), float(param_bounds[1])
        
        # 如果是 Interval 对象（BayBE 可能使用）
        if hasattr(param_bounds, 'left') and hasattr(param_bounds, 'right'):
            return float(param_bounds.left), float(param_bounds.right)
        
        # 如果是 Interval 对象（pandas Interval）
        if hasattr(param_bounds, 'left') and hasattr(param_bounds, 'right'):
            return float(param_bounds.left), float(param_bounds.right)
        
        # 尝试转换为元组
        if hasattr(param_bounds, '__iter__') and not isinstance(param_bounds, str):
            bounds_list = list(param_bounds)
            if len(bounds_list) >= 2:
                return float(bounds_list[0]), float(bounds_list[1])
        
        print(f"[WARN] 无法从 bounds 中提取值: {type(param_bounds)}, {param_bounds}")
        return None, None
        
    except Exception as e:
        print(f"[WARN] 提取 bounds 时出错: {e}, type: {type(param_bounds)}")
        return None, None


def _preprocess_experimental_results(results_df: pd.DataFrame, campaign: Campaign, state: dict = None) -> pd.DataFrame:
    """
    预处理实验结果数据
    
    包括：
    1. 列名映射（将 _SMILE 列映射到 _molecule 列）
    2. 列筛选（只保留需要的参数和目标列）
    3. 数据类型转换（参数列和目标列都转换为数值类型）
    4. 参数值验证（检查是否在搜索空间范围内）
    5. 清理无效数据（移除包含NaN的行）
    """
    processed_df = results_df.copy()
    
    # ===== 步骤1: 列名映射（将 _SMILE 映射到 _molecule）=====
    # 获取 Campaign 需要的所有参数列名
    required_param_names = list(campaign.searchspace.parameter_names)
    
    # 查找需要映射的列（用户上传的 CSV 中有 _SMILE 列，但 Campaign 需要 _molecule 列）
    column_mapping = {}
    for param_name in required_param_names:
        if param_name.endswith('_molecule'):
            # 尝试找到对应的 _SMILE 列（支持多种变体）
            base_name = param_name.replace('_molecule', '')
            possible_smile_cols = [
                f"{base_name}_SMILE",
                f"{base_name}_SMILEs",
                f"{base_name}_smile",
                f"{base_name}_smiles",
                f"{base_name}_SMILE",
            ]
            
            # 检查是否存在这些列（大小写不敏感）
            for possible_col in possible_smile_cols:
                # 精确匹配
                if possible_col in processed_df.columns:
                    column_mapping[possible_col] = param_name
                    print(f"[DEBUG] 列名映射: {possible_col} -> {param_name}")
                    break
                # 大小写不敏感匹配
                matching_cols = [col for col in processed_df.columns if col.upper() == possible_col.upper()]
                if matching_cols:
                    column_mapping[matching_cols[0]] = param_name
                    print(f"[DEBUG] 列名映射（大小写不敏感）: {matching_cols[0]} -> {param_name}")
                    break
    
    # 应用列名映射
    if column_mapping:
        processed_df = processed_df.rename(columns=column_mapping)
        print(f"[DEBUG] 已应用 {len(column_mapping)} 个列名映射")
    
    # 确保只包含Campaign需要的列
    required_columns = list(campaign.searchspace.parameter_names) + [t.name for t in campaign.objective.targets]
    
    # 保留需要的列
    available_columns = [col for col in required_columns if col in processed_df.columns]
    missing_columns = [col for col in required_columns if col not in processed_df.columns]
    
    if missing_columns:
        print(f"[ERROR] 缺少必需的列: {missing_columns}")
        print(f"[DEBUG] 原始数据列: {list(results_df.columns)}")
        print(f"[DEBUG] 需要的列: {required_columns}")
        print(f"[DEBUG] 可用的列: {available_columns}")
        # 对于分子列，提供更友好的错误信息
        molecule_missing = [col for col in missing_columns if col.endswith('_molecule')]
        if molecule_missing:
            print(f"[ERROR] 缺少分子列: {molecule_missing}")
            print(f"[ERROR] 提示: 请确保 CSV 文件中包含对应的 _SMILE 列（如 SubstanceA_SMILE）")
    
    processed_df = processed_df[available_columns]
    
    print(f"[DEBUG] _preprocess_experimental_results: processing {len(processed_df)} rows")
    print(f"[DEBUG] Available columns: {available_columns}")
    if missing_columns:
        print(f"[WARN] Missing columns: {missing_columns}")
    
    # 数据类型转换和验证
    print(f"[DEBUG] 开始数据类型转换，当前数据行数: {len(processed_df)}")
    
    # 1. 转换参数列
    for param_name in campaign.searchspace.parameter_names:
        if param_name not in processed_df.columns:
            print(f"[WARN] 参数 '{param_name}' 不在数据列中")
            continue
        
        # ===== 关键修复：检查参数类型，分类参数（分子列）不需要转换为数值 =====
        # 获取参数对象以判断类型
        param_obj = None
        if hasattr(campaign.searchspace, 'parameters'):
            for p in campaign.searchspace.parameters:
                if p.name == param_name:
                    param_obj = p
                    break
        
        # ===== 根据参数类型进行不同的处理 =====
        if param_obj and isinstance(param_obj, CategoricalParameter):
            # 分类参数（通常是 _molecule 列），需要规范化SMILES并验证
            print(f"[DEBUG] 参数 '{param_name}' 是分类参数（CategoricalParameter），开始SMILES规范化")
            
            # 获取Campaign中定义的有效SMILES值
            valid_smiles_set = set(param_obj.values)
            print(f"[DEBUG] Campaign中 '{param_name}' 的有效SMILES数量: {len(valid_smiles_set)}")
            
            # 规范化上传的SMILES值
            from baybe.utils.chemistry import get_canonical_smiles
            
            normalized_values = []
            normalization_errors = []
            original_values = processed_df[param_name].copy()
            
            for idx, smiles_value in original_values.items():
                if pd.isna(smiles_value):
                    normalized_values.append(None)
                    continue
                
                smiles_str = str(smiles_value).strip()
                
                # 首先检查是否已经是有效值（完全匹配）
                if smiles_str in valid_smiles_set:
                    normalized_values.append(smiles_str)
                    print(f"[DEBUG] 行 {idx}: SMILES '{smiles_str[:30]}...' 已匹配有效值")
                    continue
                
                # 尝试规范化
                try:
                    canonical_smiles = get_canonical_smiles(smiles_str)
                    
                    if canonical_smiles and canonical_smiles.strip():
                        # 检查规范化后的值是否在有效值列表中
                        if canonical_smiles in valid_smiles_set:
                            normalized_values.append(canonical_smiles)
                            if canonical_smiles != smiles_str:
                                print(f"[DEBUG] 行 {idx}: SMILES已规范化 '{smiles_str[:30]}...' -> '{canonical_smiles[:30]}...'")
                            else:
                                print(f"[DEBUG] 行 {idx}: SMILES无需规范化，已匹配")
                        else:
                            # 规范化后仍不在有效值列表中
                            normalized_values.append(None)
                            normalization_errors.append({
                                'row': idx,
                                'original': smiles_str,
                                'canonical': canonical_smiles,
                                'error': '规范化后的SMILES不在Campaign的有效值列表中'
                            })
                            print(f"[WARN] 行 {idx}: SMILES规范化后仍不匹配。原始: '{smiles_str[:50]}...', 规范化: '{canonical_smiles[:50]}...'")
                    else:
                        normalized_values.append(None)
                        normalization_errors.append({
                            'row': idx,
                            'original': smiles_str,
                            'canonical': None,
                            'error': '无法规范化SMILES（返回None）'
                        })
                        print(f"[WARN] 行 {idx}: 无法规范化SMILES '{smiles_str[:50]}...'")
                        
                except Exception as e:
                    normalized_values.append(None)
                    normalization_errors.append({
                        'row': idx,
                        'original': smiles_str,
                        'canonical': None,
                        'error': f'规范化失败: {str(e)}'
                    })
                    print(f"[ERROR] 行 {idx}: SMILES规范化异常 '{smiles_str[:50]}...': {str(e)}")
            
            # 替换为规范化后的值
            processed_df[param_name] = normalized_values
            
            # 如果有规范化错误，提供详细的错误信息
            if normalization_errors:
                error_details = []
                for err in normalization_errors[:5]:  # 最多显示5个错误
                    error_details.append(
                        f"  行 {err['row']}: 原始值='{err['original'][:60]}...'"
                    )
                    if err['canonical']:
                        error_details.append(f"    规范化后='{err['canonical'][:60]}...'")
                    error_details.append(f"    错误: {err['error']}")
                
                print(f"[ERROR] {param_name} 有 {len(normalization_errors)} 个SMILES无法匹配:")
                for detail in error_details:
                    print(f"[ERROR] {detail}")
                
                # 保存错误信息到state，以便在错误消息中显示
                if 'smiles_matching_errors' not in state:
                    state['smiles_matching_errors'] = {}
                state['smiles_matching_errors'][param_name] = {
                    'errors': normalization_errors,
                    'valid_smiles': list(valid_smiles_set)
                }
            
            valid_count = processed_df[param_name].notna().sum()
            print(f"[DEBUG] 参数 '{param_name}' SMILES规范化完成，有效值数量: {valid_count}/{len(processed_df)}")
        elif param_obj and isinstance(param_obj, NumericalDiscreteParameter):
            # ===== 关键修复：离散数值参数需要精确匹配离散值列表 =====
            original_values = processed_df[param_name].copy()
            valid_discrete_values = set(param_obj.values)
            print(f"[DEBUG] 参数 '{param_name}' 是离散数值参数（NumericalDiscreteParameter）")
            print(f"[DEBUG] 有效离散值: {sorted(valid_discrete_values)}")
            print(f"[DEBUG] 原始值示例: {original_values.head(3).tolist()}")
            
            # 转换为数值类型（但保持与离散值列表相同的类型）
            processed_df[param_name] = pd.to_numeric(processed_df[param_name], errors='coerce')
            
            # 检查转换失败的值
            failed_conversions = processed_df[param_name].isna() & original_values.notna()
            if failed_conversions.any():
                failed_indices = failed_conversions[failed_conversions].index.tolist()
                failed_values = [original_values.iloc[idx] for idx in failed_indices]
                print(f"[WARN] 离散参数 '{param_name}' 在行 {failed_indices} 转换失败，值: {failed_values}")
            
            # ===== 关键：验证值是否在离散值列表中（考虑浮点数精度） =====
            invalid_values = []
            for idx, val in processed_df[param_name].items():
                if pd.isna(val):
                    continue
                # 检查值是否在有效离散值列表中（考虑浮点数精度）
                val_float = float(val)
                matched = False
                for valid_val in valid_discrete_values:
                    valid_float = float(valid_val)
                    # 使用小的容差（1e-9）来处理浮点数精度问题
                    if abs(val_float - valid_float) < 1e-9:
                        # 匹配成功，但需要确保类型一致（使用离散值列表中的实际类型）
                        processed_df.at[idx, param_name] = valid_val
                        matched = True
                        break
                
                if not matched:
                    invalid_values.append((idx, val))
            
            if invalid_values:
                error_msg = f"离散参数 '{param_name}' 的值不在有效离散值列表中:\n"
                for idx, val in invalid_values[:5]:
                    error_msg += f"  行 {idx}: 值 {val} 不在 {sorted(valid_discrete_values)} 中\n"
                if len(invalid_values) > 5:
                    error_msg += f"  ... 还有 {len(invalid_values) - 5} 个无效值\n"
                print(f"[ERROR] {error_msg}")
            else:
                print(f"[DEBUG] 离散参数 '{param_name}' 所有值都匹配有效离散值列表，有效值数量: {processed_df[param_name].notna().sum()}/{len(processed_df)}")
        else:
            # 连续数值参数（NumericalContinuousParameter）
            original_values = processed_df[param_name].copy()
            print(f"[DEBUG] 转换连续数值参数 '{param_name}': 原始值示例 = {original_values.head(3).tolist()}")
            processed_df[param_name] = pd.to_numeric(processed_df[param_name], errors='coerce')
            
            # 检查转换失败的值
            failed_conversions = processed_df[param_name].isna() & original_values.notna()
            if failed_conversions.any():
                failed_indices = failed_conversions[failed_conversions].index.tolist()
                failed_values = [original_values.iloc[idx] for idx in failed_indices]
                print(f"[WARN] 连续数值参数 '{param_name}' 在行 {failed_indices} 转换失败，值: {failed_values}")
            else:
                print(f"[DEBUG] 连续数值参数 '{param_name}' 转换成功，有效值数量: {processed_df[param_name].notna().sum()}/{len(processed_df)}")
    
    # 2. 转换目标列
    for target in campaign.objective.targets:
        if target.name in processed_df.columns:
            original_values = processed_df[target.name].copy()
            print(f"[DEBUG] 转换目标 '{target.name}': 原始值示例 = {original_values.head(3).tolist()}")
            processed_df[target.name] = pd.to_numeric(processed_df[target.name], errors='coerce')
            
            # 检查转换失败的值
            failed_conversions = processed_df[target.name].isna() & original_values.notna()
            if failed_conversions.any():
                failed_indices = failed_conversions[failed_conversions].index.tolist()
                failed_values = [original_values.iloc[idx] for idx in failed_indices]
                print(f"[WARN] 目标 '{target.name}' 在行 {failed_indices} 转换失败，值: {failed_values}")
            else:
                print(f"[DEBUG] 目标 '{target.name}' 转换成功，有效值数量: {processed_df[target.name].notna().sum()}/{len(processed_df)}")
        else:
            print(f"[WARN] 目标 '{target.name}' 不在数据列中")
    
    # 3. 验证参数值是否在搜索空间范围内
    validation_errors = []
    for param_name in campaign.searchspace.parameter_names:
        if param_name not in processed_df.columns:
            continue
        
        # 获取参数的边界（对于连续参数）
        param_bounds = None
        # 尝试从 searchspace.parameters 获取
        if hasattr(campaign.searchspace, 'parameters'):
            for param in campaign.searchspace.parameters:
                if param.name == param_name:
                    if hasattr(param, 'bounds'):
                        param_bounds = param.bounds
                    break
        
        # 如果没找到，尝试从 continuous subspace 获取
        if param_bounds is None and hasattr(campaign.searchspace, 'continuous'):
            continuous = campaign.searchspace.continuous
            if continuous is not None and hasattr(continuous, 'parameter_names'):
                if param_name in continuous.parameter_names:
                    # 从 continuous 的 bounds 字典获取
                    if hasattr(continuous, 'bounds') and param_name in continuous.bounds:
                        param_bounds = continuous.bounds[param_name]
                    # 或者从参数对象获取
                    elif hasattr(continuous, 'parameters'):
                        for param in continuous.parameters:
                            if param.name == param_name and hasattr(param, 'bounds'):
                                param_bounds = param.bounds
                                break
        
        if param_bounds:
            min_val, max_val = _extract_bounds_from_param(param_bounds)
            if min_val is not None and max_val is not None:
                # 检查超出范围的值
                out_of_range = (processed_df[param_name] < min_val) | (processed_df[param_name] > max_val)
                if out_of_range.any():
                    out_indices = out_of_range[out_of_range].index.tolist()
                    out_values = [processed_df[param_name].iloc[idx] for idx in out_indices]
                    for idx, val in zip(out_indices, out_values):
                        validation_errors.append(
                            f"行 {idx}: 参数 '{param_name}' 值 {val} 超出范围 [{min_val}, {max_val}]"
                        )
                    print(f"[WARN] 参数 '{param_name}' 在行 {out_indices} 超出范围 [{min_val}, {max_val}]")
            else:
                print(f"[WARN] 无法获取参数 '{param_name}' 的边界范围，跳过范围验证")
    
    if validation_errors:
        error_msg = "参数值验证失败:\n" + "\n".join(validation_errors[:5])
        if len(validation_errors) > 5:
            error_msg += f"\n... 还有 {len(validation_errors) - 5} 个错误"
        print(f"[ERROR] {error_msg}")
        # 不立即返回错误，先尝试清理数据
    
    # 4. 验证SMILES值是否在Campaign的有效值列表中（规范化后）
    # 对于分子参数，检查规范化后的值是否都在有效值列表中
    for param_name in campaign.searchspace.parameter_names:
        if param_name not in processed_df.columns:
            continue
        
        param_obj = None
        if hasattr(campaign.searchspace, 'parameters'):
            for p in campaign.searchspace.parameters:
                if p.name == param_name:
                    param_obj = p
                    break
        
        if param_obj and isinstance(param_obj, CategoricalParameter) and param_name.endswith('_molecule'):
            # 检查是否有无法匹配的SMILES
            invalid_mask = processed_df[param_name].isna()
            if invalid_mask.any():
                invalid_rows = processed_df[invalid_mask]
                valid_smiles = list(param_obj.values)
                
                print(f"[ERROR] 参数 '{param_name}' 有 {invalid_mask.sum()} 个SMILES无法匹配到Campaign的有效值")
                print(f"[ERROR] Campaign期望的有效SMILES（共{len(valid_smiles)}个）:")
                for i, smiles in enumerate(valid_smiles, 1):
                    print(f"[ERROR]   {i}. {smiles}")
                
                # 显示无法匹配的原始值（如果有的话）
                if len(invalid_rows) > 0:
                    print(f"[ERROR] 无法匹配的行和原始值:")
                    for idx in invalid_rows.index[:5]:
                        original_val = results_df.iloc[idx] if idx < len(results_df) else "N/A"
                        print(f"[ERROR]   行 {idx}: 原始值可能在其他列中")
    
    # 5. 移除包含NaN的行（在转换和验证之后）
    before_drop = len(processed_df)
    
    # 详细分析哪些行包含 NaN 以及原因
    if before_drop > 0:
        nan_analysis = {}
        for col in processed_df.columns:
            nan_count = processed_df[col].isna().sum()
            if nan_count > 0:
                nan_analysis[col] = nan_count
                nan_rows = processed_df[processed_df[col].isna()].index.tolist()
                print(f"[DEBUG] 列 '{col}' 有 {nan_count} 个 NaN 值，行索引: {nan_rows[:10]}{'...' if len(nan_rows) > 10 else ''}")
        
        if nan_analysis:
            print(f"[DEBUG] NaN 分析: {nan_analysis}")
            # 显示每行包含 NaN 的列数
            rows_with_nan = processed_df.isna().sum(axis=1)
            print(f"[DEBUG] 包含 NaN 的行数统计: {rows_with_nan.value_counts().to_dict()}")
    
    processed_df = processed_df.dropna()
    after_drop = len(processed_df)
    
    if before_drop > after_drop:
        removed_count = before_drop - after_drop
        print(f"[WARN] 移除了 {removed_count} 行包含 NaN 的数据 (从 {before_drop} 行减少到 {after_drop} 行)")
        
        # 如果所有行都被移除，提供详细诊断
        if after_drop == 0:
            print(f"[ERROR] 所有 {before_drop} 行数据都被移除了！")
            print(f"[ERROR] 可能的原因：")
            print(f"[ERROR] 1. 所有参数列或目标列都包含 NaN")
            print(f"[ERROR] 2. 数据类型转换失败（文本无法转换为数字）")
            print(f"[ERROR] 3. 数据格式不正确（如包含占位符文本 '<请填写测量值>'）")
    
    # 5. 如果还有验证错误，返回详细错误信息
    if validation_errors and processed_df.empty:
        print(f"[ERROR] 数据预处理失败：{len(validation_errors)} 个验证错误，且处理后数据为空")
        return pd.DataFrame()  # 返回空 DataFrame，让调用者处理错误
    
    print(f"[DEBUG] _preprocess_experimental_results: 最终处理了 {len(processed_df)} 行有效数据")
    
    if len(processed_df) == 0:
        print(f"[ERROR] 数据预处理后为空！")
        print(f"[ERROR] 原始数据行数: {len(results_df)}")
        print(f"[ERROR] 原始数据列: {list(results_df.columns)}")
        print(f"[ERROR] 需要的列: {required_columns}")
        print(f"[ERROR] 可用的列: {available_columns}")
        print(f"[ERROR] 缺失的列: {missing_columns}")
    
    return processed_df


# =============================================================================
# 收敛性检查工具
# =============================================================================

def check_convergence(tool_context: ToolContext) -> str:
    """
    检查优化收敛性（集成高级分析）
    
    使用 AdaptiveRecommendationStrategy 进行详细的收敛性分析，包括：
    - 改进率计算
    - 平台期检测
    - 振荡检测
    - 收敛置信度评估
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法进行收敛性分析。"
    
    try:
        campaign = _get_campaign_from_cache(session_id)
        current_round = state.get("optimization_round", 0)
        
        if not campaign:
            return "❌ 未找到BayBE Campaign。"
        
        # 早期阶段检查
        if current_round < 2:
            return f"""
📊 **优化进展分析** (轮次 {current_round})

🔄 **当前状态**: 优化初期
- 完成轮次: {current_round}
- 建议: 继续收集更多实验数据

🎯 **下一步建议**:
- 再进行 2-3 轮实验以建立有效的代理模型
- 推荐批次大小: 3-5 个实验
- 重点: 探索参数空间
            """
        
        measurements = campaign.measurements
        targets = [t.name for t in campaign.objective.targets]
        
        # 数据量检查
        if len(measurements) < 5:
            return "📊 实验数据不足，无法进行收敛性分析。建议至少进行5轮实验。"
        
        # ===== 尝试使用高级收敛分析 =====
        if ADVANCED_CONVERGENCE_AVAILABLE:
            try:
                strategy = AdaptiveRecommendationStrategy()
                
                # 分析优化进展
                progress_analysis = strategy._analyze_optimization_progress(campaign)
                
                # 获取收敛指标
                convergence_indicators = progress_analysis.get("convergence_indicators", {})
                improvement_metrics = progress_analysis.get("improvement_metrics", {})
                improvement_rate = progress_analysis.get("improvement_rate", 0.0)
                convergence_trend = progress_analysis.get("convergence_trend", "unknown")
                
                # 构建详细的分析报告
                report_parts = [
                    f"📊 **优化收敛性分析** (轮次 {current_round})",
                    "",
                    f"📈 **总体状态**: {convergence_trend}",
                    f"- 最近改进率: {improvement_rate:.2%}",
                    f"- 总实验数: {len(measurements)}",
                    ""
                ]
                
                # 收敛指标详情
                is_converging = convergence_indicators.get("is_converging", False)
                confidence = convergence_indicators.get("convergence_confidence", 0.0)
                plateau = convergence_indicators.get("plateau_detected", False)
                oscillation = convergence_indicators.get("oscillation_detected", False)
                
                report_parts.append("🔍 **收敛指标**:")
                report_parts.append(f"- 收敛状态: {'✅ 已收敛' if is_converging else '⏳ 未收敛'}")
                report_parts.append(f"- 收敛置信度: {confidence:.0%}")
                if plateau:
                    report_parts.append("- 平台期: ✅ 已检测到（最近5个值变化 < 5%）")
                if oscillation:
                    report_parts.append("- 振荡: ⚠️ 检测到（可能存在噪声或参数空间复杂）")
                report_parts.append("")
                
                # 各目标详细分析
                if improvement_metrics:
                    report_parts.append("📊 **各目标分析**:")
                    for target, metrics in improvement_metrics.items():
                        best_val = metrics.get("best_value", 0.0)
                        recent_rate = metrics.get("recent_improvement_rate", 0.0)
                        report_parts.append(f"- {target}:")
                        report_parts.append(f"  • 最优值: {best_val:.2f}")
                        report_parts.append(f"  • 最近改进率: {recent_rate:.2%}")
                    report_parts.append("")
                
                # 建议
                if is_converging or (improvement_rate < 0.02 and confidence > 0.5):
                    report_parts.append("🛑 **建议**: 考虑停止优化")
                    report_parts.append("- 改进速度已明显放缓或达到平台期")
                    report_parts.append("- 可以使用当前最优参数进行生产验证")
                    report_parts.append("")
                    report_parts.append("📊 **下一步**: 建议运行Fitting Agent进行详细结果分析和可视化")
                elif improvement_rate < 0.05:
                    report_parts.append("⚠️ **建议**: 接近收敛，可考虑再优化1-2轮")
                    report_parts.append("- 改进速度已放缓")
                    report_parts.append("- 建议进行最后1-2轮精细优化")
                else:
                    report_parts.append("🚀 **建议**: 继续优化")
                    report_parts.append("- 仍有显著改进空间")
                    report_parts.append(f"- 建议再进行2-3轮实验（当前改进率: {improvement_rate:.2%}）")
                
                return "\n".join(report_parts)
                
            except Exception as e:
                print(f"[WARN] 高级收敛分析失败，回退到基础分析: {str(e)}")
                # 回退到基础分析
                pass
        
        # ===== 基础收敛分析（回退方案） =====
        recent_improvement = 0
        
        for target in targets:
            if target in measurements.columns:
                values = measurements[target].values
                if len(values) >= 3:
                    recent_avg = np.mean(values[-3:])
                    previous_avg = np.mean(values[-6:-3]) if len(values) >= 6 else values[0]
                    improvement = abs((recent_avg - previous_avg) / previous_avg) if previous_avg != 0 else 0
                    recent_improvement = max(recent_improvement, improvement)
        
        if recent_improvement < 0.05:
            return f"""
📊 **优化收敛性分析** (轮次 {current_round})

🎯 **收敛状态**: 接近收敛 
- 最近改进率: {recent_improvement:.3f}
- 总实验数: {len(measurements)}

🛑 **建议**: 考虑停止优化
- 改进速度已明显放缓
- 可以使用当前最优参数进行生产

📊 **最终分析**: 建议运行Fitting Agent进行详细结果分析
            """
        else:
            return f"""
📊 **优化收敛性分析** (轮次 {current_round})

▶️ **收敛状态**: 仍在改进中
- 最近改进率: {recent_improvement:.3f}
- 总实验数: {len(measurements)}

🚀 **建议**: 继续优化
- 仍有显著改进空间
- 建议再进行2-3轮实验
            """
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] 收敛性分析失败: {error_traceback}")
        return f"❌ 收敛性分析失败: {str(e)}"


# =============================================================================
# 辅助工具
# =============================================================================

def generate_result_template(tool_context: ToolContext) -> str:
    """
    生成实验结果上传模板（可选工具）
    
    从统一的实验记录表中提取待实验的行（experiment_status = "pending"），
    生成一个独立的模板文件，方便用户只关注当前轮次的实验。
    
    💡 **使用说明**:
    - **方式一（推荐）**: 直接在统一实验记录表 `experiment_log.csv` 中填写实验结果，然后上传该文件
    - **方式二（可选）**: 使用本工具生成独立的模板文件，填写后上传模板文件
    
    两种方式都可以，方式一更简单直接，方式二可以只关注当前轮次的实验。
    
    Returns:
        str: 模板文件路径和说明
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    campaign = _get_campaign_from_cache(session_id)
    
    if not campaign:
        return "❌ 未找到BayBE Campaign，无法生成模板。"
    
    try:
        # 从统一实验记录表中提取待实验的行
        unified_experiment_log = state.get("unified_experiment_log_path")
        
        if not unified_experiment_log or not os.path.exists(unified_experiment_log):
            return """
⚠️ **统一实验记录表不存在**

无法创建结果模板，因为统一实验记录表尚未创建。

💡 **解决方案**:
1. 首先使用 generate_recommendations 工具生成实验推荐
2. 推荐会自动添加到统一实验记录表中
3. 然后使用本工具提取待实验的行

🔄 **工作流程**:
generate_recommendations → generate_result_template → 进行实验 → upload_experimental_results
            """
        
        # 读取统一实验记录表
        experiment_log_df = _read_csv_clean(unified_experiment_log)
        
        # 提取待实验的行（experiment_status = "pending"）
        pending_experiments = experiment_log_df[
            experiment_log_df.get("experiment_status", pd.Series(["completed"] * len(experiment_log_df))) == "pending"
        ]
        
        if len(pending_experiments) == 0:
            return """
⚠️ **没有待实验的记录**

统一实验记录表中没有状态为 "pending" 的实验。

💡 **可能的原因**:
1. 所有推荐都已标记为完成
2. 尚未生成新的推荐

🔄 **建议**:
- 使用 generate_recommendations 生成新的实验推荐
- 或检查统一实验记录表的状态列
            """
        
        # 移除状态列（用户不需要看到）
        template_df = pending_experiments.drop(columns=["experiment_status"], errors='ignore')
        
        # 确保目标列有占位符（如果为空）
        for target in campaign.objective.targets:
            target_col = target.name
            if target_col in template_df.columns:
                # 将空值或占位符替换为明确的占位符
                template_df[target_col] = template_df[target_col].replace(
                    ["", None, pd.NA], "<请填写测量值>"
                )
            else:
                template_df[target_col] = "<请填写测量值>"
        
        # 保存模板（覆盖统一记录表中的待实验行，用户填写后可以上传）
        session_dir = state.get("session_dir", ".")
        current_round = state.get("optimization_round", 0) + 1
        template_file = os.path.join(
            session_dir,
            f"experiment_template_round_{current_round}.csv"
        )
        
        template_df.to_csv(template_file, index=False, encoding="utf-8-sig")
        
        # 保存模板文件路径到 state
        state["current_template_file"] = template_file
        
        # 统计名称列数量
        name_columns_count = len([col for col in template_df.columns if col.endswith("_name")])
        
        # 生成详细说明
        instructions = f"""
📋 **实验结果上传模板已生成**（可选工具）

📄 **模板文件**: {template_file}
📄 **统一实验记录表**: {unified_experiment_log}

💡 **两种上传方式**:

**方式一（推荐）**: 直接在统一实验记录表中填写
- 打开 `experiment_log.csv`
- 找到状态为 "pending" 的实验行
- 填写目标列的测量值
- 直接上传 `experiment_log.csv` 即可

**方式二（可选）**: 使用独立模板文件
- 使用本工具生成独立的模板文件（只包含当前轮次的待实验）
- 在模板文件中填写实验结果
- 上传模板文件

两种方式都可以，方式一更简单直接，方式二可以只关注当前轮次的实验。

📊 **模板结构**:
- 参数列: {len(campaign.searchspace.parameter_names)} 列（已自动填写推荐值）
- 化合物名称列: {name_columns_count} 列（便于阅读）
- 目标列: {len(campaign.objective.targets)} 列（需要您填写测量值）
- 总行数: {len(template_df)} 行（对应 {len(template_df)} 个待实验）

✏️ **填写说明**:

1. **参数列**（已自动填写，⚠️ **强烈建议不要修改**）:
   {', '.join(campaign.searchspace.parameter_names)}
   
   ⚠️ **修改参数值的后果**:
   - 偏离 BayBE 的最优探索路径，可能降低优化效率
   - 影响代理模型的更新质量，导致后续推荐不准确
   - 如果修改后的值超出搜索空间边界，可能导致优化失败
   - 仅在特殊情况下（如实验条件限制、安全约束）才应修改
   
   **化合物名称列**（便于阅读，上传时可保留或删除）:
   {', '.join([col for col in template_df.columns if col.endswith("_name")]) or '无'}
   
2. **目标列**（需要填写实验测量值）:
"""
        
        for target in campaign.objective.targets:
            instructions += f"   • {target.name}: 优化方向={target.mode}, 预期范围={target.bounds}\n"
        
        instructions += f"""
   
3. **元数据列**（可选）:
   • experiment_id: 实验编号
   • experiment_date: 实验日期
   • operator: 操作员
   • notes: 备注（可记录异常情况）

📤 **上传方式**（推荐使用统一表格）:
1. **方式一（推荐）**: 直接在统一实验记录表中填写，然后上传:
   upload_experimental_results("{unified_experiment_log}")
   
2. **方式二**: 填写模板文件后上传:
   upload_experimental_results("{template_file}")
   
3. **方式三**: 直接粘贴CSV内容

⚠️ **重要提示**:
- ⚠️ **参数列**: 已按 BayBE 推荐设置，修改会影响优化效果（详见上方说明）
- ✅ **目标列**: 必须填写实际测量值，不要保留占位符 `<请填写测量值>`
- ✅ **数值格式**: 所有数值列应为纯数字，避免文字说明
- 📝 **异常记录**: 如有参数修改或实验异常，请在 notes 列中详细记录原因

🔄 **下一步**: 
完成实验并填写模板后 → upload_experimental_results → check_convergence
        """
        
        return instructions
        
    except Exception as e:
        return f"❌ 模板生成失败: {str(e)}"


def get_campaign_info(tool_context: ToolContext) -> str:
    """
    获取当前Campaign的详细信息
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    campaign = _get_campaign_from_cache(session_id)
    
    if not campaign:
        return "❌ 未找到Campaign对象。请先使用 build_campaign_and_recommend 工具构建Campaign。"
    
    try:
        info = f"""
📋 **当前Campaign信息**:

🔧 **参数配置**:
- 参数数量: {len(campaign.searchspace.parameter_names)}
- 参数名称: {', '.join(campaign.searchspace.parameter_names)}

🎯 **目标配置**:
- 目标数量: {len(campaign.objective.targets)}
- 目标名称: {', '.join([t.name for t in campaign.objective.targets])}

📊 **搜索空间状态**:
- 离散参数数: {len(campaign.searchspace.discrete.exp_rep) if hasattr(campaign.searchspace, 'discrete') and campaign.searchspace.discrete is not None else 'N/A'}
- 连续参数数: {len(campaign.searchspace.continuous.parameter_names) if hasattr(campaign.searchspace, 'continuous') and campaign.searchspace.continuous is not None else 'N/A'}

🔄 **Campaign状态**:
- 是否有历史数据: {'是 (' + str(len(campaign.measurements)) + '条)' if hasattr(campaign, 'measurements') and len(campaign.measurements) > 0 else '否'}
- 优化轮次: {state.get('optimization_round', 0)}
- 准备就绪: {'是' if state.get('ready_for_optimization', False) else '否'}
"""
        
        return info
        
    except Exception as e:
        return f"❌ 获取Campaign信息失败: {str(e)}"


def check_agent_health(tool_context: ToolContext) -> str:
    """
    检查Recommender Agent的健康状态
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    campaign = _get_campaign_from_cache(session_id)
    campaign_exists = campaign is not None

    health_report = {
        "campaign_exists": campaign_exists,
        "campaign_valid": False,
        "recommendations_generated": state.get("recommendations_generated", False),
        "awaiting_results": state.get("awaiting_experimental_results", False),
        "optimization_round": state.get("optimization_round", 0),
        "last_recommendation_time": state.get("last_recommendation_time", "从未"),
        "last_upload_time": state.get("last_result_upload_time", "从未")
    }
    
    # 检查Campaign
    if campaign_exists and campaign:
        health_report["campaign_valid"] = (
            hasattr(campaign, 'searchspace') and 
            hasattr(campaign, 'objective')
        )
    
    # 判断系统状态
    if all([health_report["campaign_exists"], health_report["campaign_valid"]]):
        system_status = "🟢 系统正常"
        status_emoji = "✅"
    elif health_report["campaign_exists"]:
        system_status = "🟡 系统部分就绪"
        status_emoji = "⚠️"
    else:
        system_status = "🔴 系统未初始化"
        status_emoji = "❌"
    
    return f"""
🏥 **Recommender Agent 健康检查**

{status_emoji} **系统状态**: {system_status}

📋 **详细诊断**:
✅ Campaign对象存在: {health_report["campaign_exists"]}
✅ Campaign结构有效: {health_report["campaign_valid"]}

📊 **运行状态**:
• 已生成推荐: {'是' if health_report["recommendations_generated"] else '否'}
• 等待实验结果: {'是' if health_report["awaiting_results"] else '否'}
• 优化轮次: {health_report["optimization_round"]}

⏰ **时间信息**:
• 最后推荐时间: {health_report["last_recommendation_time"]}
• 最后上传时间: {health_report["last_upload_time"]}

{'🔧 **建议**: 系统运行正常，可以继续优化' if system_status == '🟢 系统正常' else '⚠️ **建议**: 使用 build_campaign_and_recommend 工具初始化系统'}
    """
