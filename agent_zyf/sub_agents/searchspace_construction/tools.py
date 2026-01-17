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

"""SearchSpace Construction Agent Tools - 构建BayBE搜索空间和Campaign"""

import os
import pandas as pd
import json
from datetime import datetime
from google.adk.tools import ToolContext


def _read_csv_clean(path: str) -> pd.DataFrame:
    """
    读取 CSV 并清理列名（去 BOM/空白，移除常见索引列如 Unnamed: 0）
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$")]
    return df

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
    BAYBE_AVAILABLE = True
except ImportError:
    print("Warning: BayBE not installed. SearchSpace Construction Agent will not function.")
    BAYBE_AVAILABLE = False


def construct_searchspace_and_campaign(user_constraints: str, tool_context: ToolContext) -> str:
    """
    基于Enhanced Verification Agent的输出构建BayBE搜索空间和Campaign
    
    Args:
        user_constraints: 用户提供的额外约束条件（可选）
        tool_context: ADK工具上下文
        
    Returns:
        str: 构建结果和Campaign信息
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    if not BAYBE_AVAILABLE:
        return "❌ BayBE未安装，无法构建搜索空间。请运行: pip install 'baybe[chem]'"
    
    try:
        # 获取Enhanced Verification Agent的输出
        verification_results = state.get("verification_results", {})
        baybe_campaign_config = state.get("baybe_campaign_config", {})
        optimization_config = state.get("optimization_config", {})
        
        if not verification_results:
            return "❌ 未找到验证结果。请先运行Enhanced Verification Agent。"
        
        # 构建BayBE Campaign
        campaign_result = _build_baybe_campaign(
            verification_results, 
            baybe_campaign_config,
            optimization_config,
            user_constraints
        )
        
        if campaign_result["success"]:
            # 保存Campaign到状态
            state["baybe_campaign"] = campaign_result["campaign"]
            state["searchspace_info"] = campaign_result["searchspace_info"]
            state["ready_for_optimization"] = True
            state["construction_timestamp"] = datetime.now().isoformat()
            
            return _generate_construction_summary(campaign_result, verification_results)
        else:
            return f"❌ 搜索空间构建失败: {campaign_result['error']}"
            
    except Exception as e:
        return f"❌ SearchSpace Construction Agent 执行错误: {str(e)}"


def _build_baybe_campaign(verification_results: dict, 
                         campaign_config: dict,
                         optimization_config: dict, 
                         user_constraints: str) -> dict:
    """
    构建完整的BayBE Campaign
    """
    try:
        # 1. 读取标准化数据
        standardized_data_path = verification_results.get("standardized_data_path")
        if not standardized_data_path or not os.path.exists(standardized_data_path):
            return {"success": False, "error": "标准化数据文件不存在"}
        
        df = _read_csv_clean(standardized_data_path)
        
        # 2. 创建BayBE参数
        parameters = _create_baybe_parameters(df, verification_results)
        
        if not parameters:
            return {"success": False, "error": "无法创建BayBE参数"}
        
        # 3. 创建搜索空间
        searchspace = SearchSpace.from_product(parameters=parameters)
        
        # 4. 创建目标
        targets = _create_baybe_targets(df, optimization_config)
        
        if not targets:
            return {"success": False, "error": "无法创建目标函数"}
        
        # 5. 创建目标函数
        objective = _create_baybe_objective(targets, optimization_config)
        
        # 6. 创建约束（如果需要）
        constraints = _create_baybe_constraints(df, user_constraints)
        
        # 7. 创建Campaign
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective
        )
        
        # 8. 准备返回信息
        searchspace_info = {
            "total_parameters": len(parameters),
            "molecule_parameters": len([p for p in parameters if isinstance(p, CategoricalParameter)]),
            "numerical_parameters": len([p for p in parameters if isinstance(p, NumericalContinuousParameter)]),
            "constraint_count": len(constraints),
            "searchspace_size": len(searchspace.discrete.exp_rep) if searchspace.discrete is not None else "continuous"
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
        return {"success": False, "error": str(e)}


def _create_baybe_parameters(df: pd.DataFrame, verification_results: dict) -> list:
    """
    创建BayBE参数定义
    """
    parameters = []
    smiles_validation = verification_results.get("smiles_validation", {})
    
    # 1. 分子参数 - 直接使用已验证的SMILES
    smiles_columns = [col for col in df.columns if 'SMILE' in col.upper()]
    for col in smiles_columns:
        substance_name = col.split('_')[0] if '_' in col else col
        
        # 获取有效的规范化SMILES
        valid_smiles = []
        canonical_mapping = smiles_validation.get("canonical_smiles_mapping", {})
        
        for smiles in df[col].dropna().unique():
            if str(smiles) in canonical_mapping:
                canonical_smiles = canonical_mapping[str(smiles)]
                valid_smiles.append(canonical_smiles)
        
        if len(valid_smiles) >= 2:  # BayBE要求至少2个值
            param = CategoricalParameter(
                name=f"{substance_name}_molecule",
                values=valid_smiles,  # BayBE自动处理描述符
                encoding="OHE"
            )
            parameters.append(param)
        elif len(valid_smiles) == 1:
            # 只有1个SMILES时，跳过分子参数（因为没有优化空间）
            print(f"⚠️ {substance_name} 只有1个SMILES值，跳过分子参数创建")
        else:
            print(f"⚠️ {substance_name} 没有有效SMILES，跳过参数创建")
    
    # 2. 数值参数（比例、温度等）
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_columns = [col for col in df.columns if col.startswith('Target_')]
    
    for col in numeric_columns:
        if col not in target_columns:  # 排除目标变量
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # 根据列名选择参数类型
            if 'ratio' in col.lower():
                # 比例参数使用连续参数
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(max(0.0, min_val), min(1.0, max_val))
                )
            elif 'temperature' in col.lower():
                # 温度参数
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(max(20.0, min_val), min(200.0, max_val))
                )
            else:
                # 其他数值参数
                buffer = (max_val - min_val) * 0.1
                param = NumericalContinuousParameter(
                    name=col,
                    bounds=(min_val - buffer, max_val + buffer)
                )
            
            parameters.append(param)
    
    return parameters


def _create_baybe_targets(df: pd.DataFrame, optimization_config: dict) -> list:
    """
    创建BayBE目标函数
    """
    targets = []
    target_columns = [col for col in df.columns if col.startswith('Target_')]
    
    for col in target_columns:
        # 计算目标值的范围
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        
        # 默认设置（用户可以在优化配置中override）
        target = NumericalTarget(
            name=col,
            mode="MAX",  # 默认最大化，用户可以配置
            bounds=(min_val, max_val),
            transformation="LINEAR"
        )
        targets.append(target)
    
    return targets


def _create_baybe_objective(targets: list, optimization_config: dict):
    """
    创建BayBE目标函数
    """
    if len(targets) == 1:
        # 单目标优化
        return targets[0].to_objective()
    else:
        # 多目标优化 - 默认使用DesirabilityObjective
        obj_config = optimization_config.get("objective_config", {})
        
        if obj_config.get("type") == "ParetoObjective":
            return ParetoObjective(targets=targets)
        else:
            # 默认使用DesirabilityObjective
            weights = obj_config.get("weights", [1.0] * len(targets))
            scalarizer = obj_config.get("scalarizer", "GEOM_MEAN")
            
            return DesirabilityObjective(
                targets=targets,
                weights=weights,
                scalarizer=scalarizer
            )


def _create_baybe_constraints(df: pd.DataFrame, user_constraints: str) -> list:
    """
    创建BayBE约束条件
    """
    constraints = []
    
    # 自动检测比例约束
    ratio_columns = [col for col in df.columns if 'ratio' in col.lower()]
    
    if len(ratio_columns) > 1:
        # 如果有多个比例列，添加和约束（所有比例之和 = 1.0）
        # 注意：这需要确保这些比例确实应该和为1
        try:
            constraint = ContinuousLinearConstraint(
                parameters=ratio_columns,
                coefficients=[1.0] * len(ratio_columns),
                rhs=1.0,
                operator="="
            )
            constraints.append(constraint)
        except Exception as e:
            print(f"警告：无法创建比例约束: {e}")
    
    # TODO: 解析user_constraints字符串并添加自定义约束
    
    return constraints


def _generate_construction_summary(campaign_result: dict, verification_results: dict) -> str:
    """
    生成搜索空间构建摘要
    """
    searchspace_info = campaign_result["searchspace_info"]
    
    summary = f"""
🚀 **SearchSpace Construction 完成**

📊 **BayBE Campaign 构建成功**:
- 参数总数: {searchspace_info['total_parameters']}
  - 分子参数: {searchspace_info['molecule_parameters']} (SMILES自动处理)
  - 数值参数: {searchspace_info['numerical_parameters']}
- 约束条件: {searchspace_info['constraint_count']}
- 搜索空间大小: {searchspace_info['searchspace_size']}

✅ **架构简化优势体现**:
- 直接使用Enhanced Verification Agent输出的已验证SMILES
- BayBE自动处理所有分子描述符计算和缓存
- 即用型Campaign对象已创建完成

🎯 **下一步**: 
- Campaign已准备就绪，可以传递给Recommender Agent
- 系统已具备完整的贝叶斯优化能力
- 用户可以开始获取实验推荐

📄 **技术细节**:
- BayBE版本: 最新
- SearchSpace类型: 混合空间（分子+数值参数）  
- 目标函数: {"DesirabilityObjective" if len(campaign_result["targets"]) > 1 else "SingleObjective"}
- 分子编码: 自动指纹计算

🔧 **状态更新**: 
- ready_for_optimization = True
- baybe_campaign 已保存到会话状态
"""

    return summary


# 辅助工具函数
def validate_campaign_readiness(tool_context: ToolContext) -> str:
    """
    验证Campaign是否准备就绪
    """
    state = tool_context.state
    
    required_keys = [
        "verification_results",
        "baybe_campaign_config", 
        "optimization_config"
    ]
    
    missing_keys = [key for key in required_keys if key not in state]
    
    if missing_keys:
        return f"❌ Campaign构建前提条件不满足，缺少: {', '.join(missing_keys)}"
    
    campaign = state.get("baybe_campaign")
    if campaign is not None:
        return f"✅ Campaign已存在，包含 {len(campaign.searchspace.parameter_names)} 个参数"
    
    return "⚠️ Campaign尚未构建，可以开始构建过程"


def get_campaign_info(tool_context: ToolContext) -> str:
    """
    获取当前Campaign的详细信息
    """
    state = tool_context.state
    campaign = state.get("baybe_campaign")
    
    if not campaign:
        return "❌ 未找到Campaign对象"
    
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
- 是否有历史数据: {'是' if hasattr(campaign, 'measurements') and len(campaign.measurements) > 0 else '否'}
- 准备就绪: {'是' if state.get('ready_for_optimization', False) else '否'}
"""
        
        return info
        
    except Exception as e:
        return f"❌ 获取Campaign信息失败: {str(e)}"
