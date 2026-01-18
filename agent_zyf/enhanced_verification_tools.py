

"""Enhanced Verification Agent Tools - 实现7个核心任务的工具函数"""

import os
import uuid
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from google.adk.tools import ToolContext


def _read_csv_clean(path: str) -> pd.DataFrame:
    """
    读取 CSV 并清理列名（去 BOM/空白，移除常见索引列如 Unnamed: 0）
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$")]
    return df


def _detect_suspicious_headers(df: pd.DataFrame) -> list:
    """
    检测疑似被说明文字/参数范围污染的表头列名。
    """
    suspicious = []
    for col in df.columns:
        col_str = str(col)
        # 明显的说明性文本/句子关键词
        if any(token in col_str for token in ["最大化", "最小化", "接受建议", "没有其他约束", "每批", "最多", "总共", "帕累托", "约束"]):
            suspicious.append(col_str)
            continue
        # 含有中文标点或长句特征
        if any(token in col_str for token in ["，", "。", "："]):
            if len(col_str) > 20:
                suspicious.append(col_str)
                continue
        # 参数范围直接写入表头的常见形式
        if "[" in col_str and "]" in col_str and ("ratio" in col_str.lower() or "target" in col_str.lower()):
            suspicious.append(col_str)
            continue
    return suspicious


def _reset_verification_state(state: dict, reason: str) -> None:
    """清理验证相关状态，避免污染后续流程"""
    for key in [
        "verification_results",
        "baybe_campaign_config",
        "optimization_config",
        "ready_for_optimization",
        "searchspace_info",
        "campaign_built",
    ]:
        state.pop(key, None)
    state["verification_status"] = f"failed:{reason}"

# 导入化学知识库
from .chemistry_knowledge_base import ChemistryKnowledgeBase

# 注意：以下导入需要安装BayBE
# pip install baybe
try:
    from baybe.utils.chemistry import get_canonical_smiles, name_to_smiles
    from baybe.parameters import CategoricalParameter, NumericalContinuousParameter, NumericalDiscreteParameter
    from baybe.parameters.enum import SubstanceEncoding
    BAYBE_AVAILABLE = True
except ImportError:
    print("Warning: BayBE not installed. Please run: pip install baybe")
    BAYBE_AVAILABLE = False


class SimplifiedSMILESValidator:
    """
    简化的SMILES验证器 - 无需手动计算描述符
    BayBE会在Campaign中自动处理所有分子描述符计算
    """
    
    def validate_smiles_data(self, data: pd.DataFrame) -> dict:
        """
        只验证SMILES有效性，不计算描述符
        """
        validation_results = {
            "canonical_smiles_mapping": {},
            "invalid_smiles": [],
            "substances_validated": []
        }
        
        if not BAYBE_AVAILABLE:
            # 降级处理：使用基本验证
            return self._basic_smiles_validation(data)
        
        # 识别SMILES列
        smiles_columns = [col for col in data.columns if 'SMILE' in col.upper()]
        
        for col in smiles_columns:
            substance_name = col.split('_')[0] if '_' in col else col
            
            for idx, smiles in data[col].items():
                if pd.isna(smiles) or smiles == '':
                    continue
                    
                try:
                    # 只验证并获取规范化SMILES
                    canonical_smiles = get_canonical_smiles(str(smiles))
                    
                    if canonical_smiles is not None:
                        validation_results["canonical_smiles_mapping"][smiles] = canonical_smiles
                    else:
                        validation_results["invalid_smiles"].append({
                            "substance": substance_name,
                            "row": idx,
                            "smiles": smiles,
                            "error": "无法解析分子结构"
                        })
                        
                except Exception as e:
                    validation_results["invalid_smiles"].append({
                        "substance": substance_name,
                        "row": idx, 
                        "smiles": smiles,
                        "error": str(e)
                    })
            
            validation_results["substances_validated"].append(substance_name)
            
        return validation_results
    

def _auto_correct_invalid_smiles(
    df: pd.DataFrame,
    validation_results: dict,
) -> list:
    """
    尝试根据化合物名称自动纠正无效的SMILES（在迭代开始前执行）

    策略：
    1. 使用验证阶段记录的 invalid_smiles 列表，找到对应的行和物质
    2. 在同一行中查找对应的名称列（如 SubstanceA_name）
    3. 使用 name_to_smiles 将名称转换为SMILES
    4. 使用 get_canonical_smiles 规范化后写回原DataFrame

    注意：
    - 只在 BayBE 可用时启用（需要 name_to_smiles 和 get_canonical_smiles）
    - 只修改当前 DataFrame，不直接修改 validation_results

    Returns:
        list[dict]: 自动纠正记录列表
    """
    if not BAYBE_AVAILABLE:
        return []

    invalid_items = validation_results.get("invalid_smiles", [])
    if not invalid_items:
        return []

    corrections = []

    # 所有 SMILES 列
    smiles_columns = [col for col in df.columns if "SMILE" in str(col).upper()]

    for item in invalid_items:
        try:
            substance = item.get("substance")
            row_idx = item.get("row")
            orig_smiles = item.get("smiles")

            if row_idx is None or row_idx not in df.index:
                continue

            if orig_smiles is None or (isinstance(orig_smiles, float) and np.isnan(orig_smiles)):
                continue

            orig_smiles_str = str(orig_smiles).strip()
            if not orig_smiles_str:
                continue

            # 找到对应的SMILES列：同一物质，且该行值等于原始SMILES
            target_col = None
            for col in smiles_columns:
                col_prefix = str(col).split("_")[0] if "_" in str(col) else str(col)
                if substance and col_prefix != str(substance):
                    continue
                cell_val = df.at[row_idx, col]
                if str(cell_val).strip() == orig_smiles_str:
                    target_col = col
                    break

            if target_col is None:
                continue

            # 查找对应的名称列（参考 _extract_smiles_name_mapping_with_canonical 的前缀逻辑）
            col_upper = str(target_col).upper()
            if target_col.endswith("_molecule"):
                prefix = target_col.rsplit("_molecule", 1)[0]
            else:
                idx = col_upper.find("_SMILE")
                prefix = target_col[:idx] if idx > 0 else str(target_col).split("_")[0]

            name_col = None
            candidates = [
                f"{prefix}_name",
                f"{prefix}_NAME",
                f"{prefix}_Name",
                f"{prefix}name",
                f"{prefix}NAME",
            ]
            for cand in candidates:
                if cand in df.columns and cand != target_col:
                    name_col = cand
                    break
            if name_col is None:
                # 再尝试大小写不敏感匹配
                for df_col in df.columns:
                    if str(df_col).upper() == f"{prefix.upper()}_NAME" and df_col != target_col:
                        name_col = df_col
                        break

            if name_col is None:
                continue

            name_val = df.at[row_idx, name_col]
            if name_val is None or (isinstance(name_val, float) and np.isnan(name_val)):
                continue

            name_str = str(name_val).strip()
            if not name_str:
                continue

            # 使用名称尝试生成 SMILES
            try:
                generated_smiles = name_to_smiles(name_str)
            except Exception as e:
                print(f"[WARN] name_to_smiles 失败: substance={substance}, name={name_str}, error={e}")
                continue

            if not generated_smiles:
                continue

            try:
                canonical = get_canonical_smiles(str(generated_smiles)) or str(generated_smiles)
            except Exception as e:
                print(f"[WARN] get_canonical_smiles 失败: smiles={generated_smiles}, error={e}")
                continue

            if not canonical:
                continue

            # 写回 DataFrame（在标准化前直接修正原始数据）
            df.at[row_idx, target_col] = canonical

            corrections.append(
                {
                    "substance": substance,
                    "row": int(row_idx),
                    "original_smiles": orig_smiles_str,
                    "corrected_smiles": canonical,
                    "name_column": name_col,
                    "name_value": name_str,
                }
            )

        except Exception as e:
            print(f"[WARN] _auto_correct_invalid_smiles 内部错误: {e}")
            continue

    if corrections:
        print(f"[DEBUG] 自动纠正了 {len(corrections)} 个SMILES: "
              f"{[c['substance'] for c in corrections[:3]]}"
              f"{' ...' if len(corrections) > 3 else ''}")

    return corrections

    def _basic_smiles_validation(self, data: pd.DataFrame) -> dict:
        """
        基本SMILES验证（当BayBE不可用时）
        """
        validation_results = {
            "canonical_smiles_mapping": {},
            "invalid_smiles": [],
            "substances_validated": []
        }
        
        smiles_columns = [col for col in data.columns if 'SMILE' in col.upper()]
        
        for col in smiles_columns:
            substance_name = col.split('_')[0] if '_' in col else col
            
            for idx, smiles in data[col].items():
                if pd.isna(smiles) or smiles == '':
                    continue
                    
                # 基本格式检查
                if isinstance(smiles, str) and len(smiles) > 0:
                    validation_results["canonical_smiles_mapping"][smiles] = smiles  # 保持原样
                else:
                    validation_results["invalid_smiles"].append({
                        "substance": substance_name,
                        "row": idx,
                        "smiles": smiles,
                        "error": "SMILES格式错误"
                    })
            
            validation_results["substances_validated"].append(substance_name)
            
        return validation_results
    
    def prepare_baybe_parameters(self, data: pd.DataFrame, validation_results: dict) -> list:
        """
        为BayBE准备参数定义，使用原始SMILES
        BayBE内部会自动处理描述符计算
        """
        if not BAYBE_AVAILABLE:
            return []
            
        parameters = []
        
        # 1. 分子参数 - 直接使用SMILES字符串
        smiles_columns = [col for col in data.columns if 'SMILE' in col.upper()]
        for col in smiles_columns:
            substance_name = col.split('_')[0] if '_' in col else col
            
            # 获取有效的SMILES值
            valid_smiles = []
            for smiles in data[col].dropna().unique():
                if str(smiles) in validation_results["canonical_smiles_mapping"]:
                    valid_smiles.append(validation_results["canonical_smiles_mapping"][str(smiles)])
            
            if len(valid_smiles) >= 2:  # BayBE要求至少2个值
                param = CategoricalParameter(
                    name=f"{substance_name}_molecule",
                    values=valid_smiles,  # BayBE会自动处理这些SMILES的描述符
                    encoding="OHE"
                )
                parameters.append(param)
            elif len(valid_smiles) == 1:
                # 只有1个SMILES时，跳过分子参数（因为没有优化空间）
                print(f"⚠️ {substance_name} 只有1个SMILES值，跳过分子参数创建")
            else:
                print(f"⚠️ {substance_name} 没有有效SMILES，跳过参数创建")
        
        # 2. 数值参数（比例等）
        ratio_columns = [col for col in data.columns if 'ratio' in col.lower()]
        for col in ratio_columns:
            # 安全的数值转换和范围计算
            numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(numeric_data) == 0:
                print(f"⚠️ {col} 列没有有效的数值数据，跳过参数创建")
                continue
            min_val = float(numeric_data.min())
            max_val = float(numeric_data.max())
            
            param = NumericalContinuousParameter(
                name=col,
                bounds=(max(0.0, min_val), min(1.0, max_val))
            )
            parameters.append(param)
            
        return parameters


class IntelligentParameterAdvisor:
    """
    基于化学知识库的智能参数建议系统
    
    架构设计原则：
    1. 知识库(KB)提供硬约束 - 如"环氧固化不超过250°C"
    2. 计算工具提供物质属性 - 如RDKit计算分子量、LogP等
    3. LLM负责整合和交互 - 理解用户意图，选择合适的知识库条目
    4. 用户最终确认 - 领域专家拍板
    
    注意：LLM不适合直接推演精确的扩展百分比，应该由知识库提供典型范围
    """
    
    def __init__(self):
        """初始化参数建议器，加载化学知识库"""
        self.knowledge_base = ChemistryKnowledgeBase()
        self.reaction_type = None  # 缓存识别的反应类型
    
    def analyze_experimental_context(self, data: pd.DataFrame, user_description: str = "") -> dict:
        """
        分析实验背景，提供智能参数建议
        
        流程：
        1. 识别反应类型（从物质名称和用户描述）
        2. 从知识库获取该反应类型的典型参数范围
        3. 结合当前数据范围，生成建议边界
        4. 返回建议供用户确认（而非直接使用）
        """
        suggestions = {}
        
        # 1. 识别反应类型
        substances = self._extract_substance_names(data)
        self.reaction_type = self.knowledge_base.identify_reaction_type(
            substances, user_description
        )
        
        # 2. 从知识库获取参数建议
        kb_suggestions = self.knowledge_base.get_parameter_suggestions(
            self.reaction_type, data
        )
        
        # 3. 分析分子类型和特性
        molecular_analysis = self._analyze_molecules(data)
        
        # 4. 生成综合参数边界建议
        for col in data.columns:
            # 跳过目标列
            if col.startswith('Target_'):
                continue
            
            numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(numeric_data) == 0:
                continue
            
            # ===== 连续参数处理（禁用离散参数建议）=====
            current_range = (float(numeric_data.min()), float(numeric_data.max()))
            if 'ratio' in col.lower():
                # 从知识库获取边界建议
                suggested_bounds, reasoning = self._get_ratio_bounds_from_kb(
                    col, current_range, kb_suggestions
                )
                
                suggestions[col] = {
                    "current_range": [float(x) for x in current_range],
                    "suggested_bounds": [float(x) for x in suggested_bounds] if suggested_bounds else None,
                    "reasoning": reasoning,
                    "source": "knowledge_base",  # 标明来源
                    "constraints": self._suggest_constraints(col),
                    "requires_user_confirmation": True  # 需要用户确认
                }
                
            elif 'temperature' in col.lower():
                # 从知识库获取温度边界建议
                suggested_bounds, reasoning, safety_note = self._get_temperature_bounds_from_kb(
                    current_range, kb_suggestions
                )
                
                suggestions[col] = {
                    "current_range": [float(x) for x in current_range],
                    "suggested_bounds": [float(x) for x in suggested_bounds] if suggested_bounds else None,
                    "reasoning": reasoning,
                    "safety_note": safety_note,
                    "source": "knowledge_base",
                    "requires_user_confirmation": True
                }
            else:
                # 其他数值参数（默认连续）
                suggestions[col] = {
                    "current_range": [float(x) for x in current_range],
                    "suggested_bounds": [float(x) for x in current_range],  # 默认使用当前范围
                    "source": "data_analysis",
                    "requires_user_confirmation": True
                }
        
        # 5. 添加反应类型信息和安全警告
        suggestions["_reaction_info"] = {
            "identified_type": self.reaction_type,
            "reaction_name": self.knowledge_base.REACTION_TYPES.get(
                self.reaction_type, {}
            ).get("name", "未知反应类型"),
            "safety_warnings": kb_suggestions.get("safety_warnings", []),
            "molecular_analysis": molecular_analysis
        }
        
        return suggestions
    
    def _extract_substance_names(self, data: pd.DataFrame) -> List[str]:
        """从数据中提取物质名称"""
        substances = []
        name_columns = [col for col in data.columns if 'name' in col.lower()]
        
        for col in name_columns:
            substances.extend(data[col].dropna().astype(str).unique().tolist())
        
        return substances
    
    def _analyze_molecules(self, data: pd.DataFrame) -> dict:
        """分析分子类型和特性"""
        analysis = {}
        
        smiles_columns = [col for col in data.columns if 'SMILE' in col.upper()]
        for col in smiles_columns:
            smiles_list = data[col].dropna().astype(str).tolist()
            # 确保所有值都是Python原生类型，以便JSON序列化
            avg_len = float(np.mean([len(s) for s in smiles_list])) if smiles_list else 0.0
            mol_diversity = float(len(set(smiles_list)) / len(smiles_list)) if smiles_list else 0.0
            analysis[col] = {
                "molecule_count": int(len(set(smiles_list))),
                "avg_length": avg_len,
                "contains_aromatic": bool(any('c' in s.lower() or 'C' in s for s in smiles_list)),
                "molecular_diversity": mol_diversity
            }
        
        return analysis
    
    def _get_ratio_bounds_from_kb(
        self, 
        column_name: str, 
        current_range: Tuple[float, float],
        kb_suggestions: dict
    ) -> Tuple[Tuple[float, float], str]:
        """
        从知识库获取比例参数的建议边界
        
        策略：
        1. 优先使用知识库中该反应类型的典型范围
        2. 结合当前数据范围，取并集以扩大探索空间
        3. 应用安全约束（如比例必须在0-1之间）
        """
        min_val, max_val = current_range
        
        # 从知识库获取该反应类型的比例建议
        kb_ratio_info = kb_suggestions.get("ratios", {})
        kb_individual_bounds = kb_ratio_info.get("individual_bounds", {})
        
        # 检查知识库中是否有该列的具体建议
        if column_name in kb_individual_bounds:
            kb_bounds = kb_individual_bounds[column_name]
            kb_min = kb_bounds.get("min", 0.05)
            kb_max = kb_bounds.get("max", 0.95)
            
            # 取当前范围和知识库范围的并集，扩大探索空间
            suggested_min = min(min_val, kb_min)
            suggested_max = max(max_val, kb_max)
            
            reaction_name = self.knowledge_base.REACTION_TYPES.get(
                self.reaction_type, {}
            ).get("name", "化学反应")
            reasoning = f"基于{reaction_name}的典型配比范围，结合您当前数据的探索范围"
        else:
            # 知识库中没有具体建议，使用基于物质类型的规则
            suggested_min, suggested_max, reasoning = self._infer_ratio_bounds_by_substance_type(
                column_name, current_range
            )
        
        # 应用硬约束：比例必须在0-1之间
        suggested_min = max(0.0, suggested_min)
        suggested_max = min(1.0, suggested_max)
        
        return (suggested_min, suggested_max), reasoning
    
    def _infer_ratio_bounds_by_substance_type(
        self, 
        column_name: str, 
        current_range: Tuple[float, float]
    ) -> Tuple[float, float, str]:
        """
        根据物质类型推断比例边界（当知识库中没有具体信息时）
        使用材料属性知识库中的规则
        """
        min_val, max_val = current_range
        col_lower = column_name.lower()
        
        # 根据物质类型应用不同的规则（需进一步完善）
        if 'catalyst' in col_lower or '催化' in col_lower:
            # 催化剂通常用量少
            kb_info = self.knowledge_base.REACTION_TYPES.get(
                self.reaction_type, {}
            ).get("catalyst_concentration", (0.001, 0.1))
            return (
                kb_info[0], 
                kb_info[1],
                f"催化剂典型浓度范围 {kb_info[0]*100:.1f}%-{kb_info[1]*100:.1f}%"
            )
            
        elif 'hardener' in col_lower or '固化剂' in col_lower:
            # 固化剂有化学计量比要求
            ratio_info = self.knowledge_base.SAFETY_CONSTRAINTS.get(
                "ratio_constraints", {}
            ).get("epoxy_hardener", {})
            acceptable_range = ratio_info.get("acceptable_range", (0.2, 0.5))
            return (
                acceptable_range[0],
                acceptable_range[1],
                f"固化剂配比范围，考虑化学计量比 (欠固化风险<{ratio_info.get('under_cure_risk', '0.8')})"
            )
            
        elif 'diluent' in col_lower or '稀释' in col_lower or 'solvent' in col_lower:
            # 稀释剂/溶剂有最大用量限制
            diluent_info = self.knowledge_base.MATERIAL_PROPERTIES.get(
                "diluents", {}
            ).get("reactive_diluents", {})
            max_conc = diluent_info.get("max_concentration", 0.3)
            return (
                0.0,
                max_conc,
                f"稀释剂最大用量限制为 {max_conc*100:.0f}%，过量会影响性能"
            )
            
        else:
            # 一般物质：基于当前范围适度扩展（目前这里写死了，待进一步讨论）
            # 扩展因子基于数据稀疏性，而非LLM推演的百分比
            data_span = max_val - min_val
            if data_span < 0.1:
                # 数据范围很窄，建议扩大探索
                expansion = 0.15
                reasoning = f"当前数据仅探索了{data_span*100:.0f}%的范围，建议扩大至±15%以发现潜在最优点"
            else:
                expansion = 0.1
                reasoning = f"基于当前数据范围适度扩展±10%"
            
            return (
                max(0.0, min_val - expansion),
                min(1.0, max_val + expansion),
                reasoning
            )
    
    def _get_temperature_bounds_from_kb(
        self, 
        current_range: Tuple[float, float],
        kb_suggestions: dict
    ) -> Tuple[Tuple[float, float], str, str]:
        """
        从知识库获取温度参数的建议边界
        
        策略：
        1. 使用知识库中该反应类型的典型温度范围
        2. 应用安全约束（如最高温度限制）
        3. 返回安全提示信息
        """
        min_temp, max_temp = current_range
        
        # 从知识库获取温度建议
        kb_temp_info = kb_suggestions.get("temperature", {})
        
        if kb_temp_info:
            # 知识库有该反应类型的温度建议
            recommended_range = kb_temp_info.get("recommended_range", (-200, 400))
            optimal_range = kb_temp_info.get("optimal_range", recommended_range)
            safety_note = kb_temp_info.get("safety_note", "请注意温度安全控制")
            
            # 取当前范围和推荐范围的并集
            suggested_min = min(min_temp, recommended_range[0])
            suggested_max = max(max_temp, recommended_range[1])
            
            reasoning = (
                f"基于{kb_temp_info.get('reasoning', '化学反应')}，"
                f"典型范围{recommended_range[0]}-{recommended_range[1]}°C，"
                f"最优范围{optimal_range[0]}-{optimal_range[1]}°C"
            )
        else:
            # 使用安全约束中的默认值
            safety_limits = self.knowledge_base.SAFETY_CONSTRAINTS.get(
                "temperature_limits", {}
            ).get("epoxy_systems", {})
            
            safe_max = safety_limits.get("safe_max", 200)
            flash_point = safety_limits.get("flash_point_concern", 150)
            
            # 基于当前范围适度扩展，但不超过安全限制
            buffer = 20
            suggested_min = max(20, min_temp - buffer)
            suggested_max = min(safe_max, max_temp + buffer)
            
            reasoning = f"基于通用安全考虑，温度范围 {suggested_min}-{suggested_max}°C"
            safety_note = f"安全上限: {safe_max}°C, 闪点关注温度: {flash_point}°C"
        
        # 最终安全检查
        safety_limits = self.knowledge_base.SAFETY_CONSTRAINTS.get(
            "temperature_limits", {}
        ).get("epoxy_systems", {})
        absolute_max = safety_limits.get("decomposition_risk", 300)
        suggested_max = min(suggested_max, absolute_max)
        
        return (suggested_min, suggested_max), reasoning, safety_note
    
    def _suggest_constraints(self, column_name: str) -> list:
        """建议约束条件"""
        constraints = []
        
        if 'ratio' in column_name.lower():
            constraints.append({
                "type": "sum_constraint",
                "description": "所有比例之和应等于1.0",
                "implementation": "ContinuousLinearConstraint",
                "source": "knowledge_base"
            })
        
        return constraints
    
    def get_reaction_summary(self) -> str:
        """获取当前识别的反应类型摘要"""
        if self.reaction_type:
            return self.knowledge_base.get_reaction_info_summary(self.reaction_type)
        return "尚未识别反应类型"
    
    def validate_proposed_conditions(self, conditions: dict) -> Tuple[bool, List[str]]:
        """验证提议的实验条件是否合理"""
        if self.reaction_type:
            return self.knowledge_base.validate_experimental_conditions(
                conditions, self.reaction_type
            )
        return True, ["未识别反应类型，无法进行专业验证"]


class UserDefinedEncodingHandler:
    """
    识别和处理用户在CSV中提供的特殊编码信息
    支持动态识别和标准格式引导的混合策略
    """
    
    def __init__(self):
        # 定义列类型识别规则
        self.column_type_patterns = {
            "物理性质": {
                "keywords": ["density", "viscosity", "refractive", "melting", "boiling", "tg", "密度", "粘度", "折射", "熔点", "沸点", "玻璃化"],
                "value_type": "numerical",
                "baybe_param_type": "NumericalContinuousParameter"
            },
            "功能分类": {
                "keywords": ["catalyst", "additive", "modifier", "type", "category", "function", "催化剂", "添加剂", "改性剂", "类型", "功能"],
                "value_type": "categorical", 
                "baybe_param_type": "CategoricalParameter"
            },
            "供应商信息": {
                "keywords": ["supplier", "vendor", "batch", "lot", "grade", "purity", "供应商", "批次", "等级", "纯度"],
                "value_type": "categorical",
                "baybe_param_type": "CategoricalParameter" 
            },
            "成本信息": {
                "keywords": ["cost", "price", "availability", "expensive", "cheap", "成本", "价格", "可获得性"],
                "value_type": "numerical",
                "baybe_param_type": "NumericalContinuousParameter"
            },
            "工艺参数": {
                "keywords": ["temperature", "time", "pressure", "speed", "rpm", "温度", "时间", "压力", "转速"],
                "value_type": "numerical", 
                "baybe_param_type": "NumericalContinuousParameter"
            },
            "配方特性": {
                "keywords": ["hardener", "crosslinker", "solvent", "diluent", "固化剂", "交联剂", "溶剂", "稀释剂"],
                "value_type": "categorical",
                "baybe_param_type": "CategoricalParameter"
            }
        }
    
    def identify_user_special_substances(self, df: pd.DataFrame) -> dict:
        """
        识别用户定义的特殊物质（SMILES为空但有名称的物质）
        """
        user_special_substances = {
            "substances_without_smiles": [],
            "potential_encoding_columns": [],
            "custom_descriptors": {}
        }
        
        # 找到所有物质列对
        substance_pairs = []
        for col in df.columns:
            if 'name' in col.lower() and 'substance' in col.lower():
                substance_name = col
                # 寻找对应的SMILES列
                substance_prefix = col.replace('_name', '').replace('name', '')
                smiles_col = None
                for scol in df.columns:
                    if substance_prefix in scol and 'SMILE' in scol.upper():
                        smiles_col = scol
                        break
                
                if smiles_col:
                    substance_pairs.append((substance_name, smiles_col))
        
        # 识别特殊物质（有名称但SMILES为空/无效）
        for name_col, smiles_col in substance_pairs:
            for idx, row in df.iterrows():
                substance_name = row[name_col]
                smiles_value = row[smiles_col]
                
                # 如果有物质名称但SMILES为空或无效
                if (pd.notna(substance_name) and substance_name.strip() != "" and 
                    (pd.isna(smiles_value) or smiles_value == "" or str(smiles_value).strip() == "")):
                    
                    user_special_substances["substances_without_smiles"].append({
                        "name": substance_name,
                        "column_prefix": name_col.replace('_name', '').replace('name', ''),
                        "row": idx + 1
                    })
        
        # 寻找可能的自定义编码列
        for col in df.columns:
            # 寻找包含特征描述的列（不是standard的name/SMILES/ratio列）
            if not any(keyword in col.lower() for keyword in ['name', 'smile', 'ratio', 'target', 'unnamed']):
                if df[col].notna().any():  # 如果列有数据
                    user_special_substances["potential_encoding_columns"].append(col)
                    # 收集该列的唯一值作为可能的编码
                    unique_values = df[col].dropna().unique()
                    user_special_substances["custom_descriptors"][col] = unique_values.tolist()
        
        return user_special_substances
    
    def classify_user_columns(self, df: pd.DataFrame) -> dict:
        """
        智能分类用户的所有列，识别潜在的编码信息
        """
        column_classification = {
            "标准列": {"name": [], "smiles": [], "ratio": [], "target": []},
            "识别的扩展列": {},
            "未分类列": [],
            "建议的标准格式": {}
        }
        
        for col in df.columns:
            col_lower = col.lower()
            classified = False
            
            # 1. 识别标准列
            if any(keyword in col_lower for keyword in ['name', '名称']):
                column_classification["标准列"]["name"].append(col)
                classified = True
            elif any(keyword in col_lower for keyword in ['smile', 'smiles']):
                column_classification["标准列"]["smiles"].append(col)
                classified = True
            elif any(keyword in col_lower for keyword in ['ratio', '比例']):
                column_classification["标准列"]["ratio"].append(col)
                classified = True
            elif any(keyword in col_lower for keyword in ['target', '目标']):
                column_classification["标准列"]["target"].append(col)
                classified = True
            
            # 2. 动态识别扩展列类型
            if not classified:
                for category, pattern_info in self.column_type_patterns.items():
                    if any(keyword in col_lower for keyword in pattern_info["keywords"]):
                        if category not in column_classification["识别的扩展列"]:
                            column_classification["识别的扩展列"][category] = []
                        
                        # 分析列的实际数据类型
                        sample_data = df[col].dropna().head(10)
                        if len(sample_data) > 0:
                            data_analysis = self._analyze_column_content(sample_data)
                            
                            column_classification["识别的扩展列"][category].append({
                                "column_name": col,
                                "predicted_type": pattern_info["value_type"],
                                "actual_data_type": data_analysis["inferred_type"],
                                "sample_values": data_analysis["sample_values"],
                                "baybe_param_type": pattern_info["baybe_param_type"],
                                "confidence": data_analysis["confidence"]
                            })
                        classified = True
                        break
            
            # 3. 未能分类的列
            if not classified and col.strip() != "" and "unnamed" not in col_lower:
                column_classification["未分类列"].append(col)
        
        # 4. 生成标准格式建议
        column_classification["建议的标准格式"] = self._generate_standard_format_suggestions(df)
        
        return column_classification
    
    def _analyze_column_content(self, sample_data: pd.Series) -> dict:
        """
        分析列内容，推断数据类型和置信度
        """
        analysis = {
            "inferred_type": "unknown",
            "sample_values": sample_data.tolist()[:5],  # 前5个样本
            "confidence": 0.0
        }
        
        # 尝试数值转换
        numeric_conversion = pd.to_numeric(sample_data, errors='coerce')
        numeric_ratio = numeric_conversion.notna().sum() / len(sample_data)
        
        if numeric_ratio >= 0.8:  # 80%以上可转换为数值
            analysis["inferred_type"] = "numerical"
            analysis["confidence"] = numeric_ratio
        elif len(sample_data.unique()) <= max(10, len(sample_data) * 0.5):  # 唯一值较少
            analysis["inferred_type"] = "categorical"
            analysis["confidence"] = 1.0 - (len(sample_data.unique()) / len(sample_data))
        else:
            analysis["inferred_type"] = "text"
            analysis["confidence"] = 0.5
        
        return analysis
    
    def _generate_standard_format_suggestions(self, df: pd.DataFrame) -> dict:
        """
        基于当前数据生成标准格式建议
        """
        suggestions = {
            "推荐的列命名规范": {
                "物质信息": [
                    "SubstanceA_name (物质名称)",
                    "SubstanceA_SMILES (分子结构)", 
                    "SubstanceA_ratio (比例)",
                    "SubstanceA_type (物质类型: resin/hardener/catalyst/solvent/additive)",
                    "SubstanceA_supplier (供应商)",
                    "SubstanceA_grade (等级/纯度)",
                    "SubstanceA_batch (批次号)"
                ],
                "物理性质": [
                    "SubstanceA_density (密度 g/cm³)",
                    "SubstanceA_viscosity (粘度 Pa·s)",
                    "SubstanceA_tg (玻璃化温度 °C)",
                    "SubstanceA_melting_point (熔点 °C)"
                ],
                "工艺参数": [
                    "Process_temperature (反应温度 °C)",
                    "Process_time (反应时间 min)",
                    "Process_pressure (压力 bar)",
                    "Curing_temperature (固化温度 °C)"
                ],
                "成本信息": [
                    "SubstanceA_cost_per_kg (成本 元/kg)",
                    "SubstanceA_availability (可获得性: high/medium/low)"
                ]
            },
            "当前数据映射建议": {}
        }
        
        # 基于当前数据提供具体的重命名建议
        for col in df.columns:
            if "unnamed" in col.lower():
                continue
                
            col_lower = col.lower()
            mapping_suggestion = None
            
            # 尝试映射到标准格式
            if any(keyword in col_lower for keyword in ['稀释', 'diluent', 'solvent']):
                mapping_suggestion = f"{col} → SubstanceX_type (值: solvent/diluent)"
            elif any(keyword in col_lower for keyword in ['催化', 'catalyst']):
                mapping_suggestion = f"{col} → SubstanceX_type (值: catalyst)"
            elif any(keyword in col_lower for keyword in ['密度', 'density']):
                mapping_suggestion = f"{col} → SubstanceX_density"
            elif any(keyword in col_lower for keyword in ['粘度', 'viscosity']):
                mapping_suggestion = f"{col} → SubstanceX_viscosity"
                
            if mapping_suggestion:
                suggestions["当前数据映射建议"][col] = mapping_suggestion
        
        return suggestions
    
    def create_baybe_parameters_for_special_substances(self, user_special_data: dict, df: pd.DataFrame) -> list:
        """
        为用户定义的特殊物质创建BayBE参数配置（可序列化的配置，非实际对象）
        
        注意：此函数返回的是参数配置信息（可JSON序列化），
        而不是实际的BayBE Parameter对象。实际对象应在 Recommender Agent 中创建。
        """
        parameter_configs = []
        
        # 处理没有SMILES的特殊物质
        for special_substance in user_special_data["substances_without_smiles"]:
            substance_name = special_substance["name"]
            column_prefix = special_substance["column_prefix"]
            
            # 检查是否有对应的比例列
            ratio_col = f"{column_prefix}_ratio"
            if ratio_col in df.columns:
                # 对于特殊物质，我们可以用名称作为分类参数
                unique_names = df[f"{column_prefix}_name"].dropna().unique()
                if len(unique_names) > 1:
                    # 存储参数配置，而非实际的BayBE对象
                    parameter_configs.append({
                        "param_type": "CategoricalParameter",
                        "name": f"{column_prefix}_special_substance",
                        "values": [str(name) for name in unique_names],
                        "encoding": "OHE",
                        "source": "user_defined_special_substance",
                        "substance_type": "special_without_smiles",
                        "original_column": f"{column_prefix}_name"
                    })
        
        # 处理自定义描述符列
        for col, values in user_special_data["custom_descriptors"].items():
            if len(values) > 1:  # 只有当有多个不同值时才创建参数
                # 判断是数值还是分类数据
                numeric_values = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
                
                if len(numeric_values) == len(values):  # 全是数值
                    parameter_configs.append({
                        "param_type": "NumericalContinuousParameter",
                        "name": f"custom_{col}",
                        "bounds": [float(min(numeric_values)), float(max(numeric_values))],
                        "source": "user_defined_descriptor",
                        "original_column": col
                    })
                else:  # 分类数据
                    # 确保所有值都是可序列化的
                    serializable_values = []
                    for v in values:
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            serializable_values.append(v)
                        else:
                            serializable_values.append(str(v))
                    
                    parameter_configs.append({
                        "param_type": "CategoricalParameter",
                        "name": f"custom_{col}",
                        "values": serializable_values,
                        "encoding": "OHE",
                        "source": "user_defined_descriptor",
                        "original_column": col
                    })
        
        return parameter_configs
    
    def generate_standard_csv_template(self, num_substances: int = 4) -> str:
        """
        生成包含扩展列类型的标准CSV模板
        """
        headers = []
        
        # 为每个物质生成完整的列集合
        for i in range(num_substances):
            substance = chr(65 + i)  # A, B, C, D...
            headers.extend([
                f"Substance{substance}_name",
                f"Substance{substance}_SMILES", 
                f"Substance{substance}_ratio",
                f"Substance{substance}_type",           # 功能分类
                f"Substance{substance}_supplier",       # 供应商信息
                f"Substance{substance}_grade",          # 等级/纯度
                f"Substance{substance}_density",        # 物理性质
                f"Substance{substance}_viscosity",      # 物理性质
                f"Substance{substance}_cost_per_kg",    # 成本信息
                f"Substance{substance}_availability",   # 可获得性
            ])
        
        # 添加工艺参数
        headers.extend([
            "Process_temperature",      # 工艺参数
            "Process_time", 
            "Process_pressure",
            "Curing_temperature",
            "Mixing_speed"
        ])
        
        # 添加目标变量
        headers.extend([
            "Target_mechanical_strength",
            "Target_thermal_stability", 
            "Target_chemical_resistance",
            "Target_cost_effectiveness"
        ])
        
        # 生成示例数据行
        example_rows = []
        example_rows.append([
            # SubstanceA (主树脂)
            "Epoxy_Resin_E51", "CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4", "0.6", "epoxy_resin", "Supplier_A", "Industrial_Grade", "1.15", "800", "25.5", "high",
            # SubstanceB (固化剂)  
            "Hardener_DETA", "NCCNCCN", "0.3", "hardener", "Supplier_B", "Analytical_Grade", "0.95", "20", "18.2", "medium",
            # SubstanceC (稀释剂)
            "Diluent_A", "", "0.1", "diluent", "Supplier_C", "Industrial_Grade", "0.85", "5", "12.0", "high",
            # SubstanceD (添加剂)
            "Antioxidant_BHT", "CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C", "0.0", "antioxidant", "Supplier_D", "Analytical_Grade", "1.05", "1000", "45.8", "low",
            # 工艺参数
            "80", "120", "1.0", "150", "500",
            # 目标
            "85", "200", "95", "0.8"
        ])
        
        # 添加第二行示例数据
        example_rows.append([
            # SubstanceA
            "Epoxy_Resin_E44", "CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4", "0.7", "epoxy_resin", "Supplier_A", "Industrial_Grade", "1.18", "1200", "28.0", "high",
            # SubstanceB  
            "Hardener_IPDA", "C1CCC(CC1)N", "0.25", "hardener", "Supplier_B", "Analytical_Grade", "0.92", "15", "20.5", "medium",
            # SubstanceC
            "Diluent_B", "", "0.05", "diluent", "Supplier_E", "Industrial_Grade", "0.88", "8", "15.0", "medium",
            # SubstanceD
            "UV_Stabilizer", "CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)OCC(=O)OC)C(C)(C)C", "0.0", "uv_stabilizer", "Supplier_F", "Analytical_Grade", "1.02", "2000", "52.3", "low",
            # 工艺参数
            "90", "90", "1.2", "160", "400",
            # 目标  
            "92", "220", "88", "0.75"
        ])
        
        # 构建CSV内容（使用英文避免编码问题）
        csv_content = ",".join(headers) + "\n"
        for row in example_rows:
            csv_content += ",".join(map(str, row)) + "\n"
        
        return csv_content


def diagnose_data_types(file_path: str) -> str:
    """
    诊断CSV数据中的类型问题，帮助用户找到导致类型错误的具体数据
    """
    try:
        # 兼容对话里直接粘贴CSV内容的场景
        if ',' in file_path and '\n' in file_path and not os.path.exists(file_path):
            temp_path = f"temp_diagnose_{uuid.uuid4().hex[:8]}.csv"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(file_path)
            file_path = temp_path

        df = _read_csv_clean(file_path)
        
        diagnosis_report = {
            "problematic_columns": [],
            "mixed_type_cells": [],
            "non_numeric_in_numeric_columns": []
        }
        
        print(f"🔍 正在诊断文件: {file_path}")
        print(f"📊 数据形状: {df.shape}")
        
        for col in df.columns:
            print(f"\n📋 检查列: {col}")
            
            # 检查该列是否应该是数值列
            is_expected_numeric = any(keyword in col.lower() for keyword in 
                                    ['ratio', 'temperature', 'target', 'temp', 'conc', 'concentration'])
            
            if is_expected_numeric:
                # 尝试转换为数值
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                failed_indices = df[numeric_conversion.isna() & df[col].notna()].index.tolist()
                
                if failed_indices:
                    diagnosis_report["problematic_columns"].append(col)
                    problematic_values = []
                    
                    for idx in failed_indices[:5]:  # 只显示前5个问题值
                        problematic_values.append({
                            "row": idx + 1,  # Excel行号从1开始
                            "value": repr(df.iloc[idx][col]),
                            "type": type(df.iloc[idx][col]).__name__
                        })
                    
                    diagnosis_report["non_numeric_in_numeric_columns"].append({
                        "column": col,
                        "problematic_count": len(failed_indices),
                        "total_count": len(df),
                        "examples": problematic_values
                    })
                    
                    print(f"❌ 发现 {len(failed_indices)} 个非数值条目在数值列 '{col}' 中")
                    for example in problematic_values:
                        print(f"   行 {example['row']}: {example['value']} (类型: {example['type']})")
            
            # 检查混合类型
            unique_types = df[col].dropna().apply(type).unique()
            if len(unique_types) > 1:
                diagnosis_report["mixed_type_cells"].append({
                    "column": col,
                    "types_found": [t.__name__ for t in unique_types]
                })
                print(f"⚠️ 列 '{col}' 包含混合数据类型: {[t.__name__ for t in unique_types]}")
        
        # 生成总结报告
        if diagnosis_report["problematic_columns"]:
            return f"""
🚨 **数据类型诊断结果**

❌ **发现问题列**: {len(diagnosis_report["problematic_columns"])} 个
{chr(10).join([f"   - {col}" for col in diagnosis_report["problematic_columns"]])}

📋 **详细问题**:
{chr(10).join([f"• 列 '{item['column']}': {item['problematic_count']}/{item['total_count']} 个非数值条目" 
              for item in diagnosis_report["non_numeric_in_numeric_columns"]])}

💡 **修复建议**:
1. 检查CSV文件中上述行的数据
2. 确保比例、温度、目标值列只包含数字
3. 移除或修正非数值条目（如文本、空格、特殊字符）
4. 使用Excel或文本编辑器查看原始CSV文件

🔧 **具体检查位置**:
{chr(10).join([f"列 '{item['column']}':" + chr(10) + chr(10).join([f"   行 {ex['row']}: {ex['value']}" for ex in item['examples']]) 
              for item in diagnosis_report["non_numeric_in_numeric_columns"]])}
            """
        else:
            return "✅ 数据类型检查通过，没有发现明显的类型问题。"
            
    except Exception as e:
        return f"诊断过程中出错: {str(e)}"


def _extract_smiles_name_mapping(df: pd.DataFrame) -> dict:
    """
    从 DataFrame 中提取 SMILES -> 名称 的映射（使用原始 SMILES）
    
    Returns:
        dict: {smiles_string: friendly_name, ...}
    """
    mapping = {}
    
    # 找所有可能的分子列（*_molecule 或 *_SMILE/SMILES）
    for col in df.columns:
        is_molecule_col = col.endswith("_molecule") or 'SMILE' in col.upper()
        if not is_molecule_col:
            continue
        
        # 确定前缀以便查找对应的 name 列
        if col.endswith("_molecule"):
            prefix = col.rsplit("_molecule", 1)[0]
        elif '_SMILE' in col.upper():
            prefix = col.split('_')[0] if '_' in col else col.replace('SMILES', '').replace('SMILE', '')
        else:
            continue
        
        # 查找对应的 name 列
        name_col = None
        for candidate in [f"{prefix}_name", f"{prefix}name", prefix]:
            if candidate in df.columns and candidate != col:
                name_col = candidate
                break
        
        if name_col is None:
            continue
        
        # 提取非空的 (smiles, name) 对
        for idx, row in df.iterrows():
            smiles = row[col]
            name = row[name_col]
            
            if pd.isna(smiles) or pd.isna(name):
                continue
            
            smiles_str = str(smiles).strip()
            name_str = str(name).strip()
            
            if not smiles_str or not name_str:
                continue
            
            if smiles_str not in mapping:
                mapping[smiles_str] = name_str
    
    print(f"[DEBUG] Extracted SMILES-to-name mapping (raw): {len(mapping)} entries")
    return mapping


def _extract_smiles_name_mapping_with_canonical(df: pd.DataFrame, canonical_mapping: dict) -> dict:
    """
    从 DataFrame 中提取 SMILES -> 名称 的映射，使用规范化后的 SMILES 作为键
    
    这样可以确保 BayBE 推荐中使用的规范化 SMILES 能正确匹配到化合物名称。
    
    Args:
        df: 原始数据 DataFrame
        canonical_mapping: 原始 SMILES → 规范化 SMILES 的映射
        
    Returns:
        dict: {canonical_smiles: friendly_name, ...}
    """
    # 首先提取原始 SMILES → 名称映射
    raw_mapping = {}
    
    print(f"[DEBUG] _extract_smiles_name_mapping_with_canonical: DataFrame columns = {list(df.columns)}")
    
    for col in df.columns:
        col_upper = col.upper()
        is_molecule_col = col.endswith("_molecule") or 'SMILE' in col_upper
        if not is_molecule_col:
            continue
        
        # 确定前缀以便查找对应的 name 列
        if col.endswith("_molecule"):
            prefix = col.rsplit("_molecule", 1)[0]
        elif '_SMILE' in col_upper:
            # 处理 SubstanceA_SMILE 或 SubstanceA_SMILES 格式
            # 找到 _SMILE 的位置并截取前缀
            idx = col_upper.find('_SMILE')
            prefix = col[:idx] if idx > 0 else col.split('_')[0]
        else:
            continue
        
        # 查找对应的 name 列（支持大小写变体）
        name_col = None
        # 尝试多种可能的列名格式
        candidates = [
            f"{prefix}_name",      # SubstanceA_name
            f"{prefix}_NAME",      # SubstanceA_NAME
            f"{prefix}_Name",      # SubstanceA_Name
            f"{prefix}name",       # SubstanceAname
            f"{prefix}NAME",       # SubstanceANAME
        ]
        
        for candidate in candidates:
            if candidate in df.columns and candidate != col:
                name_col = candidate
                break
        
        # 如果还没找到，尝试不区分大小写匹配
        if name_col is None:
            for df_col in df.columns:
                if df_col.upper() == f"{prefix.upper()}_NAME" and df_col != col:
                    name_col = df_col
                    break
        
        if name_col is None:
            print(f"[DEBUG] No name column found for {col}, prefix={prefix}, tried: {candidates[:3]}")
            continue
        
        print(f"[DEBUG] Found SMILES-name pair: {col} -> {name_col}")
        
        # 提取非空的 (smiles, name) 对
        for idx, row in df.iterrows():
            smiles = row[col]
            name = row[name_col]
            
            if pd.isna(smiles) or pd.isna(name):
                continue
            
            smiles_str = str(smiles).strip()
            name_str = str(name).strip()
            
            if not smiles_str or not name_str:
                continue
            
            if smiles_str not in raw_mapping:
                raw_mapping[smiles_str] = name_str
    
    print(f"[DEBUG] Raw SMILES-to-name mapping: {len(raw_mapping)} entries")
    
    # 转换为规范化 SMILES 作为键
    canonical_name_mapping = {}
    
    for original_smiles, name in raw_mapping.items():
        # 查找规范化后的 SMILES
        canonical_smiles = canonical_mapping.get(original_smiles, original_smiles)
        
        if canonical_smiles not in canonical_name_mapping:
            canonical_name_mapping[canonical_smiles] = name
            
        # 同时保留原始 SMILES 作为键（以防规范化失败）
        if original_smiles not in canonical_name_mapping:
            canonical_name_mapping[original_smiles] = name
    
    print(f"[DEBUG] Canonical SMILES-to-name mapping: {len(canonical_name_mapping)} entries")
    
    # 打印几个示例便于调试
    for i, (k, v) in enumerate(canonical_name_mapping.items()):
        if i < 3:
            print(f"[DEBUG]   Example {i+1}: '{k[:30]}...' -> '{v}'")
    
    return canonical_name_mapping


def _resolve_file_path_or_content(file_path: str, state: dict, session_id: str) -> str:
    """
    智能处理文件路径 vs 文件内容
    
    有时 LLM 会传递 CSV 内容而不是文件路径，这个函数会：
    1. 优先使用 state 中已保存的 current_data_path
    2. 如果 file_path 是实际路径，直接使用
    3. 如果 file_path 是 CSV 内容，写入临时文件
    
    Returns:
        str: 有效的文件路径，或以 "Error:" 开头的错误消息
    """
    import os
    import uuid
    
    # 策略1: 优先使用 state 中的 current_data_path
    current_data_path = state.get("current_data_path")
    if current_data_path and os.path.exists(current_data_path):
        print(f"[DEBUG] Using current_data_path from state: {current_data_path}")
        return current_data_path
    
    # 策略2: 检查 file_path 是否是有效的文件路径
    if os.path.exists(file_path):
        print(f"[DEBUG] Using provided file_path: {file_path}")
        return file_path
    
    # 策略3: 检查 file_path 是否是 CSV 内容（包含逗号和换行符）
    if ',' in file_path and '\n' in file_path:
        print(f"[DEBUG] Detected CSV content instead of file path, writing to temp file...")
        
        # 确定保存目录
        session_dir = state.get("session_dir")
        if session_dir and os.path.exists(session_dir):
            temp_file_path = os.path.join(session_dir, f"temp_uploaded_{uuid.uuid4().hex[:8]}.csv")
        else:
            temp_file_path = f"temp_uploaded_{uuid.uuid4().hex[:8]}.csv"
        
        try:
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(file_path)
            print(f"[DEBUG] CSV content written to: {temp_file_path}")
            
            # 更新 state 中的路径
            state["current_data_path"] = temp_file_path
            
            return temp_file_path
        except Exception as e:
            return f"Error: 无法写入临时文件: {str(e)}"
    
    # 策略4: 无法识别的输入
    return f"Error: 无效的文件路径 '{file_path[:50]}...'。请提供有效的CSV文件路径或确保文件已上传。"


def enhanced_verification(file_path: str, tool_context: ToolContext) -> str:
    """
    Enhanced Verification Agent 的主要工具函数
    实现7个核心任务：
    1. 数据质量验证
    2. SMILES验证  
    3. 智能参数建议
    4. 自定义编码处理
    5. 用户交互
    6. 参数配置
    7. 数据标准化
    """
    state = tool_context.state
    session_id = state.get("session_id", "unknown")
    
    try:
        # ===== 智能处理文件路径 vs 文件内容 =====
        # 有时 LLM 会传递 CSV 内容而不是文件路径
        actual_file_path = _resolve_file_path_or_content(file_path, state, session_id)
        
        if actual_file_path.startswith("Error:"):
            return actual_file_path
        
        print(f"[DEBUG] enhanced_verification: using file_path = {actual_file_path}")
        
        # ===== 任务1: 数据质量验证 =====
        quality_report = _perform_data_quality_check(actual_file_path)
        
        if not quality_report["is_valid"]:
            return f"数据质量检查失败：\n{json.dumps(quality_report, indent=2, ensure_ascii=False)}"
        
        # ===== 任务2: SMILES验证 =====
        df = _read_csv_clean(actual_file_path)
        suspicious_headers = _detect_suspicious_headers(df)
        if suspicious_headers:
            suspicious_preview = "\n".join([f"- {h}" for h in suspicious_headers[:5]])
            _reset_verification_state(state, "header_contamination")
            state.pop("smiles_to_name_map", None)
            state.pop("original_data_format", None)
            state.pop("standardized_data_path", None)
            return (
                "数据表头疑似被说明文字污染，导致列错位/类型错误。\n"
                "检测到以下可疑列名：\n"
                f"{suspicious_preview}\n\n"
                "请使用标准模板重新导入 CSV，确保表头只包含字段名（不要包含目标描述、参数范围或约束说明）。"
            )
        smiles_validator = SimplifiedSMILESValidator()
        # 第一次验证（用于识别无效SMILES）
        smiles_validation_initial = smiles_validator.validate_smiles_data(df)

        # 在迭代开始前，尝试根据名称自动纠正无效SMILES
        auto_corrections = _auto_correct_invalid_smiles(df, smiles_validation_initial)
        if auto_corrections:
            # 记录到状态中，便于后续调试和向用户解释
            state["smiles_autocorrections"] = auto_corrections
            print(f"[DEBUG] 自动纠正SMILES完成: {len(auto_corrections)} 条")

        # 使用修正后的数据重新验证SMILES，获取最终的规范化映射
        smiles_validation = smiles_validator.validate_smiles_data(df)
        
        # ===== 保存原始表格格式（用于后续推荐表格复刻） =====
        original_column_order = list(df.columns)
        original_column_types = {col: str(df[col].dtype) for col in df.columns}
        state["original_data_format"] = {
            "column_order": original_column_order,
            "column_types": original_column_types,
            "sample_row": df.iloc[0].to_dict() if len(df) > 0 else {}
        }
        print(f"[DEBUG] Saved original data format: {len(original_column_order)} columns")
        
        # ===== 提取 SMILES → 名称映射（用于后续推荐显示） =====
        # 使用规范化后的 SMILES 作为键，因为 BayBE 推荐使用的是规范化 SMILES
        smiles_to_name_map = _extract_smiles_name_mapping_with_canonical(
            df, smiles_validation.get("canonical_smiles_mapping", {})
        )
        state["smiles_to_name_map"] = smiles_to_name_map
        print(f"[DEBUG] SMILES-to-name mapping saved to state: {len(smiles_to_name_map)} entries")
        
        # 在交互数据中记录自动纠正的SMILES（仅用于向用户说明，不影响后续计算）
        if auto_corrections:
            # 简要形式：只保留物质名和行号，避免提示过长
            try:
                from copy import deepcopy
                # 只提取关键信息
                simple_corrections = [
                    {
                        "substance": c.get("substance"),
                        "row": c.get("row"),
                        "original_smiles": c.get("original_smiles"),
                        "corrected_smiles": c.get("corrected_smiles"),
                    }
                    for c in auto_corrections
                ]
            except Exception:
                simple_corrections = auto_corrections

        # ===== 任务3: 智能参数建议 =====
        parameter_advisor = IntelligentParameterAdvisor()
        parameter_suggestions = parameter_advisor.analyze_experimental_context(df)
        
        # ===== 任务4: 用户定义编码识别 =====
        encoding_handler = UserDefinedEncodingHandler()
        
        # 智能分类所有用户列
        column_classification = encoding_handler.classify_user_columns(df)
        
        # 识别用户提供的特殊物质和编码信息
        user_special_data = encoding_handler.identify_user_special_substances(df)
        
        # 为特殊物质创建BayBE参数
        special_parameters = encoding_handler.create_baybe_parameters_for_special_substances(user_special_data, df)
        
        # 整理编码信息用于后续处理
        custom_encodings = {
            "column_classification": column_classification,
            "user_special_substances": user_special_data,
            "baybe_parameters": special_parameters,
            "encoding_strategy": "user_defined"  # 标明这是用户定义的编码
        }
        
        # ===== 任务5 & 6: 用户交互和参数配置准备 =====
        # 准备用户交互所需的信息
        user_interaction_data = _prepare_user_interaction_data(
            df, quality_report, smiles_validation, parameter_suggestions, custom_encodings
        )
        # 将自动纠正信息注入到 smiles_status 中，供提示使用
        if auto_corrections:
            user_interaction_data.setdefault("smiles_status", {})
            user_interaction_data["smiles_status"]["autocorrections"] = simple_corrections
        
        # ===== 任务7: 数据标准化 =====
        standardized_data = _standardize_data(df, smiles_validation)
        
        # ===== 初始化统一的实验记录表 =====
        session_dir = state.get("session_dir", ".")
        unified_experiment_log = os.path.join(session_dir, "experiment_log.csv")
        
        # 如果统一记录表不存在，使用原始数据作为初始记录
        if not os.path.exists(unified_experiment_log):
            # 添加轮次标记列
            df_with_round = df.copy()
            df_with_round["optimization_round"] = 0
            df_with_round["experiment_status"] = "completed"  # 初始数据都是已完成的
            df_with_round.to_csv(unified_experiment_log, index=False, encoding="utf-8-sig")
            print(f"[DEBUG] Created unified experiment log: {unified_experiment_log}")
        else:
            print(f"[DEBUG] Unified experiment log already exists: {unified_experiment_log}")
        
        state["unified_experiment_log_path"] = unified_experiment_log
        
        # 保存状态信息
        state["verification_results"] = {
            "quality_report": quality_report,
            "smiles_validation": smiles_validation,
            "parameter_suggestions": parameter_suggestions,
            "custom_encodings": custom_encodings,
            "standardized_data_path": f"standardized_data_{session_id}.csv",
            "ready_for_user_interaction": True
        }
        
        # 保存标准化数据
        output_path = f"standardized_data_{session_id}.csv"
        standardized_data.to_csv(output_path, index=False)
        
        # 生成用户交互提示
        return _generate_user_interaction_prompt(user_interaction_data)
        
    except Exception as e:
        return f"Enhanced Verification 处理错误: {str(e)}"


def _perform_data_quality_check(file_path: str) -> dict:
    """
    执行数据质量检查（任务1）
    """
    try:
        df = _read_csv_clean(file_path)
        
        quality_report = {
            "is_valid": True,
            "issues": [],
            "statistics": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "missing_percentage": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
            }
        }
        
        # 检查缺失值
        missing_cols = df.isnull().sum()
        high_missing_cols = missing_cols[missing_cols > len(df) * 0.5].index.tolist()
        if high_missing_cols:
            quality_report["issues"].append({
                "type": "high_missing_data",
                "columns": high_missing_cols,
                "severity": "warning"
            })
        
        # 检查必需列
        required_patterns = ['Substance', 'SMILE', 'Target_']
        for pattern in required_patterns:
            matching_cols = [col for col in df.columns if pattern in str(col)]
            if not matching_cols:
                quality_report["issues"].append({
                    "type": "missing_required_columns",
                    "pattern": pattern,
                    "severity": "error"
                })
                quality_report["is_valid"] = False
        
        # 检查数值列的异常值
        for col in df.columns:
            # 尝试将列转换为数值类型
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            # 只处理至少有一些有效数值的列
            if numeric_data.notna().sum() < len(df) * 0.1:  # 如果有效数值少于10%，跳过
                continue
                
            # 使用清理后的数值数据计算统计量
            clean_data = numeric_data.dropna()
            if len(clean_data) < 2:  # 需要至少2个值来计算IQR
                continue
                
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = clean_data[(clean_data < Q1 - 1.5*IQR) | (clean_data > Q3 + 1.5*IQR)]
            
            if not outliers.empty:
                quality_report["issues"].append({
                    "type": "outliers_detected",
                    "column": col,
                    "count": int(len(outliers)),
                    "severity": "info"
                })
        
        return quality_report
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "issues": [{"type": "file_read_error", "severity": "error"}]
        }


def _prepare_user_interaction_data(df, quality_report, smiles_validation, parameter_suggestions, custom_encodings):
    """
    准备用户交互所需的数据（任务5支持）
    """
    # 识别目标变量
    target_columns = [col for col in df.columns if col.startswith('Target_')]
    
    # 识别可调变量
    adjustable_vars = []
    ratio_cols = [col for col in df.columns if 'ratio' in col.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    adjustable_vars = list(set(ratio_cols + numeric_cols) - set(target_columns))
    
    interaction_data = {
        "data_summary": {
            "total_experiments": len(df),
            "substances_count": len([col for col in df.columns if 'Substance' in col and 'name' in col]),
            "targets_count": len(target_columns),
            "adjustable_variables_count": len(adjustable_vars)
        },
        "target_variables": target_columns,
        "adjustable_variables": adjustable_vars,
        "parameter_suggestions": parameter_suggestions,
        "smiles_status": {
            "valid_smiles": len(smiles_validation["canonical_smiles_mapping"]),
            "invalid_smiles": len(smiles_validation["invalid_smiles"]),
            "invalid_smiles_details": smiles_validation["invalid_smiles"],  # 添加详细的无效SMILES信息
            "substances_validated": smiles_validation["substances_validated"],
            # 这里先占位，具体内容在 enhanced_verification 中根据 state['smiles_autocorrections'] 填充
            "autocorrections": []
        },
        "special_molecules": custom_encodings,
        "quality_score": 100 - quality_report["statistics"]["missing_percentage"]
    }
    
    return interaction_data


def _standardize_data(df: pd.DataFrame, smiles_validation: dict) -> pd.DataFrame:
    """
    数据标准化处理（任务7）
    """
    standardized_df = df.copy()
    
    # 1. 替换为规范化SMILES
    smiles_columns = [col for col in df.columns if 'SMILE' in col.upper()]
    for col in smiles_columns:
        for original_smiles, canonical_smiles in smiles_validation["canonical_smiles_mapping"].items():
            standardized_df[col] = standardized_df[col].replace(original_smiles, canonical_smiles)
    
    # 2. 安全的数据类型标准化和缺失值处理
    for col in standardized_df.columns:
        if col not in smiles_columns:  # 跳过SMILES列
            # 尝试转换为数值类型
            numeric_data = pd.to_numeric(standardized_df[col], errors='coerce')
            valid_numeric_ratio = numeric_data.notna().sum() / len(standardized_df)
            
            # 如果至少50%的数据可以转换为数值，则认为这是数值列
            if valid_numeric_ratio >= 0.5:
                standardized_df[col] = numeric_data
                # 用中位数填充缺失值
                median_val = numeric_data.median()
                if not pd.isna(median_val):
                    standardized_df[col] = standardized_df[col].fillna(median_val)
                    
    # 3. 特定列类型强制转换
    for col in standardized_df.columns:
        if 'ratio' in col.lower() or 'temperature' in col.lower() or 'target' in col.lower():
            # 强制转换为数值，无效值设为NaN
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        elif col.startswith('Target_'):
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
    
    return standardized_df


def _generate_user_interaction_prompt(interaction_data: dict) -> str:
    """
    生成用户交互提示（任务5）
    """
    prompt = f"""
🔍 **数据验证完成 - 需要您的优化目标确认**

📊 **数据概览**:
- 实验数量: {interaction_data['data_summary']['total_experiments']}
- 物质种类: {interaction_data['data_summary']['substances_count']}
- 目标变量: {interaction_data['data_summary']['targets_count']}
- 可调变量: {interaction_data['data_summary']['adjustable_variables_count']}
- 数据质量评分: {interaction_data['quality_score']:.1f}/100

🎯 **目标变量**: {', '.join(interaction_data['target_variables'])}

🔧 **可调变量**: {', '.join(interaction_data['adjustable_variables'])}

🧪 **SMILES验证状态**:
- 有效分子: {interaction_data['smiles_status']['valid_smiles']}
- 无效分子: {interaction_data['smiles_status']['invalid_smiles']}
- 已验证物质: {', '.join(interaction_data['smiles_status']['substances_validated'])}
"""
    # 如果有无效SMILES，显示详细信息
    invalid_smiles_details = interaction_data.get('smiles_status', {}).get('invalid_smiles_details', [])
    if invalid_smiles_details:
        prompt += "\n⚠️ **无效SMILES详情**:\n"
        # 显示所有无效SMILES的详细信息
        for i, invalid_item in enumerate(invalid_smiles_details, 1):
            if isinstance(invalid_item, dict):
                substance = invalid_item.get('substance', 'Unknown')
                row = invalid_item.get('row', 'N/A')
                smiles = invalid_item.get('smiles', 'N/A')
                error = invalid_item.get('error', '未知错误')
                prompt += f"  {i}. 物质: {substance}, 行: {row}, SMILES: `{smiles}`, 错误: {error}\n"
            else:
                # 兼容旧格式（字符串）
                prompt += f"  {i}. {invalid_item}\n"
        prompt += "\n💡 **建议**: 请检查并修正这些无效的SMILES字符串，然后重新上传数据。\n"
    
    prompt += """

"""
    # 如果有自动纠正的SMILES，向用户明确说明（但不要求用户手动修改）
    autocorrections = interaction_data.get('smiles_status', {}).get('autocorrections')
    if autocorrections:
        prompt += "\n🧠 **SMILES自动纠正说明**:\n"
        prompt += f"- 系统已根据化合物名称自动纠正 {len(autocorrections)} 条SMILES，避免因小的录入错误中断优化流程。\n"
        # 只展示前几条示例，避免信息过长
        max_examples = 3
        for i, c in enumerate(autocorrections[:max_examples], 1):
            substance = c.get('substance', 'Unknown')
            row = c.get('row', 'N/A')
            orig = c.get('original_smiles', '')
            corrected = c.get('corrected_smiles', '')
            prompt += f"  - 物质 {substance}，行 {row}: `{orig}` → `{corrected}`\n"
        if len(autocorrections) > max_examples:
            prompt += f"  - 其余 {len(autocorrections) - max_examples} 条已同样自动纠正。\n"
        prompt += "  - 您无需在CSV中手动修改这些位置，如需查看完整列表可查看系统日志或联系开发者。\n"

    prompt += "\n💡 **智能参数建议**："
    
    # 添加参数建议详情（跳过以_开头的特殊键如_reaction_info）
    for param, suggestion in interaction_data['parameter_suggestions'].items():
        # 跳过元信息键
        if param.startswith('_'):
            continue
        # 确保suggestion是字典且包含必要的键
        if not isinstance(suggestion, dict):
            continue
        if 'current_range' not in suggestion:
            continue
            
        prompt += f"\n📌 **{param}**:"
        prompt += f"\n   - 当前范围: {suggestion.get('current_range', 'N/A')}"
        
        # 检查是否为离散参数
        if suggestion.get('type') == 'NumericalDiscreteParameter':
            discrete_values = suggestion.get('values', [])
            prompt += f"\n   - 参数类型: 离散参数 (NumericalDiscreteParameter)"
            prompt += f"\n   - 离散值: {discrete_values}"
            prompt += f"\n   - 检测理由: {suggestion.get('discrete_reasoning', '自动检测')}"
            
            # 说明这些值来自实际数据
            if suggestion.get('data_based', False):
                unique_count = suggestion.get('unique_count', len(discrete_values))
                total_count = suggestion.get('total_count', 'N/A')
                prompt += f"\n   - 数据来源: 从实际数据中检测到 {unique_count} 个唯一值（共 {total_count} 行数据）"
                prompt += f"\n   - 注意: 这些离散值是基于当前数据自动检测的，如果数据量较少，建议值可能不完整"
        else:
            prompt += f"\n   - 参数类型: 连续参数 (NumericalContinuousParameter)"
            prompt += f"\n   - 建议范围: {suggestion.get('suggested_bounds', 'N/A')}"
            prompt += f"\n   - 理由: {suggestion.get('reasoning', 'N/A')}"
    
    # 添加智能列分类结果
    if interaction_data['special_molecules'].get('column_classification'):
        classification = interaction_data['special_molecules']['column_classification']
        
        prompt += f"\n\n📋 **数据结构分析**:"
        
        # 显示识别的扩展列
        if classification['识别的扩展列']:
            prompt += f"\n🎯 **智能识别的扩展列类型**:"
            for category, columns in classification['识别的扩展列'].items():
                if columns:
                    prompt += f"\n   📌 {category}:"
                    for col_info in columns:
                        confidence_str = f"({col_info['confidence']:.1%}置信度)" if col_info['confidence'] > 0 else ""
                        prompt += f"\n      - {col_info['column_name']}: {col_info['actual_data_type']} {confidence_str}"
                        prompt += f"\n        样本值: {col_info['sample_values'][:3]}"
        
        # 显示特殊物质
        if interaction_data['special_molecules'].get('user_special_substances', {}).get('substances_without_smiles'):
            user_special = interaction_data['special_molecules']['user_special_substances']
            prompt += f"\n\n🔬 **识别到您的特殊物质**:"
            special_substances_summary = {}
            for substance in user_special['substances_without_smiles']:
                name = substance['name']
                if name not in special_substances_summary:
                    special_substances_summary[name] = []
                special_substances_summary[name].append(substance['row'])
            
            for name, rows in special_substances_summary.items():
                prompt += f"\n   - {name}: 出现在 {len(rows)} 个实验中，无SMILES，将使用名称编码"
        
        # 显示未分类列建议
        if classification['未分类列']:
            prompt += f"\n\n❓ **未分类的列** (可能需要您的说明):"
            for col in classification['未分类列']:
                prompt += f"\n   - {col}"
        
        # 显示标准格式建议
        if classification['建议的标准格式']['当前数据映射建议']:
            prompt += f"\n\n💡 **数据格式优化建议**:"
            for current_col, suggestion in classification['建议的标准格式']['当前数据映射建议'].items():
                prompt += f"\n   - {suggestion}"
    
    prompt += f"""

❓ **请回答以下问题以完成优化配置**:

1. **优化目标确认**: 
   对于每个目标变量，请指定：
   - 优化方向 (最大化/最小化/目标值匹配)
   - 期望的目标值范围

2. **多目标优化策略** (如果有多个目标):
   💡 **期望度方法 (desirability)**: 
      - 适用于您知道各目标的相对重要性
      - 需要指定各目标的权重（如50%:50%）
      - 返回单一最优解
   
   💡 **帕累托方法 (pareto)**: 
      - 适用于目标相互冲突（如强度vs延展性）
      - 不需要指定权重
      - 返回帕累托前沿上的多个方案供您选择

3. **参数边界确认**:
   是否接受上述智能建议的参数范围？如需调整请说明。

4. **约束条件**:
   是否有特殊的约束条件？系统支持以下约束类型：
   
   ✅ **支持的约束类型**:
   - **累加等于**: `SubstanceA_ratio + SubstanceB_ratio = 1.0` (类型: `sum_equals_one`)
   - **累加大于**: `SubstanceA_ratio + SubstanceB_ratio >= 1.0` (类型: `sum_greater_than`)
   - **累加小于**: `SubstanceA_ratio + SubstanceB_ratio <= 0.8` (类型: `sum_less_than`)
   - **线性等式**: `a1*x1 + a2*x2 = b` (类型: `linear_equality`)
   - **线性不等式**: `a1*x1 + a2*x2 >= b` 或 `<= b` (类型: `linear_inequality`)
   
   💡 **示例**:
   - "SubstanceA_ratio + SubstanceB_ratio 必须大于 1.0"
   - "所有比例之和必须等于 1.0"
   - "SubstanceA_ratio + SubstanceB_ratio <= 0.8"

5. **获取函数（可选）**:
   - 默认使用 BayBE 内置策略
   - 可选项示例: `qEI`, `qUCB`, `qNEI`, `qPI`
   - 如不确定可回答 “默认”

6. **比例和为1的自动约束（可选）**:
   - 是否启用自动“比例之和 = 1.0”的约束
   - 默认启用；如不需要可回答“关闭”

7. **实验设计参数**:
   - 计划的实验批次大小 (batch_size)
   - 最大实验轮数 (max_iterations)
   - 预算约束 (总实验数量限制)

请提供您的回答，我将根据您的需求生成优化配置。

**示例回答**:
"最大化Target_alpha_tg和Target_gamma_elongation，使用帕累托方法因为这两个目标可能冲突。接受建议的参数范围。没有其他约束。每批10组实验，最多20轮，总共200次实验。"
"""
    
    return prompt


# 主要的增强验证工具函数
def collect_optimization_goals(
    targets: str,
    batch_size: int,
    max_iterations: int,
    total_budget: int,
    accept_suggested_parameters: bool,
    tool_context: ToolContext,
    optimization_strategy: str = "desirability",
    constraints: str = "[]",
    custom_parameter_bounds: str = "{}",
    acquisition_function: str = "default",
    auto_ratio_sum_constraint: bool = True
) -> str:
    """
    收集用户的优化目标和配置（任务5和6）
    
    LLM负责理解用户的自然语言输入并提取结构化参数，此工具只接收结构化数据。
    
    Args:
        targets: JSON格式的目标列表，例如：
                 '[{"name": "Target_alpha_tg", "mode": "MAX", "weight": 0.5, "bounds": [0, 100]}, 
                   {"name": "Target_gamma_elongation", "mode": "MAX", "weight": 0.5, "bounds": [0, 100]}]'
                 mode可选值: "MAX"(最大化), "MIN"(最小化), "MATCH"(目标值匹配)
                 weight: 0-1之间的权重值（仅desirability策略需要）
                 bounds: 目标值的边界范围（desirability策略必需）
        
        batch_size: 每批实验的数量（如用户说"同时开展10组实验"则为10）
        
        max_iterations: 最大迭代轮数（如用户说"最大20轮"则为20）
        
        total_budget: 总实验次数预算（如用户说"总共200次实验"则为200）
        
        accept_suggested_parameters: 用户是否接受系统建议的参数范围（True/False）
        
        optimization_strategy: 多目标优化策略，可选值：
            - "desirability": 期望度方法 - 使用权重将多目标合并为单一标量
                             适用于用户明确知道各目标相对重要性的情况
                             需要指定每个目标的weight和bounds
            - "pareto": 帕累托方法 - 探索帕累托前沿，返回所有非支配解
                       适用于目标相互冲突、用户想看所有权衡方案的情况
                       不需要指定权重，推荐结果会分布在帕累托前沿上
            默认为 "desirability"
        
        constraints: JSON格式的约束条件列表，支持的约束类型：
                    
                    **1. sum_equals_one**: 累加等于约束
                       '[{"type": "sum_equals_one", "parameters": ["SubstanceA_ratio", "SubstanceB_ratio"]}]'
                       表示: SubstanceA_ratio + SubstanceB_ratio = 1.0
                    
                    **2. sum_greater_than**: 累加大于等于约束
                       '[{"type": "sum_greater_than", "parameters": ["SubstanceA_ratio", "SubstanceB_ratio"], "threshold": 1.0}]'
                       表示: SubstanceA_ratio + SubstanceB_ratio >= 1.0
                    
                    **3. sum_less_than**: 累加小于等于约束
                       '[{"type": "sum_less_than", "parameters": ["SubstanceA_ratio", "SubstanceB_ratio"], "threshold": 0.8}]'
                       表示: SubstanceA_ratio + SubstanceB_ratio <= 0.8
                    
                    **4. linear_equality**: 线性等式约束
                       '[{"type": "linear_equality", "parameters": ["x1", "x2"], "coefficients": [1.0, 2.0], "rhs": 1.0}]'
                       表示: 1.0*x1 + 2.0*x2 = 1.0
                    
                    **5. linear_inequality**: 线性不等式约束
                       '[{"type": "linear_inequality", "parameters": ["x1", "x2"], "coefficients": [1.0, 1.0], "rhs": 1.0, "operator": ">="}]'
                       表示: 1.0*x1 + 1.0*x2 >= 1.0 (operator 可选: ">=", "<=")
                    
                    如果用户说"没有约束"，则为空列表"[]"
        
        custom_parameter_bounds: JSON格式的自定义参数边界，例如：
                                '{"SubstanceA_ratio": {"min": 0.5, "max": 0.9}}'
                                如果用户接受建议的参数范围，则为空对象"{}"

        acquisition_function: 获取函数偏好，可选值:
            - "default" (使用 BayBE 默认策略)
            - "qEI" / "qUCB" / "qNEI" / "qPI"

        auto_ratio_sum_constraint: 是否启用自动“比例之和=1.0”约束（默认 True）
        
        tool_context: ADK工具上下文
    
    Returns:
        配置完成的确认信息
    """
    state = tool_context.state
    verification_results = state.get("verification_results", {})
    
    # 调试信息
    print(f"\n[DEBUG] collect_optimization_goals state:")
    print(f"   verification_results exists: {bool(verification_results)}")
    if verification_results and isinstance(verification_results, dict):
        print(f"   verification_results keys: {list(verification_results.keys())}")
    
    try:
        # 解析JSON参数
        targets_list = json.loads(targets) if isinstance(targets, str) else targets
        constraints_list = json.loads(constraints) if isinstance(constraints, str) else constraints
        custom_bounds = json.loads(custom_parameter_bounds) if isinstance(custom_parameter_bounds, str) else custom_parameter_bounds
        
        # 调试：打印边界信息
        print(f"[DEBUG] collect_optimization_goals: custom_parameter_bounds (原始): {custom_parameter_bounds}")
        print(f"[DEBUG] collect_optimization_goals: custom_bounds (解析后): {custom_bounds}")
        print(f"[DEBUG] collect_optimization_goals: custom_bounds 类型: {type(custom_bounds)}")
        if isinstance(custom_bounds, dict):
            print(f"[DEBUG] collect_optimization_goals: custom_bounds keys: {list(custom_bounds.keys())}")
            for key, value in custom_bounds.items():
                print(f"[DEBUG] collect_optimization_goals:   {key}: {value} (type: {type(value)})")
        
        # 验证优化策略
        valid_strategies = ["desirability", "pareto"]
        if optimization_strategy not in valid_strategies:
            optimization_strategy = "desirability"  # 默认使用期望度方法
        
        # 验证目标列表
        if not targets_list:
            return """
❌ **配置错误**: 未提供优化目标

请告诉我您要优化的目标变量，例如：
- "我想最大化 Target_alpha_tg 和 Target_gamma_elongation"
- "最小化 Target_cost"

可用的目标变量请查看上方的数据验证结果。

💡 **优化策略选择**:
- 如果您知道各目标的相对重要性，使用 **desirability**（期望度）方法并指定权重
- 如果目标相互冲突且您想看所有可能的权衡方案，使用 **pareto**（帕累托）方法
"""
        
        # 根据策略处理目标
        if optimization_strategy == "desirability":
            # 期望度方法：需要权重和边界
            # 验证并规范化目标权重
            total_weight = sum(t.get("weight", 0) for t in targets_list)
            if abs(total_weight - 1.0) > 0.01:
                # 自动归一化权重
                for t in targets_list:
                    if total_weight > 0:
                        t["weight"] = t.get("weight", 1.0 / len(targets_list)) / total_weight
                    else:
                        t["weight"] = 1.0 / len(targets_list)
            
            # 确保每个目标有bounds（desirability方法必需）
            for t in targets_list:
                if "bounds" not in t or t["bounds"] is None:
                    # 使用默认边界
                    t["bounds"] = [0, 100]
                    
        elif optimization_strategy == "pareto":
            # 帕累托方法：不需要权重，探索帕累托前沿
            # 移除权重信息（帕累托不使用）
            for t in targets_list:
                t.pop("weight", None)
        
        # 构建优化配置
        optimization_config = {
            "targets": targets_list,
            "optimization_strategy": optimization_strategy,
            "parameters": verification_results.get("parameter_suggestions", {}),
            "constraints": constraints_list,
            "experimental_settings": {
                "batch_size": batch_size,
                "max_iterations": max_iterations,
                "total_budget": total_budget
            },
            "accept_suggested_parameters": accept_suggested_parameters,
            "custom_parameter_bounds": custom_bounds,
            "acquisition_function": acquisition_function,
            "auto_ratio_sum_constraint": auto_ratio_sum_constraint
        }
        
        # 生成BayBE兼容的配置
        baybe_config = _generate_baybe_config(optimization_config, verification_results)
        
        # 更新状态
        state["optimization_config"] = optimization_config
        state["baybe_campaign_config"] = baybe_config
        state["verification_status"] = "completed_with_user_input"
        state["ready_for_optimization"] = True
        
        # 构建详细的目标信息显示（根据策略不同显示不同内容）
        if optimization_strategy == "desirability":
            strategy_info = "🎯 **优化策略**: DesirabilityObjective（期望度方法）\n"
            strategy_info += "   - 使用权重将多目标合并为单一标量进行优化\n"
            strategy_info += "   - 适合已知各目标相对重要性的情况\n\n"
            
            targets_summary = "🎯 **优化目标详情**:\n"
            for i, target in enumerate(targets_list, 1):
                mode = target.get('mode', 'MAX')
                mode_str = "最大化" if mode == 'MAX' else ("最小化" if mode == 'MIN' else "目标值匹配")
                weight_pct = target.get('weight', 0) * 100
                bounds = target.get('bounds', [0, 100])
                targets_summary += f"   {i}. {target.get('name')}: {mode_str}\n"
                targets_summary += f"      权重: {weight_pct:.1f}%, 边界: {bounds}\n"
        else:  # pareto
            strategy_info = "🎯 **优化策略**: ParetoObjective（帕累托方法）\n"
            strategy_info += "   - 探索帕累托前沿，返回所有非支配解\n"
            strategy_info += "   - 适合目标相互冲突、想看所有权衡方案的情况\n"
            strategy_info += "   - 推荐结果会分布在帕累托前沿上，供您选择\n\n"
            
            targets_summary = "🎯 **优化目标详情**:\n"
            for i, target in enumerate(targets_list, 1):
                mode = target.get('mode', 'MAX')
                mode_str = "最大化" if mode == 'MAX' else ("最小化" if mode == 'MIN' else "目标值匹配")
                targets_summary += f"   {i}. {target.get('name')}: {mode_str}\n"
        
        # 约束条件显示
        constraints_summary = ""
        if constraints_list:
            constraints_summary = "\n📏 **约束条件**:\n"
            for i, constraint in enumerate(constraints_list, 1):
                constraints_summary += f"   {i}. {constraint.get('type', '未知类型')}: {constraint.get('description', str(constraint))}\n"
        else:
            constraints_summary = "\n📏 **约束条件**: 无特殊约束\n"
        
        # 参数边界显示
        params_summary = "\n📐 **参数边界**: "
        if accept_suggested_parameters:
            params_summary += "使用系统建议的参数范围\n"
        else:
            params_summary += "使用用户自定义范围\n"
            if custom_bounds:
                for param, bounds in custom_bounds.items():
                    params_summary += f"   - {param}: [{bounds.get('min', '?')}, {bounds.get('max', '?')}]\n"
        
        return f"""
✅ **优化配置已完成**

📋 **配置摘要**:
- 目标数量: {len(targets_list)}
- 参数数量: {len(optimization_config.get('parameters', {}))}
- 约束条件: {len(constraints_list)}
- 特殊编码: {len(verification_results.get('custom_encodings', {}))}

{strategy_info}{targets_summary}{constraints_summary}{params_summary}
⚙️ **实验设置**:
- 批次大小 (batch_size): {batch_size}
- 最大轮数 (max_iterations): {max_iterations}
- 总实验预算: {total_budget}

🚀 **下一步**: 系统将构建BayBE搜索空间并准备优化Campaign。

📄 **BayBE配置已保存到会话状态**，可以传递给 Recommender Agent。
        """
        
    except json.JSONDecodeError as e:
        return f"""
❌ **JSON解析错误**: {str(e)}

请确保目标和约束条件使用正确的JSON格式。

目标格式示例:
[{{"name": "Target_alpha_tg", "mode": "MAX", "weight": 0.5}}]

约束格式示例:
[{{"type": "sum_equals_one", "parameters": ["ratio_A", "ratio_B"]}}]
"""
    except Exception as e:
        import traceback
        return f"配置处理出错: {str(e)}\n{traceback.format_exc()}\n请重新提供配置信息。"


def _generate_baybe_config(optimization_config: dict, verification_results: dict) -> dict:
    """
    生成BayBE兼容的配置格式（任务6）
    
    支持两种多目标优化策略：
    1. DesirabilityObjective - 期望度方法，使用权重合并多目标
    2. ParetoObjective - 帕累托方法，探索帕累托前沿
    """
    if not BAYBE_AVAILABLE:
        return {"error": "BayBE not available"}
    
    optimization_strategy = optimization_config.get("optimization_strategy", "desirability")
    targets = optimization_config.get("targets", [])
    
    # 构建目标配置
    target_configs = []
    for target in targets:
        target_config = {
            "name": target.get("name"),
            "mode": target.get("mode", "MAX"),
        }
        # desirability 方法需要 bounds 和 transformation
        if optimization_strategy == "desirability":
            bounds = target.get("bounds", [0, 100])
            target_config["bounds"] = bounds
            # 根据模式选择转换函数
            if target.get("mode") == "MATCH":
                target_config["transformation"] = "BELL"  # 或 "TRIANGULAR"
            else:
                target_config["transformation"] = "LINEAR"
        target_configs.append(target_config)
    
    # 根据策略构建objective配置
    if optimization_strategy == "pareto":
        objective_config = {
            "type": "ParetoObjective",
            "description": "探索帕累托前沿，返回所有非支配解",
            "note": "推荐结果会分布在帕累托前沿上，适合目标相互冲突的情况"
        }
    else:  # desirability
        # 提取权重
        weights = [t.get("weight", 1.0 / len(targets)) for t in targets]
        objective_config = {
            "type": "DesirabilityObjective",
            "weights": weights,
            "scalarizer": "GEOM_MEAN",  # 几何平均，对极端值更敏感
            "description": "使用权重将多目标合并为单一标量"
        }
    
    # 根据开发文档的标准格式生成配置
    baybe_config = {
        "campaign_info": {
            "name": "chemical_optimization",
            "created_at": datetime.now().isoformat(),
            "description": "ChemBoMAS Enhanced Verification Agent generated configuration",
            "optimization_strategy": optimization_strategy
        },
        "targets": target_configs,
        "parameters": [],  # 由 Recommender Agent 填充
        "constraints": optimization_config.get("constraints", []),
        "objective_config": objective_config,
        "experimental_config": {
            "batch_size": optimization_config["experimental_settings"]["batch_size"],
            "max_iterations": optimization_config["experimental_settings"]["max_iterations"],
            "total_budget": optimization_config["experimental_settings"]["total_budget"],
            "recommender": "TwoPhaseMetaRecommender"
        }
    }
    
    return baybe_config


# 测试代码
if __name__ == "__main__":
    print("🧪 Enhanced Verification Tools 功能测试")
    print("=" * 50)
    
    # 测试1: SMILES验证器
    print("\n1. 测试SMILES验证器...")
    validator = SimplifiedSMILESValidator()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'SubstanceA_SMILE': ['CCO', 'CCCCO', 'invalid_smiles', ''],
        'SubstanceB_SMILE': ['CC(C)O', 'CCCCCO', 'CCC', 'another_invalid'],
        'SubstanceA_ratio': [0.5, 0.6, 0.7, 0.8],
        'Target_alpha_tg': [80, 85, 90, 95]
    })
    
    validation_results = validator.validate_smiles_data(test_data)
    print(f"   有效SMILES: {len(validation_results['canonical_smiles_mapping'])}")
    print(f"   无效SMILES: {len(validation_results['invalid_smiles'])}")
    print(f"   验证的物质: {validation_results['substances_validated']}")
    
    # 测试2: 参数建议器（基于知识库）
    print("\n2. 测试参数建议器（已整合化学知识库）...")
    advisor = IntelligentParameterAdvisor()
    suggestions = advisor.analyze_experimental_context(test_data, "环氧树脂固化实验")
    
    # 显示识别的反应类型
    reaction_info = suggestions.get("_reaction_info", {})
    print(f"   识别的反应类型: {reaction_info.get('reaction_name', '未知')}")
    print(f"   安全警告数量: {len(reaction_info.get('safety_warnings', []))}")
    
    # 显示参数建议
    param_count = len([k for k in suggestions.keys() if not k.startswith('_')])
    print(f"   参数建议数量: {param_count}")
    for param, suggestion in suggestions.items():
        if not param.startswith('_'):  # 跳过元信息
            print(f"   {param}:")
            print(f"      当前范围: {suggestion.get('current_range')}")
            # 检查是否为离散参数
            if suggestion.get('type') == 'NumericalDiscreteParameter':
                print(f"      参数类型: 离散参数")
                print(f"      离散值: {suggestion.get('values', [])}")
                print(f"      检测理由: {suggestion.get('discrete_reasoning', '自动检测')}")
            else:
                print(f"      参数类型: 连续参数")
                print(f"      建议范围: {suggestion.get('suggested_bounds')}")
                print(f"      理由: {suggestion.get('reasoning', 'N/A')}")
            print(f"      来源: {suggestion.get('source', 'unknown')}")
    
    # 测试3: 用户定义编码处理器
    print("\n3. 测试用户定义编码处理器...")
    encoder = UserDefinedEncodingHandler()
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'SubstanceA_name': ['树脂A', '树脂B'],
        'SubstanceA_SMILES': ['CCO', 'CCCCO'], 
        'SubstanceB_name': ['稀释剂A', '稀释剂B'],
        'SubstanceB_SMILES': ['', ''],  # 特殊物质
        'SubstanceA_density': [1.15, 1.18],  # 物理性质
        'Process_temperature': [80, 90]  # 工艺参数
    })
    
    user_special_data = encoder.identify_user_special_substances(test_df)
    classification = encoder.classify_user_columns(test_df)
    
    print(f"   识别到特殊物质: {len(user_special_data['substances_without_smiles'])} 个")
    print(f"   识别到扩展列类型: {len(classification['识别的扩展列'])} 种")
    
    # 测试4: BayBE可用性
    print("\n4. 测试BayBE可用性...")
    if BAYBE_AVAILABLE:
        print("   ✅ BayBE已安装，可以使用完整功能")
        
        # 测试BayBE参数创建
        try:
            parameters = validator.prepare_baybe_parameters(test_data, validation_results)
            print(f"   ✅ 成功创建 {len(parameters)} 个BayBE参数")
        except Exception as e:
            print(f"   ❌ BayBE参数创建失败: {e}")
    else:
        print("   ⚠️ BayBE未安装，使用降级模式")
        print("   建议运行: pip install baybe")
    
    print("\n" + "=" * 50)
    print("📊 Enhanced Verification Tools 测试完成")
    
    if BAYBE_AVAILABLE:
        print("🎉 所有功能可用！系统已准备好进行完整的BayBE集成")
    else:
        print("🔧 核心功能可用！安装BayBE后即可使用完整功能")
        print("   运行: pip install baybe")
