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

"""化学反应和材料知识库 - 为智能参数建议提供化学专业知识"""

import pandas as pd
from typing import Dict, List, Tuple, Optional


class ChemistryKnowledgeBase:
    """
    化学领域知识库，提供反应类型、材料属性和安全约束的专业知识
    """
    
    # 反应类型知识库
    REACTION_TYPES = {
        "epoxy_curing": {
            "name": "环氧固化反应",
            "description": "环氧树脂与固化剂的交联固化反应",
            "typical_temperature": (60, 120),  # °C
            "temperature_optimal": (80, 100),
            "catalyst_concentration": (0.01, 0.1),  # 质量分数
            "curing_time_range": (30, 180),  # 分钟
            "common_hardeners": [
                "IPDA",  # 异佛尔酮二胺
                "DICY",  # 双氰胺
                "Amine",  # 胺类固化剂
                "Anhydride"  # 酸酐类
            ],
            "incompatible_combinations": [
                ("strong_acid_catalyst", "amine_hardener"),
                ("moisture_sensitive", "high_humidity_condition")
            ],
            "safety_warnings": [
                "避免高温暴聚（温度梯度控制在5°C/min以内）",
                "确保充分混合（搅拌时间≥5分钟）",
                "控制放热速率（监测温度上升）",
                "使用个人防护装备（手套、护目镜）"
            ],
            "quality_indicators": {
                "glass_transition_temp": (50, 180),  # Tg, °C
                "impact_strength": (50, 200),  # kJ/m²
                "tensile_strength": (30, 120)  # MPa
            }
        },
        
        "polymerization": {
            "name": "聚合反应",
            "description": "单体聚合形成高分子化合物",
            "typical_temperature": (40, 100),
            "initiator_concentration": (0.001, 0.05),
            "reaction_time_range": (60, 480),
            "safety_warnings": [
                "需要惰性气氛保护（氮气或氩气）",
                "严格控制温度避免暴聚",
                "监测反应放热",
                "使用阻聚剂预防意外聚合"
            ]
        },
        
        "catalytic_synthesis": {
            "name": "催化合成",
            "description": "催化剂参与的有机合成反应",
            "catalyst_loading": (0.001, 0.1),
            "typical_temperature": (25, 150),
            "pressure_range": (1, 10),  # bar
            "common_catalysts": ["Pd", "Pt", "Ru", "Ni"],
            "safety_warnings": [
                "贵金属催化剂昂贵，需精确称量",
                "某些催化剂对空气和水敏感",
                "反应可能产生氢气，注意通风"
            ]
        }
    }
    
    # 材料属性知识库
    MATERIAL_PROPERTIES = {
        "epoxy_resins": {
            "name": "环氧树脂",
            "typical_viscosity": (800, 15000),  # mPa·s @ 25°C
            "glass_transition_temp": (50, 180),  # °C
            "density": (1.0, 1.3),  # g/cm³
            "epoxy_value": (0.4, 0.6),  # mol/100g
            "common_types": {
                "DGEBA": "双酚A型环氧树脂（通用型）",
                "TGDDM": "四甘油二胺型环氧树脂（高性能）",
                "Novolac": "酚醛型环氧树脂（耐高温）"
            }
        },
        
        "hardeners": {
            "name": "固化剂",
            "amine_hardeners": {
                "viscosity_range": (10, 1000),
                "equivalent_weight": (30, 200),
                "mixing_ratio": {
                    "description": "环氧当量/胺当量",
                    "typical_range": (0.8, 1.2)
                }
            },
            "anhydride_hardeners": {
                "curing_temperature": (120, 180),
                "requires_accelerator": True
            }
        },
        
        "diluents": {
            "name": "稀释剂",
            "reactive_diluents": {
                "description": "参与固化反应的活性稀释剂",
                "viscosity_reduction": (30, 70),  # %
                "max_concentration": 0.3  # 最大用量30%
            },
            "non_reactive_diluents": {
                "description": "不参与反应的惰性稀释剂",
                "volatility": "高",
                "max_concentration": 0.15  # 最大用量15%
            }
        }
    }
    
    # 安全和约束规则
    SAFETY_CONSTRAINTS = {
        "temperature_limits": {
            "epoxy_systems": {
                "safe_max": 200,  # °C
                "flash_point_concern": 150,
                "decomposition_risk": 250
            }
        },
        
        "ratio_constraints": {
            "epoxy_hardener": {
                "stoichiometric_ratio": 1.0,
                "acceptable_range": (0.8, 1.2),
                "under_cure_risk": "<0.8",
                "over_cure_brittleness": ">1.2"
            }
        },
        
        "incompatible_substances": [
            {
                "substance_1": "强酸",
                "substance_2": "强碱",
                "risk": "剧烈反应放热"
            },
            {
                "substance_1": "氧化剂",
                "substance_2": "还原剂",
                "risk": "爆炸危险"
            }
        ]
    }
    
    def __init__(self):
        """初始化化学知识库"""
        self.reaction_database = self.REACTION_TYPES
        self.material_database = self.MATERIAL_PROPERTIES
        self.safety_database = self.SAFETY_CONSTRAINTS
    
    def identify_reaction_type(self, substances: List[str], user_description: str = "") -> str:
        """
        基于物质列表和用户描述识别反应类型
        
        Args:
            substances: 物质名称列表
            user_description: 用户对实验的描述
            
        Returns:
            str: 识别的反应类型
        """
        # 关键词匹配逻辑
        keywords_mapping = {
            "epoxy_curing": ["环氧", "epoxy", "固化", "curing", "树脂", "resin"],
            "polymerization": ["聚合", "polymerization", "单体", "monomer"],
            "catalytic_synthesis": ["催化", "catalysis", "合成", "synthesis"]
        }
        
        combined_text = " ".join(substances + [user_description]).lower()
        
        for reaction_type, keywords in keywords_mapping.items():
            if any(keyword in combined_text for keyword in keywords):
                return reaction_type
        
        return "general_chemical_reaction"  # 默认类型
    
    def get_parameter_suggestions(
        self, 
        reaction_type: str, 
        current_data: pd.DataFrame
    ) -> Dict:
        """
        基于反应类型和当前数据提供参数建议
        
        Args:
            reaction_type: 反应类型
            current_data: 当前实验数据
            
        Returns:
            Dict: 参数建议字典
        """
        suggestions = {}
        
        if reaction_type not in self.reaction_database:
            return self._get_default_suggestions(current_data)
        
        reaction_info = self.reaction_database[reaction_type]
        
        # 温度参数建议
        if "Temperature" in current_data.columns or "temperature" in str(current_data.columns).lower():
            temp_col = [col for col in current_data.columns if 'temperature' in col.lower()][0]
            current_min = current_data[temp_col].min()
            current_max = current_data[temp_col].max()
            
            suggestions["temperature"] = {
                "current_range": (float(current_min), float(current_max)),
                "recommended_range": reaction_info["typical_temperature"],
                "optimal_range": reaction_info.get("temperature_optimal", reaction_info["typical_temperature"]),
                "reasoning": f"{reaction_info['name']}的典型温度范围",
                "safety_note": self._get_temperature_safety_note(reaction_type)
            }
        
        # 比例参数建议
        ratio_columns = [col for col in current_data.columns if 'ratio' in col.lower()]
        if ratio_columns:
            suggestions["ratios"] = {
                "constraint": "所有比例之和应等于1.0",
                "individual_bounds": self._suggest_ratio_bounds(reaction_type, ratio_columns),
                "reasoning": "确保配方总量一致性"
            }
        
        # 催化剂/固化剂浓度建议
        if "catalyst" in reaction_type or "epoxy" in reaction_type:
            key = "catalyst_concentration" if "catalyst" in reaction_type else "catalyst_concentration"
            if key in reaction_info:
                suggestions["catalyst_concentration"] = {
                    "recommended_range": reaction_info[key],
                    "reasoning": "典型催化剂/固化剂浓度范围",
                    "common_values": reaction_info.get("common_hardeners", [])
                }
        
        # 安全建议
        suggestions["safety_warnings"] = reaction_info.get("safety_warnings", [])
        
        return suggestions
    
    def validate_experimental_conditions(
        self, 
        conditions: Dict, 
        reaction_type: str
    ) -> Tuple[bool, List[str]]:
        """
        验证实验条件的化学合理性和安全性
        
        Args:
            conditions: 实验条件字典
            reaction_type: 反应类型
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 警告/错误列表)
        """
        is_valid = True
        warnings = []
        
        if reaction_type not in self.reaction_database:
            return True, ["未找到该反应类型的验证规则"]
        
        reaction_info = self.reaction_database[reaction_type]
        
        # 验证温度
        if "temperature" in conditions:
            temp = conditions["temperature"]
            temp_range = reaction_info.get("typical_temperature", (0, 500))
            
            if temp < temp_range[0] or temp > temp_range[1]:
                warnings.append(
                    f"⚠️ 温度 {temp}°C 超出典型范围 {temp_range}°C。"
                    f"可能影响反应效果。"
                )
                if temp > 200:
                    warnings.append(
                        f"🔴 安全警告：温度过高 ({temp}°C > 200°C)，存在安全风险！"
                    )
                    is_valid = False
        
        # 验证比例和
        ratio_keys = [k for k in conditions.keys() if 'ratio' in k.lower()]
        if ratio_keys:
            ratio_sum = sum(conditions[k] for k in ratio_keys)
            if abs(ratio_sum - 1.0) > 0.01:
                warnings.append(
                    f"⚠️ 比例之和 ({ratio_sum:.3f}) 不等于 1.0，请检查配方。"
                )
        
        # 检查不兼容组合
        incompatible = reaction_info.get("incompatible_combinations", [])
        for combo in incompatible:
            if all(c in str(conditions.values()).lower() for c in combo):
                warnings.append(
                    f"🔴 警告：检测到不兼容组合 {combo}，可能导致危险反应！"
                )
                is_valid = False
        
        return is_valid, warnings
    
    def suggest_quality_metrics(self, reaction_type: str) -> List[str]:
        """
        建议该反应类型的质量指标
        
        Args:
            reaction_type: 反应类型
            
        Returns:
            List[str]: 建议的质量指标列表
        """
        if reaction_type not in self.reaction_database:
            return ["Yield", "Purity", "Conversion"]
        
        reaction_info = self.reaction_database[reaction_type]
        quality_indicators = reaction_info.get("quality_indicators", {})
        
        return [
            f"{key} (范围: {value})" 
            for key, value in quality_indicators.items()
        ]
    
    def get_reaction_info_summary(self, reaction_type: str) -> str:
        """
        获取反应类型的摘要信息
        
        Args:
            reaction_type: 反应类型
            
        Returns:
            str: 格式化的摘要信息
        """
        if reaction_type not in self.reaction_database:
            return f"未找到反应类型 '{reaction_type}' 的信息"
        
        info = self.reaction_database[reaction_type]
        
        summary = f"""
📚 **{info['name']}**

📝 **描述**: {info['description']}

🌡️ **典型温度**: {info.get('typical_temperature', 'N/A')} °C

⚠️ **安全注意事项**:
{''.join([f'   - {warning}\\n' for warning in info.get('safety_warnings', [])])}

🎯 **质量指标**:
{''.join([f'   - {k}: {v}\\n' for k, v in info.get('quality_indicators', {}).items()])}
        """
        
        return summary
    
    # 辅助方法
    def _get_default_suggestions(self, current_data: pd.DataFrame) -> Dict:
        """为未知反应类型提供默认建议"""
        return {
            "general": "请提供更多反应类型信息以获得专业建议",
            "temperature": "建议温度范围: 25-150°C",
            "safety": "遵循标准化学实验安全规程"
        }
    
    def _suggest_ratio_bounds(self, reaction_type: str, ratio_columns: List[str]) -> Dict:
        """为比例参数建议边界"""
        n_ratios = len(ratio_columns)
        
        # 基本约束：每个比例在0到1之间，且和为1
        suggestions = {}
        for col in ratio_columns:
            suggestions[col] = {
                "min": 0.05,  # 最小5%
                "max": 0.95,  # 最大95%
                "recommended": 1.0 / n_ratios  # 均分
            }
        
        return suggestions
    
    def _get_temperature_safety_note(self, reaction_type: str) -> str:
        """获取温度安全提示"""
        safety_limits = self.safety_database.get("temperature_limits", {})
        
        if reaction_type in safety_limits or "epoxy_systems" in safety_limits:
            limits = safety_limits.get(reaction_type, safety_limits["epoxy_systems"])
            return f"安全上限: {limits['safe_max']}°C, 闪点关注温度: {limits['flash_point_concern']}°C"
        
        return "请注意温度安全控制"


# 使用示例
if __name__ == "__main__":
    # 创建知识库实例
    kb = ChemistryKnowledgeBase()
    
    # 识别反应类型
    reaction_type = kb.identify_reaction_type(
        substances=["南亚127e环氧树脂", "1,5-戊二胺"],
        user_description="环氧树脂固化实验"
    )
    print(f"识别的反应类型: {reaction_type}")
    
    # 获取反应信息摘要
    print(kb.get_reaction_info_summary(reaction_type))
    
    # 验证实验条件
    test_conditions = {
        "temperature": 95,
        "SubstanceA_ratio": 0.6,
        "SubstanceB_ratio": 0.4
    }
    
    is_valid, warnings = kb.validate_experimental_conditions(test_conditions, reaction_type)
    print(f"\n实验条件验证: {'✅ 有效' if is_valid else '❌ 无效'}")
    for warning in warnings:
        print(warning)


