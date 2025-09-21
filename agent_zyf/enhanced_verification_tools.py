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

"""Enhanced Verification Agent Tools - 实现7个核心任务的工具函数"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.adk.tools import ToolContext

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
    基于化学知识的智能参数建议系统
    """
    
    def analyze_experimental_context(self, data: pd.DataFrame, user_description: str = "") -> dict:
        """
        分析实验背景，提供智能参数建议
        """
        suggestions = {}
        
        # 1. 分析分子类型和特性
        molecular_analysis = self._analyze_molecules(data)
        
        # 2. 生成参数边界建议
        for col in data.columns:
            if 'ratio' in col.lower():
                # 安全的数值转换
                numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
                if len(numeric_data) == 0:
                    continue
                current_range = [float(numeric_data.min()), float(numeric_data.max())]
                suggestions[col] = {
                    "current_range": current_range,
                    "suggested_bounds": self._suggest_ratio_bounds(col, current_range),
                    "reasoning": f"基于{col}的当前取值范围和化学常识",
                    "constraints": self._suggest_constraints(col)
                }
            elif 'temperature' in col.lower():
                # 安全的数值转换
                numeric_data = pd.to_numeric(data[col], errors='coerce').dropna()
                if len(numeric_data) == 0:
                    continue
                current_range = [float(numeric_data.min()), float(numeric_data.max())]
                suggestions[col] = {
                    "current_range": current_range,
                    "suggested_bounds": self._suggest_temperature_bounds(current_range),
                    "reasoning": "基于反应类型和安全考虑",
                }
        
        return suggestions
    
    def _analyze_molecules(self, data: pd.DataFrame) -> dict:
        """
        分析分子类型和特性
        """
        analysis = {}
        
        smiles_columns = [col for col in data.columns if 'SMILE' in col.upper()]
        for col in smiles_columns:
            smiles_list = data[col].dropna().astype(str).tolist()
            analysis[col] = {
                "molecule_count": len(set(smiles_list)),
                "avg_length": np.mean([len(s) for s in smiles_list]),
                "contains_aromatic": any('c' in s.lower() or 'C' in s for s in smiles_list),
                "molecular_diversity": len(set(smiles_list)) / len(smiles_list) if smiles_list else 0
            }
        
        return analysis
    
    def _suggest_ratio_bounds(self, column_name: str, current_range: list) -> tuple:
        """
        建议比例参数的边界
        """
        min_val, max_val = current_range
        
        # 基于化学常识的建议
        if 'catalyst' in column_name.lower():
            # 催化剂通常是少量的
            return (0.001, 0.1)
        elif 'solvent' in column_name.lower():
            # 溶剂可以是主要成分
            return (0.0, 0.5)
        else:
            # 一般物质的合理范围
            buffer = (max_val - min_val) * 0.2
            return (max(0.0, min_val - buffer), min(1.0, max_val + buffer))
    
    def _suggest_temperature_bounds(self, current_range: list) -> tuple:
        """
        建议温度参数的边界
        """
        min_temp, max_temp = current_range
        
        # 基于安全和实用性的建议
        safety_buffer = 20  # 安全缓冲区
        return (max(20, min_temp - safety_buffer), min(200, max_temp + safety_buffer))
    
    def _suggest_constraints(self, column_name: str) -> list:
        """
        建议约束条件
        """
        constraints = []
        
        if 'ratio' in column_name.lower():
            constraints.append({
                "type": "sum_constraint",
                "description": "所有比例之和应等于1.0",
                "implementation": "DiscreteSumConstraint"
            })
        
        if 'temperature' in column_name.lower():
            constraints.append({
                "type": "safety_constraint", 
                "description": "温度应在安全操作范围内",
                "range": (20, 200)
            })
        
        return constraints


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
        为用户定义的特殊物质创建BayBE参数
        """
        parameters = []
        
        # 处理没有SMILES的特殊物质
        for special_substance in user_special_data["substances_without_smiles"]:
            substance_name = special_substance["name"]
            column_prefix = special_substance["column_prefix"]
            
            # 检查是否有对应的比例列
            ratio_col = f"{column_prefix}_ratio"
            if ratio_col in df.columns:
                # 对于特殊物质，我们可以用名称作为分类参数
                # 或者根据比例创建数值参数
                unique_names = df[f"{column_prefix}_name"].dropna().unique()
                if len(unique_names) > 1:
                    # 如果有多个不同的特殊物质名称，创建分类参数
                    from baybe.parameters import CategoricalParameter
                    param = CategoricalParameter(
                        name=f"{column_prefix}_special_substance",
                        values=[str(name) for name in unique_names],
                        encoding="OHE"  # One-Hot Encoding
                    )
                    parameters.append({
                        "parameter": param,
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
                    from baybe.parameters import NumericalContinuousParameter
                    param = NumericalContinuousParameter(
                        name=f"custom_{col}",
                        bounds=(float(min(numeric_values)), float(max(numeric_values)))
                    )
                else:  # 分类数据
                    from baybe.parameters import CategoricalParameter
                    param = CategoricalParameter(
                        name=f"custom_{col}",
                        values=[str(v) for v in values],
                        encoding="OHE"
                    )
                
                parameters.append({
                    "parameter": param,
                    "source": "user_defined_descriptor",
                    "original_column": col
                })
        
        return parameters
    
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
        df = pd.read_csv(file_path)
        
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
        # ===== 任务1: 数据质量验证 =====
        quality_report = _perform_data_quality_check(file_path)
        
        if not quality_report["is_valid"]:
            return f"数据质量检查失败：\n{json.dumps(quality_report, indent=2, ensure_ascii=False)}"
        
        # ===== 任务2: SMILES验证 =====
        df = pd.read_csv(file_path)
        smiles_validator = SimplifiedSMILESValidator()
        smiles_validation = smiles_validator.validate_smiles_data(df)
        
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
        
        # ===== 任务7: 数据标准化 =====
        standardized_data = _standardize_data(df, smiles_validation)
        
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
        df = pd.read_csv(file_path)
        
        quality_report = {
            "is_valid": True,
            "issues": [],
            "statistics": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
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
                    "count": len(outliers),
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
            "substances_validated": smiles_validation["substances_validated"]
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

💡 **智能参数建议**:
"""
    
    # 添加参数建议详情
    for param, suggestion in interaction_data['parameter_suggestions'].items():
        prompt += f"\n📌 **{param}**:"
        prompt += f"\n   - 当前范围: {suggestion['current_range']}"
        prompt += f"\n   - 建议范围: {suggestion['suggested_bounds']}" 
        prompt += f"\n   - 理由: {suggestion['reasoning']}"
    
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
   - 优先级权重

2. **参数边界确认**:
   是否接受上述智能建议的参数范围？如需调整请说明。

3. **约束条件**:
   是否有特殊的约束条件（如某些物质不能同时使用）？

4. **实验设计参数**:
   - 计划的实验批次大小 (batch_size)
   - 最大实验轮数 (n_doe_iterations)
   - 预算约束 (总实验数量限制)

请提供您的回答，我将根据您的需求生成优化配置。
"""
    
    return prompt


# 主要的增强验证工具函数
def collect_optimization_goals(user_response: str, tool_context: ToolContext) -> str:
    """
    收集用户的优化目标和配置（任务5和6）
    """
    state = tool_context.state
    verification_results = state.get("verification_results", {})
    
    try:
        # 解析用户响应（这里简化处理，实际可能需要更复杂的NLP）
        optimization_config = _parse_user_response(user_response, verification_results)
        
        # 生成BayBE兼容的配置
        baybe_config = _generate_baybe_config(optimization_config, verification_results)
        
        # 更新状态
        state["optimization_config"] = optimization_config
        state["baybe_campaign_config"] = baybe_config
        state["verification_status"] = "completed_with_user_input"
        state["ready_for_searchspace_construction"] = True
        
        return f"""
✅ **优化配置已完成**

📋 **配置摘要**:
- 目标数量: {len(optimization_config.get('targets', []))}
- 参数数量: {len(optimization_config.get('parameters', []))}
- 约束条件: {len(optimization_config.get('constraints', []))}
- 特殊编码: {len(verification_results.get('custom_encodings', {}))}

🚀 **下一步**: 系统将构建BayBE搜索空间并准备优化Campaign。

📄 **BayBE配置已保存到会话状态**，可以传递给SearchSpace Construction Agent。
        """
        
    except Exception as e:
        return f"解析用户配置时出错: {str(e)}\n请重新提供配置信息。"


def _parse_user_response(user_response: str, verification_results: dict) -> dict:
    """
    解析用户响应（简化版本）
    """
    # 这里是简化的解析逻辑，实际应用中可能需要更复杂的NLP
    config = {
        "targets": [],
        "parameters": verification_results.get("parameter_suggestions", {}),
        "constraints": [],
        "experimental_settings": {
            "batch_size": 5,  # 默认值
            "max_iterations": 20
        }
    }
    
    # 简单的关键词提取（实际应该使用更智能的解析）
    if "最大化" in user_response or "maximize" in user_response.lower():
        config["default_optimization"] = "MAX"
    elif "最小化" in user_response or "minimize" in user_response.lower():
        config["default_optimization"] = "MIN"
    
    return config


def _generate_baybe_config(optimization_config: dict, verification_results: dict) -> dict:
    """
    生成BayBE兼容的配置格式（任务6）
    """
    if not BAYBE_AVAILABLE:
        return {"error": "BayBE not available"}
    
    # 根据开发文档的标准格式生成配置
    baybe_config = {
        "campaign_info": {
            "name": "chemical_optimization",
            "created_at": datetime.now().isoformat(),
            "description": "ChemBoMAS Enhanced Verification Agent generated configuration"
        },
        "targets": [],
        "parameters": [],
        "constraints": [],
        "objective_config": {
            "type": "DesirabilityObjective",
            "weights": [1.0],  # 默认权重
            "scalarizer": "GEOM_MEAN"
        },
        "experimental_config": {
            "batch_size": optimization_config["experimental_settings"]["batch_size"],
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
    
    # 测试2: 参数建议器
    print("\n2. 测试参数建议器...")
    advisor = IntelligentParameterAdvisor()
    suggestions = advisor.analyze_experimental_context(test_data, "环氧树脂固化实验")
    print(f"   参数建议数量: {len(suggestions)}")
    for param, suggestion in suggestions.items():
        print(f"   {param}: {suggestion['current_range']} → {suggestion['suggested_bounds']}")
    
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
