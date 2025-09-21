# Verification Agent 功能增强技术规范

## 文档概述
本文档详细描述了 Verification Agent 的功能增强需求，包括数据完整性验证、用户交互模块的设计，以及与 BayBE 优化引擎的集成规划。

---

## 当前 Verification Agent 分析

### 现有功能
```python
def verification(file_path: str, tool_context: ToolContext) -> str:
    """
    当前功能：
    - 读取CSV文件
    - 计算基本统计信息（行数、列数）
    - 识别物质列和目标列
    - 验证列名格式
    """
```

### 局限性
1. **数据质量检查不足**: 仅检查列名格式，未验证数据完整性
2. **缺乏用户交互**: 无法获取用户的优化目标和约束条件
3. **参数传递不完整**: 未收集BO所需的关键参数
4. **错误处理简单**: 缺乏详细的异常情况处理

---

## 功能增强设计

### 1. 数据完整性验证模块

#### 1.1 LLM驱动的数据质量检查

**实现方案**:
```python
def llm_data_quality_check(df: pd.DataFrame, tool_context: ToolContext) -> dict:
    """
    使用LLM进行智能数据质量分析
    
    Returns:
        quality_report: {
            "missing_values": {...},
            "outliers": {...},
            "data_types": {...},
            "anomalies": {...},
            "suggestions": [...]
        }
    """
    
    # 1. 生成数据摘要供LLM分析
    data_summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "sample_rows": df.head(3).to_dict('records')
    }
    
    # 2. 构建LLM提示
    prompt = f"""
    请分析以下化学实验数据的质量问题：
    
    数据摘要：{data_summary}
    
    请检查：
    1. 缺失值情况（null、空字符串、"NaN"等）
    2. 异常值（超出合理范围的数值）
    3. 数据类型一致性
    4. SMILES字符串有效性
    5. 比例数据的合理性（如是否和为1）
    
    返回JSON格式的分析报告。
    """
    
    # 3. 调用LLM分析
    # 4. 解析LLM响应
    # 5. 生成详细报告
```

#### 1.2 自动化数据清理建议

**功能设计**:
- 识别并标记可疑数据点
- 提供数据修复建议
- 自动生成数据清理脚本
- 支持用户确认后执行清理操作

```python
def generate_cleaning_suggestions(quality_report: dict) -> list:
    """
    基于质量报告生成清理建议
    
    Returns:
        suggestions: [
            {
                "issue": "missing_values",
                "location": "row 5, column 'SubstanceA_ratio'",
                "suggestion": "interpolate from neighboring values",
                "confidence": 0.85,
                "auto_fix": True
            },
            ...
        ]
    """
```

### 2. 用户交互模块设计

#### 2.1 优化目标收集接口

**交互流程**:
```python
def collect_optimization_objectives(df: pd.DataFrame, tool_context: ToolContext) -> dict:
    """
    与用户交互收集优化目标
    
    交互内容：
    1. 识别目标变量
    2. 确定优化方向（最大化/最小化）
    3. 设置目标值范围
    4. 确定优先级权重
    """
    
    # 1. 自动识别潜在目标变量
    target_columns = [col for col in df.columns if col.startswith('Target_')]
    
    # 2. 与用户交互确认优化目标
    conversation_prompt = f"""
    我识别到以下目标变量：{target_columns}
    
    请为每个目标变量指定：
    1. 优化方向（maximize/minimize/match）
    2. 期望的目标值范围（用于bounds设置）
    3. 变换函数类型（LINEAR/BELL/TRIANGULAR）
    
    例如：
    - Target_alpha_tg: mode=MAX, bounds=(80, 100), transformation=LINEAR
    - Target_beta_impactstrength: mode=MAX, bounds=(100, 150), transformation=BELL
    
    注意：多目标权重将在后续的DesirabilityObjective配置中统一设置
    """
    
    # 3. 解析用户响应
    # 4. 验证输入合理性
    # 5. 生成优化配置
```

#### 2.2 可调变量识别模块

**实现逻辑**:
```python
def identify_adjustable_variables(df: pd.DataFrame, tool_context: ToolContext) -> dict:
    """
    识别可调整的变量及其约束
    
    自动识别：
    1. 比例变量（ratio列）
    2. 类别变量（name列中的不同选择）
    3. 连续变量（温度、时间等）
    
    用户确认：
    1. 变量的调整范围和边界
    2. 变量间的约束关系
    3. 参数类型（连续/离散/分类）
    """
    
    # 1. 自动识别变量类型
    ratio_vars = [col for col in df.columns if 'ratio' in col.lower()]
    name_vars = [col for col in df.columns if 'name' in col.lower()]
    
    # 2. 分析变量取值范围
    variable_analysis = {}
    for var in ratio_vars:
        variable_analysis[var] = {
            "type": "continuous",
            "current_range": [df[var].min(), df[var].max()],
            "suggested_bounds": [max(0, df[var].min() - 0.1), min(1, df[var].max() + 0.1)]
        }
    
    # 3. 与用户交互确认
    # 4. 收集约束条件
```

#### 2.3 BayBE约束条件定义

**基于BayBE官方API的约束类型**:
```python
# 参考: https://emdgroup.github.io/baybe/stable/examples/Constraints_Discrete/custom_constraints.html
from baybe.constraints import (
    DiscreteSumConstraint,
    DiscreteProductConstraint, 
    DiscreteExcludeConstraint,
    ContinuousLinearConstraint
)

constraint_types = {
    "discrete_sum_constraint": {
        "description": "离散参数和约束",
        "example": "所有比例参数之和等于特定值",
        "baybe_class": "DiscreteSumConstraint",
        "implementation": """
        DiscreteSumConstraint(
            parameters=["SubstanceA_ratio", "SubstanceB_ratio"],
            condition=ThresholdCondition(threshold=1.0, operator="=")
        )
        """
    },
    "discrete_product_constraint": {
        "description": "离散参数乘积约束", 
        "example": "某些参数乘积的约束",
        "baybe_class": "DiscreteProductConstraint",
        "implementation": """
        DiscreteProductConstraint(
            parameters=["param1", "param2"],
            condition=ThresholdCondition(threshold=0.5, operator="<=")
        )
        """
    },
    "discrete_exclude_constraint": {
        "description": "排除特定参数组合",
        "example": "某些物质不能同时使用",
        "baybe_class": "DiscreteExcludeConstraint", 
        "implementation": """
        DiscreteExcludeConstraint(
            parameters=["SubstanceA_name", "SubstanceB_name"],
            combos_to_exclude=[("催化剂A", "催化剂B")]
        )
        """
    },
    "continuous_linear_constraint": {
        "description": "连续参数线性约束",
        "example": "连续变量的线性不等式约束",
        "baybe_class": "ContinuousLinearConstraint",
        "implementation": """
        ContinuousLinearConstraint(
            parameters=["ratio_A", "ratio_B"],
            coefficients=[1.0, 1.0],
            rhs=1.0,
            operator="="
        )
        """
    }
}
```

### 3. 参数传递接口设计

#### 3.1 标准化配置格式

**基于BayBE官方API的配置格式**:
```python
# 参考: https://emdgroup.github.io/baybe/stable/examples/Searchspaces/hybrid_space.html
from baybe import Campaign
from baybe.parameters import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    CategoricalParameter
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective, ParetoObjective
from baybe.constraints import (
    DiscreteSumConstraint,
    ContinuousLinearConstraint
)

optimization_config = {
    "campaign_info": {
        "name": "chemical_optimization",
        "created_at": "2025-01-xx",
        "description": "Chemical reaction optimization campaign"
    },
    
    # BayBE目标定义 - 不包含权重，权重在DesirabilityObjective中定义
    "targets": [
        {
            "name": "Target_alpha_tg",
            "mode": "MAX",  # BayBE标准格式：MAX, MIN, MATCH
            "bounds": (70, 100),  # 元组格式
            "transformation": "LINEAR"  # LINEAR, BELL, TRIANGULAR
        },
        {
            "name": "Target_beta_impactstrength", 
            "mode": "MAX",
            "bounds": (80, 150),
            "transformation": "LINEAR"
        }
    ],
    
    # BayBE参数定义
    "parameters": [
        {
            "name": "SubstanceA_ratio",
            "type": "NumericalContinuousParameter",
            "bounds": (0.1, 0.8),  # 元组格式
            "tolerance": 0.01
        },
        {
            "name": "SubstanceB_ratio", 
            "type": "NumericalDiscreteParameter",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5],  # 离散值列表
            "tolerance": 0.01
        },
        {
            "name": "SubstanceA_name",
            "type": "CategoricalParameter",
            "values": ["南亚127e", "催化剂A", "催化剂B"],
            "encoding": "OHE"  # One-Hot Encoding
        }
    ],
    
    # BayBE约束定义
    "constraints": [
        {
            "type": "DiscreteSumConstraint",
            "parameters": ["SubstanceA_ratio", "SubstanceB_ratio"],
            "condition": {
                "threshold": 1.0,
                "operator": "="
            }
        }
    ],
    
    # BayBE目标函数配置（替代权重）
    "objective_config": {
        "type": "DesirabilityObjective",  # 或 "ParetoObjective" 用于帕累托优化
        "weights": [0.6, 0.4],  # 仅在DesirabilityObjective中使用
        "scalarizer": "GEOM_MEAN"  # GEOM_MEAN 或 MEAN
    },
    
    # 实验设置
    "experimental_settings": {
        "batch_size": 5,
        "recommender": "TwoPhaseMetaRecommender"  # BayBE推荐器类型
    }
}
```

#### 3.2 BayBE搜索空间构建示例

**基于BayBE混合搜索空间的实际实现**:
```python
# 参考: https://emdgroup.github.io/baybe/stable/examples/Searchspaces/hybrid_space.html
def create_chemical_searchspace(config: dict) -> SearchSpace:
    """
    为化学实验创建BayBE搜索空间
    """
    from baybe.parameters import (
        NumericalContinuousParameter,
        NumericalDiscreteParameter, 
        CategoricalParameter
    )
    from baybe.searchspace import SearchSpace
    
    parameters = []
    
    # 连续参数示例: 物质比例
    cont_parameters = [
        NumericalContinuousParameter(
            name="SubstanceA_ratio",
            bounds=(0.1, 0.8),
            tolerance=0.01
        ),
        NumericalContinuousParameter(
            name="Temperature",
            bounds=(80.0, 120.0),
            tolerance=1.0
        )
    ]
    
    # 离散参数示例: 反应时间
    disc_parameters = [
        NumericalDiscreteParameter(
            name="Reaction_time",
            values=[30, 60, 90, 120, 180],  # 分钟
            tolerance=5.0
        ),
        NumericalDiscreteParameter(
            name="SubstanceB_ratio",
            values=[0.1, 0.15, 0.2, 0.25, 0.3],
            tolerance=0.01
        )
    ]
    
    # 分类参数示例: 催化剂选择
    cat_parameters = [
        CategoricalParameter(
            name="Catalyst_type",
            values=["Pd", "Pt", "Ru", "Ni"],
            encoding="OHE"
        ),
        CategoricalParameter(
            name="Solvent_type", 
            values=["THF", "DCM", "Toluene", "EtOH"],
            encoding="OHE"
        )
    ]
    
    # 组合所有参数
    all_parameters = cont_parameters + disc_parameters + cat_parameters
    
    # 创建搜索空间
    searchspace = SearchSpace.from_product(parameters=all_parameters)
    
    return searchspace

def apply_constraints_to_searchspace(searchspace: SearchSpace, constraints_config: list):
    """
    为搜索空间应用约束条件
    """
    from baybe.constraints import (
        DiscreteSumConstraint,
        DiscreteExcludeConstraint,
        ContinuousLinearConstraint
    )
    from baybe.constraints.conditions import ThresholdCondition
    
    constraints = []
    
    for constraint_config in constraints_config:
        if constraint_config["type"] == "DiscreteSumConstraint":
            constraint = DiscreteSumConstraint(
                parameters=constraint_config["parameters"],
                condition=ThresholdCondition(
                    threshold=constraint_config["condition"]["threshold"],
                    operator=constraint_config["condition"]["operator"]
                )
            )
            constraints.append(constraint)
    
    return constraints
```

#### 3.3 状态管理

**增强的状态传递**:
```python
def update_session_state(tool_context: ToolContext, config: dict):
    """
    更新会话状态，传递给下游智能体
    """
    state = tool_context.state
    
    # 添加BayBE优化配置
    state["baybe_config"] = config
    state["verification_status"] = "completed_with_user_input"
    state["data_quality_score"] = calculate_quality_score(config)
    state["ready_for_baybe"] = True
    
    # 添加搜索空间信息
    state["searchspace_info"] = {
        "parameter_count": len(config["parameters"]),
        "constraint_count": len(config.get("constraints", [])),
        "target_count": len(config["targets"])
    }
    
    # 添加审计信息
    state["verification_timestamp"] = datetime.now().isoformat()
    state["user_preferences"] = extract_user_preferences(config)
```

---

## BayBE 集成技术规范

### 1. BayBE 概述

**官方文档**: [https://emdgroup.github.io/baybe/stable/](https://emdgroup.github.io/baybe/stable/)  
**GitHub**: https://github.com/emdgroup/baybe

**核心优势**:
- 现代化的贝叶斯优化框架（基于PyTorch/BoTorch）
- 原生支持混合搜索空间（连续、离散、分类参数）
- 灵活的约束处理系统
- 优秀的多目标优化支持（Pareto前沿、加权组合）
- 内置特征处理和描述符优化能力
- 活跃的开源社区和完善的文档

**重要更新**:
基于BayBE官方文档的深入研究，本规范文档已进行以下关键修正：
1. **删除手动权重配置**: BayBE中目标权重在`DesirabilityObjective`中统一管理
2. **更新约束API**: 使用真实的BayBE约束类（`DiscreteSumConstraint`、`ContinuousLinearConstraint`等）
3. **修正搜索空间构建**: 基于`SearchSpace.from_product()`的官方方法
4. **参数类型规范**: 严格按照BayBE的参数类型体系（`NumericalContinuousParameter`、`NumericalDiscreteParameter`、`CategoricalParameter`）

### 2. 集成架构设计

#### 2.1 适配器模式

```python
class BayBEAdapter:
    """
    基于BayBE官方API的适配器实现
    参考: https://emdgroup.github.io/baybe/stable/examples/Searchspaces/hybrid_space.html
    """
    
    def __init__(self, optimization_config: dict):
        self.config = optimization_config
        self.campaign = self._create_campaign()
    
    def _create_campaign(self):
        """根据配置创建BayBE Campaign"""
        from baybe import Campaign
        from baybe.parameters import (
            NumericalContinuousParameter, 
            NumericalDiscreteParameter,
            CategoricalParameter
        )
        from baybe.searchspace import SearchSpace
        from baybe.targets import NumericalTarget
        from baybe.objectives import DesirabilityObjective, ParetoObjective
        from baybe.constraints import (
            DiscreteSumConstraint,
            ContinuousLinearConstraint
        )
        
        # 1. 创建参数
        parameters = []
        for param_config in self.config["parameters"]:
            if param_config["type"] == "NumericalContinuousParameter":
                param = NumericalContinuousParameter(
                    name=param_config["name"],
                    bounds=param_config["bounds"],
                    tolerance=param_config.get("tolerance", 0.01)
                )
            elif param_config["type"] == "NumericalDiscreteParameter":
                param = NumericalDiscreteParameter(
                    name=param_config["name"],
                    values=param_config["values"],
                    tolerance=param_config.get("tolerance", 0.01)
                )
            elif param_config["type"] == "CategoricalParameter":
                param = CategoricalParameter(
                    name=param_config["name"],
                    values=param_config["values"],
                    encoding=param_config.get("encoding", "OHE")
                )
            parameters.append(param)
        
        # 2. 创建搜索空间
        searchspace = SearchSpace.from_product(parameters=parameters)
        
        # 3. 创建目标
        targets = []
        for target_config in self.config["targets"]:
            target = NumericalTarget(
                name=target_config["name"],
                mode=target_config["mode"],
                bounds=target_config.get("bounds"),
                transformation=target_config.get("transformation", "LINEAR")
            )
            targets.append(target)
        
        # 4. 创建目标函数
        objective_config = self.config.get("objective_config", {})
        if objective_config.get("type") == "ParetoObjective":
            objective = ParetoObjective(targets=targets)
        else:
            # 默认使用DesirabilityObjective
            objective = DesirabilityObjective(
                targets=targets,
                weights=objective_config.get("weights"),
                scalarizer=objective_config.get("scalarizer", "GEOM_MEAN")
            )
        
        # 5. 创建Campaign
        return Campaign(
            searchspace=searchspace,
            objective=objective
        )
    
    def recommend_experiments(self, batch_size: int = 5) -> pd.DataFrame:
        """生成实验推荐"""
        recommendations = self.campaign.recommend(batch_size=batch_size)
        return recommendations
    
    def add_experiments(self, experiments: pd.DataFrame):
        """添加实验结果"""
        self.campaign.add_measurements(experiments)
```

#### 2.2 渐进式迁移策略

**阶段1: 并行测试**
```python
class HybridRecommender:
    """
    同时支持EDBO+和BayBE的混合推荐器
    """
    
    def __init__(self, config: dict, use_baybe: bool = False):
        self.use_baybe = use_baybe
        
        if use_baybe:
            self.recommender = BayBEAdapter(config)
        else:
            self.recommender = EDBOPlusAdapter(config)
    
    def generate_recommendations(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.recommender.recommend_experiments(data)
```

### 3. 性能对比框架

```python
class OptimizationBenchmark:
    """
    EDBO+ vs BayBE 性能对比
    """
    
    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data
        self.results = {}
    
    def run_comparison(self, config: dict):
        """运行对比测试"""
        
        # 测试EDBO+
        edbo_results = self._test_edbo_plus(config)
        
        # 测试BayBE
        baybe_results = self._test_baybe(config)
        
        # 生成对比报告
        return self._generate_comparison_report(edbo_results, baybe_results)
    
    def _evaluate_metrics(self, true_values, predictions):
        """评估指标"""
        return {
            "rmse": mean_squared_error(true_values, predictions, squared=False),
            "r2": r2_score(true_values, predictions),
            "mae": mean_absolute_error(true_values, predictions),
            "convergence_rate": self._calculate_convergence_rate(predictions)
        }
```

---

## 实施计划

### 第一阶段：数据质量增强 (2周)
1. 实现LLM驱动的数据质量检查
2. 开发自动化清理建议功能
3. 完善错误处理和用户反馈

### 第二阶段：用户交互模块 (3周)
1. 设计优化目标收集界面
2. 实现可调变量识别逻辑
3. 开发约束条件定义功能
4. 完善参数验证机制

### 第三阶段：BayBE集成准备 (2周)
1. 研究BayBE API和最佳实践
2. 设计适配器架构
3. 实现配置格式转换
4. 开发性能对比工具

### 第四阶段：测试和优化 (2周)
1. 单元测试和集成测试
2. 性能基准测试
3. 用户体验优化
4. 文档完善

---

## 测试策略

### 单元测试
```python
def test_data_quality_check():
    """测试数据质量检查功能"""
    
def test_user_interaction():
    """测试用户交互流程"""
    
def test_parameter_conversion():
    """测试参数格式转换"""
    
def test_baybe_integration():
    """测试BayBE集成"""
```

### 集成测试
- 完整工作流程测试
- 边界条件测试
- 异常情况处理测试
- 性能压力测试

### 用户验收测试
- 真实数据测试
- 用户体验评估
- 功能完整性验证

---

## 风险评估

### 技术风险
1. **BayBE兼容性**: 可能存在API变更
2. **性能影响**: 新功能可能影响响应时间
3. **LLM稳定性**: LLM调用可能不稳定

### 缓解措施
1. 版本锁定和向后兼容设计
2. 性能监控和优化
3. 降级方案和错误处理
4. 充分的测试覆盖

---

## 成功指标

### 功能指标
- 数据质量检测准确率 > 95%
- 用户交互完成率 > 90%
- 参数传递成功率 = 100%

### 性能指标
- 验证流程耗时 < 60秒
- 用户交互响应时间 < 3秒
- 系统可用性 > 99%

### 用户体验指标
- 用户满意度 > 4.0/5.0
- 任务完成率 > 95%
- 错误率 < 5%
