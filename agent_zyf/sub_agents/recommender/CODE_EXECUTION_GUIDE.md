# BayBE 代码执行工具使用指南

## 概述

> 注意：当前版本已禁用 `execute_baybe_code`（不在 Recommender 工具列表中）。
> 请使用标准工具（如 `build_campaign_and_recommend`、`generate_recommendations`）完成流程。

`execute_baybe_code` 工具允许 LLM 直接生成并执行 Python 代码来调用 BayBE，提供了比预定义工具更灵活的方式。

## 为什么需要这个工具？

### 当前架构的问题
- **工具函数过于复杂**：`tools.py` 有 2753 行代码
- **处理逻辑死板**：预定义工具无法覆盖所有场景
- **灵活性不足**：特殊需求需要修改工具函数代码
- **维护成本高**：每次新需求都要修改大量代码

### 代码执行工具的优势
- ✅ **更灵活**：LLM 可以根据具体需求生成代码
- ✅ **更直接**：直接调用 BayBE API，无需中间层
- ✅ **更快速**：无需修改工具函数，立即尝试新方法
- ✅ **更实验性**：可以测试 BayBE 的新功能

## 使用场景

### 1. 标准工具无法满足的需求
当预定义工具（`build_campaign_and_recommend`、`generate_recommendations` 等）无法处理用户的特殊需求时。

### 2. 复杂的 Campaign 构建
需要特殊的参数组合、约束条件或目标函数配置。

### 3. 实验性的优化策略
想要尝试 BayBE 的新功能或自定义推荐策略。

### 4. 复杂的数据处理
需要在优化前进行特殊的数据预处理或后处理。

## 可用上下文

执行代码时可以访问以下变量和模块：

### BayBE 相关
```python
Campaign
CategoricalParameter, NumericalContinuousParameter, NumericalDiscreteParameter
SearchSpace
NumericalTarget
DesirabilityObjective, ParetoObjective
DiscreteSumConstraint, ContinuousLinearConstraint
ThresholdCondition
add_fake_measurements
BotorchRecommender, RandomRecommender, FPSRecommender  # 推荐器（如果可用）
```

### 标准库
```python
pd  # pandas
np  # numpy
json
os
tempfile
datetime
```

### 上下文变量
```python
campaign              # 当前 session 的 Campaign 对象（如果存在）
state                 # 当前 session 的状态字典
session_id            # 当前 session ID
verification_results  # Enhanced Verification 的结果（包含SMILES验证、参数建议等）
optimization_config   # 优化配置（目标、约束、batch_size等）
baybe_campaign_config # BayBE Campaign 配置
standardized_data_path # 标准化数据文件路径
current_data_path     # 当前数据文件路径
session_dir           # Session 目录路径
```

### 辅助函数
```python
_get_campaign_from_cache(session_id)  # 获取 Campaign
_save_campaign_to_cache(session_id, campaign)  # 保存 Campaign
_read_csv_clean(path)  # 安全读取 CSV
```

## 使用示例

### 示例 1: 创建自定义 Campaign

```python
from baybe.parameters import NumericalContinuousParameter, CategoricalParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective
from baybe import Campaign

# 创建参数
temp_param = NumericalContinuousParameter(name="temperature", bounds=(0, 200))
catalyst_param = CategoricalParameter(
    name="catalyst", 
    values=["A", "B", "C"],
    encoding="OHE"
)

# 创建搜索空间
searchspace = SearchSpace.from_product(parameters=[temp_param, catalyst_param])

# 创建目标
target = NumericalTarget(name="yield", mode="MAX")

# 创建目标函数
objective = DesirabilityObjective(targets=[target])

# 创建 Campaign
campaign = Campaign(searchspace=searchspace, objective=objective)

# 保存到缓存
_save_campaign_to_cache(session_id, campaign)

print(f"Campaign created with {len(searchspace.parameter_names)} parameters")
result = f"Campaign created successfully. Parameters: {searchspace.parameter_names}"
```

### 示例 2: 自定义推荐策略

```python
if campaign:
    # 使用自定义 batch_size
    recommendations = campaign.recommend(batch_size=10)
    
    # 添加自定义筛选
    filtered = recommendations[recommendations['temperature'] > 100]
    
    print(f"Generated {len(recommendations)} recommendations")
    print(f"Filtered to {len(filtered)} high-temperature experiments")
    
    result = filtered.to_dict('records')
else:
    result = "No campaign available"
```

### 示例 3: 复杂的数据处理

```python
import pandas as pd

# 从 state 获取数据路径
data_path = state.get('standardized_data_path')
if data_path and os.path.exists(data_path):
    df = _read_csv_clean(data_path)
    
    # 自定义数据处理
    df['normalized_yield'] = (df['Target_yield'] - df['Target_yield'].min()) / \
                              (df['Target_yield'].max() - df['Target_yield'].min())
    
    # 保存处理后的数据
    output_path = os.path.join(os.path.dirname(data_path), 'processed_data.csv')
    df.to_csv(output_path, index=False)
    
    result = f"Processed {len(df)} rows. Saved to {output_path}"
else:
    result = "Data file not found"
```

### 示例 4: 使用验证结果和配置

```python
# 使用verification_results中的信息
if verification_results:
    # 获取SMILES验证结果
    smiles_validation = verification_results.get('smiles_validation', {})
    canonical_mapping = smiles_validation.get('canonical_smiles_mapping', {})
    
    # 获取参数建议
    parameter_suggestions = verification_results.get('parameter_suggestions', {})
    
    print(f"已验证的SMILES数量: {len(canonical_mapping)}")
    print(f"参数建议数量: {len(parameter_suggestions)}")
    
    result = {
        'smiles_count': len(canonical_mapping),
        'parameter_suggestions': list(parameter_suggestions.keys())
    }
else:
    result = "验证结果不可用"

# 使用optimization_config
if optimization_config:
    targets = optimization_config.get('targets', [])
    batch_size = optimization_config.get('batch_size', 5)
    
    print(f"优化目标数量: {len(targets)}")
    print(f"批次大小: {batch_size}")
    
    result = {
        'targets': targets,
        'batch_size': batch_size
    }
```

### 示例 5: 读取和处理数据文件

```python
# 读取标准化数据
if standardized_data_path and os.path.exists(standardized_data_path):
    df = _read_csv_clean(standardized_data_path)
    
    # 数据统计
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'smiles_cols': [col for col in df.columns if 'SMILE' in col.upper()],
        'target_cols': [col for col in df.columns if col.startswith('Target_')]
    }
    
    result = stats
else:
    result = "数据文件不存在"
```

## 安全限制

代码执行包含安全检查，禁止以下操作：
- `__import__`, `eval()`, `exec()`, `compile()`
- `open()`, `file()`
- `input()`, `raw_input()`
- `exit()`, `quit()`, `sys.exit`
- `subprocess`, `os.system`, `os.popen`
- `shutil`
- `pickle.loads`, `marshal.loads`

## 执行限制

- **时间限制**：代码执行时间限制为30秒（Unix系统）
- **资源限制**：建议避免处理过大的数据集
- **状态同步**：修改Campaign后会自动保存到缓存

## 代码模板库

我们提供了常用操作的代码模板，位于 `CODE_TEMPLATES.md`：
- 基础Campaign构建
- 推荐生成
- 数据处理
- 分析和可视化
- 高级操作（约束、批量推荐等）

可以直接复制模板代码，根据需求修改后使用。

## 最佳实践

### ✅ 推荐做法
1. **优先使用标准工具**：对于常见操作，使用预定义工具更可靠
2. **代码简洁明了**：生成易于理解和调试的代码
3. **错误处理**：在代码中包含适当的错误处理
4. **保存结果**：如果修改了 Campaign，确保保存到缓存
5. **返回有意义的结果**：使用 `result` 变量返回执行结果

### ❌ 避免的做法
1. **不要过度使用**：不要用代码执行替代所有标准工具
2. **不要重新导入已提供的 BayBE 类**：
   - 运行环境中已经预先导入了常用的 BayBE 类，并放在全局命名空间中
   - 例如可以**直接**写：
     ```python
     objective = ParetoObjective(targets=[...])
     constraint = ContinuousLinearConstraint(...)
     ```
     而**不要**再写：
     ```python
     from baybe.objective import ParetoObjective      # ❌ 当前版本没有这个模块
     from baybe.objectives import ParetoObjective     # 在本工具环境中也不需要
     from baybe.constraints import SumConstraint      # ❌ 请使用 DiscreteSumConstraint / ContinuousLinearConstraint
     from baybe.constraints import LinearInequalityConstraint  # ❌
     ```
   - 支持直接使用的类包括（无需显式 import）：
     - `DesirabilityObjective`, `ParetoObjective`
     - `DiscreteSumConstraint`, `ContinuousLinearConstraint`
     - 以及 `Campaign`, 各种 Parameter / Target / SearchSpace 类等
3. **不要忽略错误**：确保代码能处理异常情况
4. **不要忘记保存**：修改 Campaign 后记得保存
5. **不要执行危险操作**：遵守安全限制

## 连续/离散参数的控制规则

当系统从 Enhanced Verification 给出参数建议时，**离散/连续的最终选择以用户配置为准**：

- **强制连续**：在 `custom_parameter_bounds` 中提供 `min/max`，系统不会再根据建议强制离散。  
  例：`{"SubstanceA_ratio": {"min": 0.6, "max": 0.8}}`
- **强制离散**：在 `custom_parameter_bounds` 中提供 `values` 列表，系统会按离散参数处理。  
  例：`{"SubstanceA_ratio": {"values": [0.6, 0.7, 0.8]}}`
- **跟随建议**：仅在 `accept_suggested_parameters=true` 时采纳系统的离散建议。

> 注意：只有显式提供 `values` 列表才会被当作“用户强制离散”。

## 与标准工具的对比

| 特性 | 标准工具 | 代码执行工具 |
|------|---------|-------------|
| **灵活性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可靠性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **维护性** | ⭐⭐ | ⭐⭐⭐⭐ |
| **适用场景** | 标准工作流 | 特殊/实验性需求 |

## 总结

代码执行工具是对现有工具系统的补充，提供了更大的灵活性。建议：
- **80% 的情况**：使用标准工具（更可靠、更易用）
- **20% 的情况**：使用代码执行工具（特殊需求、实验性功能）

这样既保持了系统的稳定性，又提供了足够的灵活性来处理各种场景。

