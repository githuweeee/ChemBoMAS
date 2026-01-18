# BayBE 代码执行模板库（已弃用/存档）

这个文件包含常用的BayBE操作代码模板，可以直接使用或根据需求修改。

## 1. 基础Campaign构建

### 模板1: 从验证结果构建Campaign

```python
# 从验证结果和配置构建Campaign
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective
from baybe import Campaign

# 读取标准化数据
if standardized_data_path and os.path.exists(standardized_data_path):
    df = _read_csv_clean(standardized_data_path)
    
    # 创建参数列表
    parameters = []
    
    # 添加分子参数（从SMILES列）
    smiles_cols = [col for col in df.columns if 'SMILE' in col.upper()]
    for col in smiles_cols:
        valid_smiles = df[col].dropna().unique().tolist()
        if len(valid_smiles) >= 2:
            param = CategoricalParameter(
                name=f"{col.split('_')[0]}_molecule",
                values=valid_smiles,
                encoding="OHE"
            )
            parameters.append(param)
    
    # 添加数值参数
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    target_cols = [col for col in df.columns if col.startswith('Target_')]
    
    for col in numeric_cols:
        if col not in target_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            param = NumericalContinuousParameter(
                name=col,
                bounds=(min_val, max_val)
            )
            parameters.append(param)
    
    # 创建搜索空间
    searchspace = SearchSpace.from_product(parameters=parameters)
    
    # 创建目标（从optimization_config获取）
    targets = []
    if optimization_config:
        targets_config = optimization_config.get('targets', [])
        for target_config in targets_config:
            if isinstance(target_config, str):
                import json
                target_config = json.loads(target_config)
            target = NumericalTarget(
                name=target_config['name'],
                mode=target_config.get('mode', 'MAX')
            )
            targets.append(target)
    
    # 创建目标函数
    if len(targets) == 1:
        objective = DesirabilityObjective(targets=targets)
    elif len(targets) > 1:
        # 多目标：使用Pareto或Desirability
        strategy = optimization_config.get('optimization_strategy', 'pareto')
        if strategy == 'pareto':
            from baybe.objectives import ParetoObjective
            objective = ParetoObjective(targets=targets)
        else:
            objective = DesirabilityObjective(targets=targets)
    else:
        result = "错误：未找到优化目标配置"
    else:
        # 创建Campaign
        campaign = Campaign(searchspace=searchspace, objective=objective)
        
        # 保存到缓存
        _save_campaign_to_cache(session_id, campaign)
        
        result = f"Campaign创建成功！\n参数数量: {len(parameters)}\n目标数量: {len(targets)}"
else:
    result = "错误：未找到标准化数据文件"
```

## 2. 推荐生成

### 模板2: 生成推荐并格式化

```python
if campaign:
    # 生成推荐
    batch_size = optimization_config.get('batch_size', 5)
    recommendations = campaign.recommend(batch_size=batch_size)
    
    # 格式化输出
    print(f"生成了 {len(recommendations)} 个推荐")
    print("\n推荐内容:")
    print(recommendations.to_string())
    
    # 保存到文件
    session_dir = state.get("session_dir", ".")
    output_path = os.path.join(session_dir, f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    result = {
        'recommendations': recommendations.to_dict('records'),
        'output_path': output_path,
        'count': len(recommendations)
    }
else:
    result = "错误：Campaign不存在，请先构建Campaign"
```

### 模板3: 使用自定义推荐器

```python
if campaign and 'BotorchRecommender' in globals():
    from baybe.recommenders import BotorchRecommender
    
    # 创建自定义推荐器
    recommender = BotorchRecommender()
    
    # 获取当前测量数据
    measurements = campaign.measurements if hasattr(campaign, 'measurements') else pd.DataFrame()
    
    # 生成推荐
    recommendations = recommender.recommend(
        batch_size=5,
        searchspace=campaign.searchspace,
        objective=campaign.objective,
        measurements=measurements
    )
    
    result = recommendations
else:
    result = "错误：Campaign或推荐器不可用"
```

## 3. 数据处理

### 模板4: 读取和处理数据

```python
# 读取标准化数据
if standardized_data_path and os.path.exists(standardized_data_path):
    df = _read_csv_clean(standardized_data_path)
    
    # 数据统计
    stats = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'smiles_columns': [col for col in df.columns if 'SMILE' in col.upper()],
        'target_columns': [col for col in df.columns if col.startswith('Target_')],
        'numeric_columns': df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    }
    
    # 目标列统计
    target_stats = {}
    for col in stats['target_columns']:
        target_stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }
    
    result = {
        'data_stats': stats,
        'target_stats': target_stats
    }
else:
    result = "错误：数据文件不存在"
```

### 模板5: 添加测量数据

```python
if campaign:
    # 准备测量数据（示例）
    # 实际使用时，应该从用户上传的结果中读取
    measurements_data = {
        'SubstanceA_molecule': ['CCO', 'CCCO'],  # SMILES
        'temperature': [100, 120],
        'Target_yield': [85.5, 90.2]
    }
    
    measurements_df = pd.DataFrame(measurements_data)
    
    # 添加到Campaign
    campaign.add_measurements(measurements_df)
    
    # 保存更新后的Campaign
    _save_campaign_to_cache(session_id, campaign)
    
    result = f"成功添加 {len(measurements_df)} 条测量数据"
else:
    result = "错误：Campaign不存在"
```

## 4. 分析和可视化

### 模板6: Campaign信息分析

```python
if campaign:
    info = {
        'searchspace_size': campaign.searchspace.size,
        'parameter_names': campaign.searchspace.parameter_names,
        'num_parameters': len(campaign.searchspace.parameter_names),
        'objective_type': type(campaign.objective).__name__,
    }
    
    # 如果有测量数据
    if hasattr(campaign, 'measurements') and campaign.measurements is not None:
        info['num_measurements'] = len(campaign.measurements)
        info['has_measurements'] = True
    else:
        info['num_measurements'] = 0
        info['has_measurements'] = False
    
    result = info
else:
    result = "错误：Campaign不存在"
```

### 模板7: 优化进度分析

```python
if campaign and hasattr(campaign, 'measurements'):
    measurements = campaign.measurements
    
    if len(measurements) > 0:
        # 分析目标值的变化
        target_cols = [col for col in measurements.columns if col.startswith('Target_')]
        
        progress = {}
        for target_col in target_cols:
            values = measurements[target_col].dropna()
            if len(values) > 0:
                progress[target_col] = {
                    'best': float(values.max()) if 'MAX' in str(campaign.objective) else float(values.min()),
                    'latest': float(values.iloc[-1]),
                    'mean': float(values.mean()),
                    'improvement': float(values.iloc[-1] - values.iloc[0]) if len(values) > 1 else 0
                }
        
        result = {
            'total_experiments': len(measurements),
            'target_progress': progress
        }
    else:
        result = "还没有测量数据"
else:
    result = "错误：Campaign或测量数据不存在"
```

## 5. 高级操作

### 模板8: 自定义约束

```python
from baybe.constraints import DiscreteSumConstraint
from baybe.constraints.conditions import ThresholdCondition

# 创建约束：例如，两个参数的和必须等于1
if campaign:
    param_names = campaign.searchspace.parameter_names
    
    # 示例：创建比例约束
    # 假设有ratio_A和ratio_B两个参数，它们的和必须等于1
    if 'ratio_A' in param_names and 'ratio_B' in param_names:
        constraint = DiscreteSumConstraint(
            parameters=['ratio_A', 'ratio_B'],
            condition=ThresholdCondition(threshold=1.0, operator='==')
        )
        
        # 注意：约束需要在创建SearchSpace时添加
        # 这里只是示例，实际使用时需要在构建时添加
        result = "约束创建成功（需要在构建SearchSpace时添加）"
    else:
        result = "未找到所需的参数"
else:
    result = "错误：Campaign不存在"
```

### 模板9: 批量实验推荐

```python
if campaign:
    # 生成多批推荐
    batch_sizes = [5, 5, 5]  # 3批，每批5个
    all_recommendations = []
    
    for i, batch_size in enumerate(batch_sizes):
        recommendations = campaign.recommend(batch_size=batch_size)
        recommendations['batch_number'] = i + 1
        all_recommendations.append(recommendations)
    
    # 合并所有推荐
    combined = pd.concat(all_recommendations, ignore_index=True)
    
    # 保存
    output_path = os.path.join(session_dir, "batch_recommendations.csv")
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    result = {
        'total_recommendations': len(combined),
        'batches': len(batch_sizes),
        'output_path': output_path
    }
else:
    result = "错误：Campaign不存在"
```

## 使用说明

1. **复制模板**：选择适合的模板，复制代码
2. **修改参数**：根据实际需求修改参数和逻辑
3. **执行代码**：使用标准工具流程执行（当前版本禁用 `execute_baybe_code`）
4. **检查结果**：查看返回结果，确认操作成功

## 注意事项

- 所有模板都假设必要的上下文变量已存在（campaign, state等）
- 某些操作需要Campaign已存在
- 数据文件路径需要验证存在性
- 建议先测试简单操作，再尝试复杂逻辑

