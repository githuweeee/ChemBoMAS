# BayBE 迁移实施指南

## 概述
本文档为从 EDBO+ 迁移到 BayBE 提供详细的技术指南和实施路线图。

---

## BayBE 技术调研

### 项目信息
- **GitHub**: https://github.com/emdgroup/baybe
- **文档**: https://emdgroup.github.io/baybe/
- **开发商**: EMD Group (默克集团)
- **许可证**: Apache 2.0

### 核心特性
1. **现代化架构**: 基于 PyTorch 和 BoTorch
2. **多目标优化**: 原生支持多目标贝叶斯优化
3. **灵活约束**: 支持线性和非线性约束
4. **丰富的获取函数**: qEI, qNEI, qPI, qUCB 等
5. **参数类型多样**: 连续、离散、分类参数
6. **实验设计**: 支持 DoE 和主动学习

---

## EDBO+ vs BayBE 对比分析

| 特性 | EDBO+ | BayBE | 迁移影响 |
|------|-------|-------|----------|
| 基础框架 | 自定义 | PyTorch/BoTorch | 需要重写核心逻辑 |
| 多目标优化 | 支持 | 原生支持 | 更好的多目标处理 |
| 约束处理 | 基础 | 高级 | 需要约束重新定义 |
| 参数类型 | 有限 | 丰富 | 更灵活的参数定义 |
| 社区支持 | 较少 | 活跃 | 更好的长期维护 |
| 学习曲线 | 中等 | 中等 | 需要团队培训 |

---

## 迁移技术实现

### 1. BayBE 环境配置

#### 安装依赖
```bash
# 添加到 requirements.txt
baybe
torch
botorch
gpytorch
```

#### 基础配置检查
```python
def check_baybe_installation():
    """验证 BayBE 安装和配置"""
    try:
        import baybe
        from baybe import Campaign
        print(f"BayBE 版本: {baybe.__version__}")
        return True
    except ImportError as e:
        print(f"BayBE 安装失败: {e}")
        return False
```

### 2. 参数转换模块

#### EDBO+ 到 BayBE 参数映射
```python
from baybe.parameters import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter, 
    CategoricalParameter
)
from baybe.constraints import LinearConstraint
from baybe.objectives import NumericalTarget

class ParameterConverter:
    """EDBO+ 参数格式转换为 BayBE 格式"""
    
    def __init__(self):
        self.parameter_mapping = {
            'continuous': NumericalContinuousParameter,
            'discrete': NumericalDiscreteParameter,
            'categorical': CategoricalParameter
        }
    
    def convert_parameters(self, edbo_config: dict) -> list:
        """转换参数定义"""
        baybe_parameters = []
        
        for param in edbo_config.get('parameters', []):
            param_name = param['name']
            param_type = param['type']
            
            if param_type == 'continuous':
                baybe_param = NumericalContinuousParameter(
                    name=param_name,
                    bounds=param['bounds'],
                    tolerance=param.get('tolerance', 0.01)
                )
            elif param_type == 'categorical':
                baybe_param = CategoricalParameter(
                    name=param_name,
                    values=param['values'],
                    encoding=param.get('encoding', 'OHE')
                )
            elif param_type == 'discrete':
                baybe_param = NumericalDiscreteParameter(
                    name=param_name,
                    values=param['values'],
                    tolerance=param.get('tolerance', 0.01)
                )
            
            baybe_parameters.append(baybe_param)
        
        return baybe_parameters
    
    def convert_objectives(self, edbo_config: dict) -> list:
        """转换目标函数定义"""
        baybe_objectives = []
        
        for obj in edbo_config.get('objectives', []):
            objective = NumericalTarget(
                name=obj['name'],
                mode=obj['mode'].upper(),  # "MAXIMIZE" 或 "MINIMIZE"
                bounds=obj.get('bounds'),
                tolerance=obj.get('tolerance', 0.01)
            )
            baybe_objectives.append(objective)
        
        return baybe_objectives
    
    def convert_constraints(self, edbo_config: dict) -> list:
        """转换约束条件"""
        baybe_constraints = []
        
        for const in edbo_config.get('constraints', []):
            if const['type'] == 'LinearConstraint':
                constraint = LinearConstraint(
                    parameters=const['parameters'],
                    coefficients=const['coefficients'],
                    rhs=const['rhs'],
                    operator=const['operator']
                )
                baybe_constraints.append(constraint)
        
        return baybe_constraints
```

### 3. BayBE 推荐引擎

```python
import pandas as pd
from baybe import Campaign
from baybe.recommenders import (
    FPSRecommender,
    TwoPhaseMetaRecommender,
    RandomRecommender
)

class BayBERecommendationEngine:
    """BayBE 推荐引擎实现"""
    
    def __init__(self, optimization_config: dict):
        self.config = optimization_config
        self.converter = ParameterConverter()
        self.campaign = self._create_campaign()
        
    def _create_campaign(self) -> Campaign:
        """创建 BayBE Campaign"""
        parameters = self.converter.convert_parameters(self.config)
        objectives = self.converter.convert_objectives(self.config)
        constraints = self.converter.convert_constraints(self.config)
        
        # 选择推荐器
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=FPSRecommender(),
            recommender=RandomRecommender()  # 在有足够数据后切换
        )
        
        campaign = Campaign(
            parameters=parameters,
            objectives=objectives,
            constraints=constraints,
            recommender=recommender
        )
        
        return campaign
    
    def recommend_experiments(self, batch_size: int = 5) -> pd.DataFrame:
        """生成实验推荐"""
        recommendations = self.campaign.recommend(batch_size=batch_size)
        
        # 添加预测信息
        if len(self.campaign.measurements) > 0:
            predictions = self.campaign.posterior.predict(recommendations)
            recommendations = pd.concat([recommendations, predictions], axis=1)
        
        return recommendations
    
    def add_measurements(self, experiments: pd.DataFrame):
        """添加实验测量结果"""
        self.campaign.add_measurements(experiments)
        
        # 自动切换推荐器
        if len(self.campaign.measurements) >= 10:
            from baybe.recommenders import BotorchRecommender
            self.campaign.recommender = BotorchRecommender()
    
    def get_optimization_progress(self) -> dict:
        """获取优化进展信息"""
        if len(self.campaign.measurements) == 0:
            return {"status": "No measurements yet"}
        
        measurements = self.campaign.measurements
        objectives = [obj.name for obj in self.campaign.objectives]
        
        progress = {
            "total_experiments": len(measurements),
            "best_values": {},
            "improvement_trend": {},
            "convergence_status": "In Progress"
        }
        
        for obj_name in objectives:
            if obj_name in measurements.columns:
                values = measurements[obj_name].dropna()
                if len(values) > 0:
                    progress["best_values"][obj_name] = {
                        "current_best": values.max(),  # 假设最大化
                        "initial": values.iloc[0],
                        "improvement": values.max() - values.iloc[0]
                    }
        
        return progress
```

### 4. 对比测试框架

```python
class EDBOvsBayBEComparison:
    """EDBO+ 与 BayBE 性能对比"""
    
    def __init__(self, test_dataset: pd.DataFrame, config: dict):
        self.test_data = test_dataset
        self.config = config
        self.results = {}
    
    def run_benchmark(self, n_iterations: int = 20, n_repeats: int = 5):
        """运行基准测试"""
        
        edbo_results = []
        baybe_results = []
        
        for repeat in range(n_repeats):
            print(f"运行第 {repeat + 1} 轮对比测试...")
            
            # 测试 EDBO+
            edbo_result = self._test_edbo_plus(n_iterations)
            edbo_results.append(edbo_result)
            
            # 测试 BayBE  
            baybe_result = self._test_baybe(n_iterations)
            baybe_results.append(baybe_result)
        
        # 汇总结果
        self.results = {
            "edbo_plus": self._aggregate_results(edbo_results),
            "baybe": self._aggregate_results(baybe_results),
            "comparison": self._compare_results(edbo_results, baybe_results)
        }
        
        return self.results
    
    def _test_baybe(self, n_iterations: int) -> dict:
        """测试 BayBE 性能"""
        engine = BayBERecommendationEngine(self.config)
        
        results = {
            "iterations": [],
            "best_values": [],
            "convergence_rate": [],
            "time_per_iteration": []
        }
        
        # 初始化实验
        initial_experiments = self.test_data.sample(5)
        engine.add_measurements(initial_experiments)
        
        for i in range(n_iterations):
            start_time = time.time()
            
            # 生成推荐
            recommendations = engine.recommend_experiments(batch_size=1)
            
            # 模拟实验（从测试数据中选择最接近的点）
            simulated_result = self._simulate_experiment(recommendations.iloc[0])
            
            # 添加结果
            engine.add_measurements(simulated_result)
            
            # 记录指标
            elapsed_time = time.time() - start_time
            progress = engine.get_optimization_progress()
            
            results["iterations"].append(i + 1)
            results["time_per_iteration"].append(elapsed_time)
            results["best_values"].append(progress["best_values"])
            
        return results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        if not self.results:
            return "请先运行基准测试"
        
        report = f"""
# EDBO+ vs BayBE 性能对比报告

## 测试配置
- 迭代次数: {len(self.results['edbo_plus']['iterations'])}
- 重复次数: 5
- 测试数据集大小: {len(self.test_data)}

## 性能指标对比

### 收敛速度
- EDBO+ 平均收敛时间: {self.results['comparison']['convergence_time']['edbo_plus']:.2f} 迭代
- BayBE 平均收敛时间: {self.results['comparison']['convergence_time']['baybe']:.2f} 迭代

### 最终优化效果  
- EDBO+ 最佳目标值: {self.results['comparison']['final_best']['edbo_plus']:.4f}
- BayBE 最佳目标值: {self.results['comparison']['final_best']['baybe']:.4f}

### 计算效率
- EDBO+ 平均每次迭代时间: {self.results['comparison']['avg_time']['edbo_plus']:.2f}s
- BayBE 平均每次迭代时间: {self.results['comparison']['avg_time']['baybe']:.2f}s

## 推荐
{self._generate_recommendation()}
        """
        
        return report
    
    def _generate_recommendation(self) -> str:
        """基于测试结果生成迁移建议"""
        baybe_better_convergence = (
            self.results['comparison']['convergence_time']['baybe'] < 
            self.results['comparison']['convergence_time']['edbo_plus']
        )
        baybe_better_final = (
            self.results['comparison']['final_best']['baybe'] > 
            self.results['comparison']['final_best']['edbo_plus']
        )
        baybe_faster = (
            self.results['comparison']['avg_time']['baybe'] < 
            self.results['comparison']['avg_time']['edbo_plus']
        )
        
        score = sum([baybe_better_convergence, baybe_better_final, baybe_faster])
        
        if score >= 2:
            return "**推荐迁移到 BayBE**: 在多个关键指标上表现更优"
        elif score == 1:
            return "**谨慎考虑迁移**: BayBE 在某些方面更优，建议进一步测试"
        else:
            return "**暂不推荐迁移**: EDBO+ 在当前场景下表现更好"
```

---

## 迁移实施步骤

### 阶段1: 环境准备 (1周)
1. **依赖安装**
   ```bash
   pip install baybe torch botorch
   ```

2. **基础测试**
   ```python
   # 验证安装
   python -c "import baybe; print(baybe.__version__)"
   ```

3. **示例运行**
   - 运行 BayBE 官方示例
   - 测试基本功能

### 阶段2: 适配器开发 (2周)
1. **参数转换器**
   - 实现 `ParameterConverter` 类
   - 单元测试验证转换正确性

2. **推荐引擎**
   - 实现 `BayBERecommendationEngine` 类
   - 集成测试验证功能

3. **状态管理**
   - 适配现有状态管理系统
   - 确保向后兼容

### 阶段3: 对比测试 (2周)
1. **基准测试**
   - 实现对比测试框架
   - 运行性能基准测试

2. **结果分析**
   - 分析测试结果
   - 生成迁移建议报告

3. **风险评估**
   - 识别潜在问题
   - 制定缓解策略

### 阶段4: 渐进式集成 (3周)
1. **并行运行**
   - 实现混合推荐器
   - 支持运行时切换

2. **用户测试**
   - 邀请用户测试新功能
   - 收集反馈和改进建议

3. **性能监控**
   - 部署监控系统
   - 跟踪关键指标

### 阶段5: 完全迁移 (1周)
1. **默认切换**
   - 将 BayBE 设为默认引擎
   - 保留 EDBO+ 作为备选

2. **文档更新**
   - 更新用户文档
   - 更新开发文档

3. **团队培训**
   - 培训开发团队
   - 制作使用指南

---

## 风险管理

### 技术风险
1. **性能下降**
   - 风险: BayBE 可能在某些场景下性能不如 EDBO+
   - 缓解: 保留并行运行能力，支持动态切换

2. **兼容性问题**
   - 风险: 现有数据格式可能不兼容
   - 缓解: 实现完善的数据转换层

3. **学习成本**
   - 风险: 团队需要时间学习新工具
   - 缓解: 提供培训和详细文档

### 业务风险
1. **用户体验影响**
   - 风险: 迁移过程可能影响用户体验
   - 缓解: 渐进式迁移，保持功能向后兼容

2. **项目延期**
   - 风险: 迁移可能影响其他开发计划
   - 缓解: 合理安排时间，并行开发

---

## 成功指标

### 技术指标
- [ ] BayBE 集成成功率 = 100%
- [ ] 性能对比测试完成
- [ ] 向后兼容性验证通过
- [ ] 单元测试覆盖率 > 90%

### 性能指标
- [ ] 优化效果提升 > 5%
- [ ] 收敛速度提升 > 10%
- [ ] 系统响应时间 < 30秒

### 用户指标
- [ ] 用户满意度 > 4.0/5.0
- [ ] 功能完整性 = 100%
- [ ] 错误率 < 1%

---

## 后续维护计划

### 持续监控
1. 性能指标监控
2. 错误日志分析
3. 用户反馈收集

### 定期更新
1. BayBE 版本升级
2. 功能增强开发
3. 安全补丁应用

### 社区参与
1. 贡献 BayBE 开源项目
2. 分享使用经验
3. 参与技术讨论

---

## 参考资源

### 官方文档
- [BayBE GitHub](https://github.com/emdgroup/baybe)
- [BayBE 文档](https://emdgroup.github.io/baybe/)
- [BoTorch 文档](https://botorch.org/)

### 学习资源
- BayBE 教程和示例
- 贝叶斯优化理论基础
- PyTorch 深度学习框架

### 社区支持
- BayBE Issues 和讨论区
- 相关技术论坛
- 学术会议和论文
