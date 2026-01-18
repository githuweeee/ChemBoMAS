# ChemBoMAS 说明文档

## App简介

ChemBoMAS (Chemical Bayesian Optimization Multi-Agent System) 是一个基于Google Agent Development Kit (ADK) 和BayBE贝叶斯优化框架构建的智能化学实验优化系统。该系统通过多智能体协作，实现从数据验证、实验推荐、结果分析到持续优化的完整闭环工作流程。

### 开发背景

在化学实验优化领域，传统的实验设计方法往往需要大量的试错实验，成本高、周期长。随着人工智能和贝叶斯优化技术的发展，智能化的实验设计成为可能。ChemBoMAS应运而生，旨在：

1. **降低实验成本**：通过贝叶斯优化算法，以最少的实验次数找到最优条件
2. **提高优化效率**：利用多智能体系统实现自动化的实验设计和结果分析
3. **简化操作流程**：提供友好的交互界面，降低使用门槛
4. **集成领域知识**：内置化学知识库，提供专业的参数建议和安全约束

### 应用能力

ChemBoMAS系统具备以下核心能力：

1. **智能数据验证**：自动验证实验数据的完整性和SMILES分子结构的有效性
2. **参数智能建议**：基于化学知识库，为实验参数提供合理的边界建议
3. **贝叶斯优化推荐**：使用BayBE框架进行高效的实验条件推荐
4. **多目标优化**：支持单目标、多目标和帕累托前沿优化
5. **迭代优化管理**：自动管理多轮实验的迭代优化流程
6. **结果分析与可视化**：提供模型性能分析和专业图表生成
7. **收敛分析与阶段建议**：基于优化进展给出阶段性建议与停止提示

系统特别适用于：
- 环氧固化反应优化
- 聚合反应条件优化
- 催化合成反应优化
- 材料配方优化
- 工艺参数优化

---

## 最佳实践

### Example 1: 环氧固化反应优化

**场景描述**：优化环氧树脂固化反应，目标是同时提高玻璃化转变温度(Tg)和冲击强度。

**输入**：
```csv
SubstanceA_name,SubstanceA_SMILE,SubstanceA_ratio,SubstanceB_name,SubstanceB_SMILE,SubstanceB_ratio,Target_alpha_tg,Target_beta_impactstrength
南亚127e,CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4,0.6,固化剂1,5胺类,NCCCCCN,0.3,80,110
南亚127e,CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4,0.7,固化剂1,5胺类,NCCCCCN,0.2,90,100
南亚127e,CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4,0.8,固化剂1,5胺类,NCCCCCN,0.1,60,120
```

**操作流程**：
1. 上传初始实验数据CSV文件
2. 系统自动验证数据质量和SMILES有效性
3. 系统基于化学知识库提供参数边界建议（如：固化剂比例范围0.1-0.4，温度范围60-120°C）
4. 用户确认优化目标：最大化Tg和冲击强度
5. 系统生成第一轮实验推荐（5个实验条件）

**输出**：
```
推荐实验条件（第1轮）：
实验1: SubstanceA_ratio=0.65, SubstanceB_ratio=0.25, 预期Tg=85°C, 预期冲击强度=115
实验2: SubstanceA_ratio=0.70, SubstanceB_ratio=0.20, 预期Tg=88°C, 预期冲击强度=105
实验3: SubstanceA_ratio=0.75, SubstanceB_ratio=0.15, 预期Tg=75°C, 预期冲击强度=125
实验4: SubstanceA_ratio=0.60, SubstanceB_ratio=0.30, 预期Tg=82°C, 预期冲击强度=108
实验5: SubstanceA_ratio=0.68, SubstanceB_ratio=0.22, 预期Tg=87°C, 预期冲击强度=112

已生成结果模板文件: result_template_xxx.csv
请按照推荐条件进行实验，并填写测量结果。
```

**后续迭代**：
- 用户完成实验并上传结果
- 系统自动更新BayBE Campaign
- 生成下一轮推荐（基于新数据优化）
- 持续迭代直到收敛或达到目标

---

### Example 2: 多目标帕累托优化

**场景描述**：优化材料配方，需要平衡性能（最大化）和成本（最小化）两个冲突目标。

**输入**：
```csv
SubstanceA_name,SubstanceA_SMILE,SubstanceA_ratio,SubstanceB_name,SubstanceB_SMILE,SubstanceB_ratio,Target_performance,Target_cost
材料A,CC(C)O,0.5,材料B,NCCCN,0.5,85.0,25.5
材料A,CC(C)O,0.6,材料B,NCCCN,0.4,90.0,28.0
材料A,CC(C)O,0.7,材料B,NCCCN,0.3,88.0,30.5
```

**操作流程**：
1. 上传数据并设置多目标优化模式
2. 配置优化目标：
   - 目标1：性能（MAX，权重0.6）
   - 目标2：成本（MIN，权重0.4）
3. 选择帕累托优化策略
4. 系统生成推荐

**输出**：
```
帕累托前沿分析：
已识别3个帕累托最优解：

解1（高性能）：SubstanceA_ratio=0.72, SubstanceB_ratio=0.28
  - 预测性能: 92.5
  - 预测成本: 32.0
  - 权衡评分: 0.68

解2（平衡）：SubstanceA_ratio=0.65, SubstanceB_ratio=0.35
  - 预测性能: 89.0
  - 预测成本: 28.5
  - 权衡评分: 0.72

解3（低成本）：SubstanceA_ratio=0.58, SubstanceB_ratio=0.42
  - 预测性能: 86.5
  - 预测成本: 26.0
  - 权衡评分: 0.75

推荐实验：解2（平衡方案），在性能和成本之间取得最佳平衡。
```

---

### Example 3: 特殊物质处理（无SMILES的稀释剂）

**场景描述**：配方中包含无法用SMILES表示的稀释剂或专有添加剂，需要系统智能处理。

**输入**：
```csv
SubstanceA_name,SubstanceA_SMILE,SubstanceA_ratio,SubstanceB_name,SubstanceB_SMILE,SubstanceB_ratio,SubstanceC_name,SubstanceC_SMILE,SubstanceC_ratio,Target_yield
环氧树脂,CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4,0.6,固化剂,NCCCCCN,0.3,稀释剂A,,0.1,87.5
环氧树脂,CC(C)(C1=CC=C(C=C1)OCC2CO2)C3=CC=C(C=C3)OCC4CO4,0.7,固化剂,NCCCCCN,0.2,稀释剂B,,0.1,89.2
```

**操作流程**：
1. 上传包含空SMILES列的数据（稀释剂A和稀释剂B的SMILES列为空）
2. 系统自动识别特殊物质（有名称但无SMILES）
3. 系统为特殊物质创建分类参数（CategoricalParameter）
4. 使用物质名称作为编码值进行优化

**输出**：
```
数据验证结果：
✓ 已验证2个标准物质（有SMILES）
✓ 识别到2个特殊物质：稀释剂A, 稀释剂B
✓ 特殊物质将使用名称编码进行优化

参数配置：
- SubstanceA_molecule: CategoricalParameter (SMILES编码)
- SubstanceB_molecule: CategoricalParameter (SMILES编码)  
- SubstanceC_molecule: CategoricalParameter (名称编码: ["稀释剂A", "稀释剂B"])

推荐实验条件：
实验1: SubstanceA_ratio=0.65, SubstanceB_ratio=0.25, SubstanceC=稀释剂A, SubstanceC_ratio=0.10
实验2: SubstanceA_ratio=0.68, SubstanceB_ratio=0.22, SubstanceC=稀释剂B, SubstanceC_ratio=0.10
实验3: SubstanceA_ratio=0.70, SubstanceB_ratio=0.20, SubstanceC=稀释剂A, SubstanceC_ratio=0.10
```

---

### Example 4: 自适应策略优化

**场景描述**：系统根据优化进展自动调整推荐策略，从探索阶段转向利用阶段。

**输入**：已完成3轮实验，共15个数据点

**操作流程**：
1. 系统分析优化进展
2. 检测到改进率下降（从15%降至3%）
3. 自动切换策略：从探索（qEI）转向利用（qUCB）
4. 调整批次大小：从5个降至3个（精细化优化）

**输出**：
```
优化进展分析（第3轮后）：

当前状态：
- 已完成实验: 15个
- 当前最优: Tg=92°C, 冲击强度=125
- 改进率: 3.2% (较上一轮下降)
- 收敛指标: 0.15 (接近收敛)

策略调整：
✓ 从"探索阶段"切换到"利用阶段"
✓ 获取函数: qEI → qUCB
✓ 批次大小: 5 → 3
✓ 聚焦区域: 当前最优解附近

推荐实验（精细化）：
实验1: SubstanceA_ratio=0.68, SubstanceB_ratio=0.22 (最优解附近)
实验2: SubstanceA_ratio=0.70, SubstanceB_ratio=0.20 (最优解附近)
实验3: SubstanceA_ratio=0.66, SubstanceB_ratio=0.24 (最优解附近)

建议：再完成1-2轮实验后，如改进率<2%，可考虑停止优化。
```

---

### Example 5: 结果分析与可视化

**场景描述**：完成多轮优化后，系统生成性能分析和可视化报告。

**输入**：已完成5轮优化，共25个实验数据点

**操作流程**：
1. 调用Fitting Agent进行结果分析
2. 训练代理模型（Random Forest）
3. 生成特征重要性分析
4. 创建可视化图表

**输出**：
```
优化结果分析报告：

模型性能指标：
- R² Score: 0.89
- RMSE: 4.2
- MAE: 3.1
- 交叉验证得分: 0.86

特征重要性排序：
1. SubstanceB_ratio: 0.35 (最重要)
2. SubstanceA_ratio: 0.28
3. Process_temperature: 0.18
4. SubstanceA_molecule: 0.12
5. Curing_time: 0.07

最优条件：
- SubstanceA_ratio: 0.68
- SubstanceB_ratio: 0.22
- Process_temperature: 85°C
- 预测Tg: 94°C
- 预测冲击强度: 128

已生成可视化文件：
- predicted_vs_actual_plot.png (预测vs实际值对比图)
- feature_importance_plot.png (特征重要性图)
- convergence_plot.png (收敛曲线图)
- pareto_front_plot.png (帕累托前沿图，如适用)

优化建议：
1. SubstanceB_ratio是最关键参数，建议在0.20-0.25范围内精细调整
2. 当前结果已接近收敛，继续优化的边际收益较小
3. 建议进行验证实验确认最优条件
```

---

### Example 6: 约束条件应用

**场景描述**：在优化过程中应用化学约束，如比例之和必须为1，温度不能超过安全上限。

**输入**：
```csv
SubstanceA_ratio,SubstanceB_ratio,SubstanceC_ratio,Process_temperature,Target_yield
0.6,0.3,0.1,80,87.5
0.7,0.2,0.1,90,89.2
0.5,0.4,0.1,75,85.0
```

**操作流程**：
1. 系统识别比例参数
2. 自动应用约束：SubstanceA_ratio + SubstanceB_ratio + SubstanceC_ratio = 1.0
3. 应用温度安全约束：Process_temperature ≤ 200°C
4. 在约束条件下进行优化

**输出**：
```
约束条件配置：

已应用约束：
1. 比例和约束: SubstanceA_ratio + SubstanceB_ratio + SubstanceC_ratio = 1.0
2. 温度上限约束: Process_temperature ≤ 200°C
3. 比例非负约束: 所有比例 ≥ 0

搜索空间信息：
- 有效搜索空间大小: 约10,000个组合
- 约束后可行域: 约8,500个组合
- 约束减少率: 15%

推荐实验（满足所有约束）：
实验1: SubstanceA_ratio=0.65, SubstanceB_ratio=0.25, SubstanceC_ratio=0.10, Process_temperature=85°C
实验2: SubstanceA_ratio=0.68, SubstanceB_ratio=0.22, SubstanceC_ratio=0.10, Process_temperature=88°C
实验3: SubstanceA_ratio=0.70, SubstanceB_ratio=0.20, SubstanceC_ratio=0.10, Process_temperature=82°C

所有推荐均满足约束条件。
```

---

## 参考文献

### 算法相关文献

1. **Bayesian Optimization**
   - Frazier, P. I. (2018). A tutorial on Bayesian optimization. *arXiv preprint arXiv:1807.02811*.
   - Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148-175.

2. **Multi-objective Optimization**
   - Emmerich, M. T., & Deutz, A. H. (2018). A tutorial on multiobjective optimization: fundamentals and evolutionary methods. *Natural computing*, 17(3), 585-609.
   - Knowles, J. (2006). ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. *IEEE Transactions on Evolutionary Computation*, 10(1), 50-66.

3. **Acquisition Functions**
   - Mockus, J. (1974). On Bayesian methods for seeking the extremum. *Optimization Techniques IFIP Technical Conference*, 400-404.
   - Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: no regret and experimental design. *ICML*.

4. **Gaussian Process Regression**
   - Rasmussen, C. E., & Williams, C. K. (2006). *Gaussian processes for machine learning*. MIT press.
   - Quiñonero-Candela, J., & Rasmussen, C. E. (2005). A unifying view of sparse approximate Gaussian process regression. *Journal of Machine Learning Research*, 6(Dec), 1939-1959.

5. **Active Learning**
   - Settles, B. (2009). Active learning literature survey. *University of Wisconsin-Madison Department of Computer Sciences*.
   - Cohn, D. A., Ghahramani, Z., & Jordan, M. I. (1996). Active learning with statistical models. *Journal of artificial intelligence research*, 4, 129-145.

### 化学信息学相关文献

6. **Molecular Descriptors**
   - Todeschini, R., & Consonni, V. (2008). *Handbook of molecular descriptors*. John Wiley & Sons.
   - Moriwaki, H., Tian, Y. S., Kawashita, N., & Takagi, T. (2018). Mordred: a molecular descriptor calculator. *Journal of cheminformatics*, 10(1), 1-14.

7. **SMILES Representation**
   - Weininger, D. (1988). SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. *Journal of chemical information and computer sciences*, 28(1), 31-36.
   - O'Boyle, N. M., Banck, M., James, C. A., Morley, C., Vandermeersch, T., & Hutchison, G. R. (2011). Open Babel: an open chemical toolbox. *Journal of cheminformatics*, 3(1), 1-14.

8. **Chemical Reaction Optimization**
   - Hase, F., Roch, L. M., & Aspuru-Guzik, A. (2018). Next-generation experimentation with self-driving laboratories. *Trends in Chemistry*, 1(3), 282-291.
   - Granda, J. M., Donina, L., Dragone, V., Long, D. L., & Cronin, L. (2018). Controlling an organic synthesis robot with machine learning to search for new reactivity. *Nature*, 559(7714), 377-381.

### 系统开发相关文献

9. **Multi-Agent Systems**
   - Wooldridge, M. (2009). *An introduction to multiagent systems*. John Wiley & Sons.
   - Stone, P., & Veloso, M. (2000). Multiagent systems: A survey from a machine learning perspective. *Autonomous Robots*, 8(3), 345-383.

10. **BayBE Framework**
    - BayBE Documentation: https://emdgroup.github.io/baybe/
    - BayBE GitHub Repository: https://github.com/emdgroup/baybe
    - EMD Group. (2024). BayBE: Bayesian Optimization for Black-box Experiments. *Merck KGaA*.

11. **Google Agent Development Kit**
    - Google ADK Documentation: https://developers.google.com/adk
    - Google. (2024). Agent Development Kit: Building AI Agents with Google's Framework.

12. **Experimental Design**
    - Box, G. E., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for experimenters: design, innovation, and discovery* (Vol. 2). Wiley-Interscience.
    - Montgomery, D. C. (2017). *Design and analysis of experiments*. John Wiley & Sons.

13. **Machine Learning in Chemistry**
    - Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., & Walsh, A. (2018). Machine learning for molecular and materials science. *Nature*, 559(7715), 547-555.
    - Raccuglia, P., Elbert, K. C., Adler, D. F., Falk, C., Wenny, M. B., Mollo, A., ... & Norquist, A. J. (2016). Machine-learning-assisted materials discovery using failed experiments. *Nature*, 533(7601), 73-76.

14. **Adaptive Experimentation**
    - Kandasamy, K., Dasarathy, G., Oliva, J. B., Schneider, J., & Póczos, B. (2017). Gaussian process bandit optimisation with multi-fidelity evaluations. *Advances in neural information processing systems*, 30.
    - Hernández-Lobato, J. M., Gelbart, M. A., Hoffman, M. W., Adams, R. P., & Ghahramani, Z. (2015). Predictive entropy search for Bayesian optimization with unknown constraints. *International conference on machine learning*, 1699-1707.

---

*文档版本: v1.0*  
*最后更新: 2025年1月*  
*项目地址: ChemBoMAS*

