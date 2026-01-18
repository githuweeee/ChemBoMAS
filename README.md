# ChemBoMAS Agent

This agent is designed to assist in chemical experiment optimization. It uses a multi-agent system built with the Google Agent Development Kit (ADK) to guide a user through a cycle of data submission, feature engineering, experiment recommendation, and results analysis.

## 架构设计亮点

### 智能参数边界推荐系统

ChemBoMAS采用了一套科学的参数边界推荐架构，明确区分了各组件的职责：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        边界推荐职责分配                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │   知识库 (KB)     │    │   计算工具（todo）│    │    LLM       │   │
│  │                  │    │                  │    │               │  │
│  │ • 反应类型规则    │    │ • RDKit分子计算   │    │ • 理解用户意图 │  │
│  │ • 典型温度范围    │    │ • 物质属性查询    │    │ • 选择KB条目   │  │
│  │ • 安全约束        │    │ • 数据范围分析    │    │ • 整合信息    │   │
│  │ • 不兼容组合      │    │                  │    │ • 生成建议    │   │
│  └────────┬─────────┘    └────────┬─────────┘    └──────┬───────┘   │
│           │                       │                      │          │
│           └───────────────────────┴──────────────────────┘          │
│                                   ↓                                 │
│                        ┌──────────────────┐                         │
│                        │  用户确认/修改    │                         │
│                        │  (领域专家拍板)   │                         │
│                        └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

#### 核心设计原则

1. **知识库提供硬约束**：`ChemistryKnowledgeBase` 包含了反应类型规则、典型参数范围、安全约束等专业知识
   - 例如：环氧固化反应典型温度 60-120°C，安全上限 200°C
   - 催化剂典型浓度范围 0.1%-10%

2. **计算工具提供物质属性**：RDKit用于分子结构验证和属性计算
   - SMILES规范化和验证
   - 分子描述符计算（由BayBE内部处理）

3. **LLM负责整合和交互**：理解用户意图，选择合适的知识库条目
   - **注意**：LLM不适合直接推演精确的扩展百分比
   - LLM的角色是"整合者"而非"计算者"

4. **用户最终确认**：所有建议都需要领域专家确认
   - 系统提供 `requires_user_confirmation: True` 标记
   - 用户可以接受或修改建议范围

#### 边界建议流程

```
用户上传初期实验数据
        ↓
IntelligentParameterAdvisor 分析数据
        ↓
从 ChemistryKnowledgeBase 获取反应类型的典型范围
        ↓
结合当前数据范围生成"建议边界"（取并集扩大探索空间）
        ↓
生成带有理由说明的建议（source: "knowledge_base"）
        ↓
用户确认或修改边界
        ↓
确认后的边界传给 BayBE 构建 SearchSpace
```

#### 关键实现文件

- `chemistry_knowledge_base.py`：化学领域知识库，包含反应类型、材料属性、安全约束
- `enhanced_verification_tools.py`：`IntelligentParameterAdvisor` 类，整合知识库进行智能边界推荐

#### BayBE SearchSpace 说明

**重要**：BayBE不会自动推断参数边界，必须显式提供 `bounds` 参数：

```python
NumericalContinuousParameter(
    name="Temperature",
    bounds=(60, 120),  # 必须显式提供
)
```

因此，系统的边界建议最终需要用户确认后才能用于BayBE。

## 工作流程

本系统（Agent）主要遵循以下步骤进行化学实验优化：

1.  **数据上传**：用户上传包含化学反应信息的CSV文件。该文件应包含各物质名称、SMILES结构、比例参数以及实验目标值（如产率、成本等）。

2.  **数据验证**：系统首先验证上传数据格式是否合法，检查各物质和目标的列名是否符合规范。如果表头有自然语言描述（如非机器可识别字段），系统会中止并提示使用标准模板。

3.  **参数提取与搜索空间构建**：
    * 增强验证工具会校验SMILES格式，有效提取可调整的参数。
    * 分子描述符（molecular descriptor）的计算由BayBE主流程自动完成，无需用户手动生成。
    * 系统按照BayBE要求构建搜索空间并施加参数约束。
    * （补充：`generate_descriptor`为遗留工具，仅在需手动导出描述符时使用，依赖rdkit/mordred）

4.  **贝叶斯优化与实验推荐**：
    * 系统利用构建好的搜索空间，通过BayBE（Bayesian Optimization for Black-box Experiments）推荐下一批实验条件。
    * BayBE内部算法会自动处理特征优化与描述符计算。
    * 目标是以尽可能少的实验，找到最优实验组合，实现高效探索。

5.  **实验结果上传**：
    * 获取推荐条件后，使用`generate_result_template`生成标准上传模板，推荐直接打开session文件夹内的`experiment_log.csv`文件，按照推荐的条件实验，填写测量值。
    * 填写后，通过`upload_experimental_results`上传数据（支持文件路径或直接输入CSV内容，推荐直接上传`experiment_log.csv`文件，如在本地运行，请记得关闭CSV文件！！！），上传成功后，系统会自动校验上传格式，并完成BayBE Campaign的数据更新。
    * 系统会自动校验上传格式，并完成BayBE Campaign的数据更新。

6.  **分析与可视化**：
    * 用户上传结果后，`fitting`智能体会自动训练随机森林模型。
    * 系统会输出两类可视化图表：
        * **预测值与真实值对比图**（Predicted vs. Actual），用于判断模型精准度。
        * **特征重要性排名图**（Feature Importance），展示各参数对实验结果的影响程度。

通过上述循环，系统支持化学实验的持续迭代优化与自动学习。

## 安装与环境部署

运行本系统前，请先安装所需的Python依赖环境。

### Telemetry notice (BayBE)
- BayBE 默认会尝试将匿名运行指标上报到 `public.telemetry.baybe.p.uptimize.merckgroup.com:4317`，在内网/无权限环境下可能看到 `PERMISSION_DENIED` 日志，但不影响功能。
- 如需禁用遥测：启动前设置环境变量  
  - PowerShell: `$env:BAYBE_DISABLE_TELEMETRY="1"; adk web`  
  - 或在其他 shell 中 `export BAYBE_DISABLE_TELEMETRY=1` 后再启动。

### 启动配置选项

#### 默认启动（本地访问）
```bash
adk web
# 默认端口: 8000
# 默认地址: 127.0.0.1 (仅本机访问)
```

#### 指定端口启动
```bash
# 在指定端口启动（仍仅本机访问）
adk web --port 8080

# PowerShell
adk web --port 8080
```

#### 允许外部访问（局域网/远程）
```bash
# 绑定所有网络接口，允许外部访问
adk web --host 0.0.0.0 --port 8080

# PowerShell
adk web --host 0.0.0.0 --port 8080
```

#### 完整启动命令示例

**Windows (PowerShell)**:
```powershell
# 本地访问，端口 8000（默认）
adk web

# 本地访问，自定义端口
adk web --port 8080

# 允许外部访问，自定义端口
$env:BAYBE_DISABLE_TELEMETRY="1"
adk web --host 0.0.0.0 --port 8080
```

**Linux/macOS (Bash)**:
```bash
# 本地访问，端口 8000（默认）
adk web

# 本地访问，自定义端口
adk web --port 8080

# 允许外部访问，自定义端口
export BAYBE_DISABLE_TELEMETRY=1
adk web --host 0.0.0.0 --port 8080
```

**使用 Python 模块方式**:
```bash
# 如果 adk 命令不可用，可以使用 Python 模块方式
python -m google.adk web --port 8080
python -m google.adk web --host 0.0.0.0 --port 8080
```

#### 参数说明
- `--port <端口号>`: 指定服务器监听端口（默认: 8000）
- `--host <地址>`: 指定绑定地址
  - `127.0.0.1` 或 `localhost`: 仅本机访问（默认）
  - `0.0.0.0`: 绑定所有网络接口，允许外部访问

### System Requirements
- **Python**: 3.12.7 (当前测试环境)
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Memory**: 最低 8GB RAM (推荐 16GB+)

### Installation Steps

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Current Package Versions (Tested)

The following package versions are currently installed and tested in the development environment:

```
pandas==2.3.2
numpy==1.26.4
rdkit==2024.09.5
mordred==1.2.0
scikit-learn==1.7.1
matplotlib==3.10.5
seaborn==0.13.2
setuptools==80.9.0
baybe==0.13.2
google-adk==1.12.0
```

## Recent Updates (2026-01)

为便于发布与复现，以下为近期关键更新汇总：

1. **工具调用收敛与稳定性增强**
   - 由于测试发现直接构建代码运行不稳定，Recommender 已移除 `execute_baybe_code`（工具列表与提示中均禁用）
   - 首次构建强制使用 `build_campaign_and_recommend`

2. **离散参数禁用（仅保留连续参数）**
   - 实际化学研究中不应有离散参数，Enhanced Verification 不再输出离散建议
   - Campaign 构建阶段不再创建 `NumericalDiscreteParameter`

3. **CSV 表头污染拦截**
   - 若表头混入说明文字/参数范围，验证阶段直接提示并中止
   - 防止列错位导致的数值/文本混乱

4. **鲁棒性修复**
   - 边界计算对 `None/NaN` 做兜底处理，避免类型错误
   - 读取 CSV 自动过滤 `Unnamed:*` 空白索引列

5. **文档与依赖同步**
   - `requirements.txt` 与 README 版本列表保持一致
   - 新增 `baybe` 与 `google-adk` 依赖说明

6. **获取函数偏好**
   - 用户可选择 `qEI` / `qUCB` / `qNEI` / `qPI`
   - 构建 Campaign 时将应用该偏好

### Verify Installation

After installing the dependencies, verify that everything is working correctly:

```python
# Run this verification script
python -c "
import pandas as pd
import numpy as np
import rdkit
import mordred
import sklearn
import matplotlib
import seaborn

print('✓ Package verification successful!')
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
print(f'rdkit: {rdkit.__version__}')
print(f'mordred: {mordred.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print(f'seaborn: {seaborn.__version__}')
print('All dependencies are correctly installed!')
"
```

### Environment Configuration

Create a `.env` file in the project root with the following configuration:

```bash
# Google ADK Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/chembonas.log
```

## Usage

Once the dependencies are installed, you can interact with the main orchestrator agent, which will guide you through the workflow described above. Start by providing an initial data file when prompted.

### Experiment Result Upload

After receiving experimental recommendations, follow these steps to upload results:

#### Method 1: Using Auto-Generated Template (Recommended)

```python
# 1. Generate recommendations
recommendations = generate_recommendations(batch_size=5)

# 2. Generate result upload template
template = generate_result_template()
# This creates: result_template_[session_id]_[timestamp].csv

# 3. Perform experiments and fill in measured values in the template

# 4. Upload results (file path)
result = upload_experimental_results("result_template_filled.csv")

# 5. Check optimization progress
progress = check_convergence()
```

#### Method 2: Direct CSV Content Upload

```python
# Upload by pasting CSV content directly
csv_content = """
SubstanceA_molecule,SubstanceA_ratio,SubstanceB_molecule,SubstanceB_ratio,Target_yield,Target_quality
CC(C)O,0.6,NCCCN,0.4,87.5,4.2
CCO,0.7,NCCCCN,0.3,89.2,4.5
"""

result = upload_experimental_results(csv_content)
```

#### System Health Check

At any time, you can check the system status:

```python
health = check_agent_health()
# Output: System status, Campaign readiness, optimization round, etc.
```

### Complete Optimization Cycle Example

```python
# Round 1
recommendations = generate_recommendations("3")  # Get 3 experiments
template = generate_result_template()            # Generate template
# ... Perform experiments ...
upload_experimental_results("results.csv")       # Upload results
check_convergence()                              # Check progress

# Round 2 (if not converged)
recommendations = generate_recommendations("3")  # Get new recommendations
# ... Repeat the cycle ...
```

## 化学知识库 (ChemistryKnowledgeBase)

系统内置了化学领域知识库，为智能参数建议提供专业支持。

### 支持的反应类型

| 反应类型 | 典型温度范围 | 关键参数 |
|---------|------------|---------|
| 环氧固化 (epoxy_curing) | 60-120°C | 催化剂浓度 1%-10% |
| 聚合反应 (polymerization) | 40-100°C | 引发剂浓度 0.1%-5% |
| 催化合成 (catalytic_synthesis) | 25-150°C | 催化剂负载量 0.1%-10% |

### 安全约束示例

```python
SAFETY_CONSTRAINTS = {
    "temperature_limits": {
        "epoxy_systems": {
            "safe_max": 200,       # °C - 安全上限
            "flash_point_concern": 150,  # 闪点关注温度
            "decomposition_risk": 250    # 分解风险温度
        }
    },
    "ratio_constraints": {
        "epoxy_hardener": {
            "acceptable_range": (0.8, 1.2),  # 化学计量比
            "under_cure_risk": "<0.8",
            "over_cure_brittleness": ">1.2"
        }
    }
}
```

### 扩展知识库

如需添加新的反应类型或材料属性，编辑 `chemistry_knowledge_base.py` 文件：

```python
# 添加新的反应类型
REACTION_TYPES["new_reaction"] = {
    "name": "新反应类型名称",
    "typical_temperature": (min_temp, max_temp),
    "catalyst_concentration": (min_conc, max_conc),
    "safety_warnings": ["警告1", "警告2"],
    # ...
}
```

## LLM能力边界说明

### LLM擅长的任务

| 任务 | 说明 |
|-----|------|
| 识别反应类型 | 从物质名称和用户描述识别反应类型 |
| 应用已知规则 | 应用知识库中的规则和约束 |
| 语义理解 | 理解用户的优化意图和偏好 |
| 整合信息 | 生成人类可读的建议和解释 |

### LLM不擅长的任务

| 任务 | 原因 |
|-----|------|
| 精确数值推演 | 无法准确推算"应该扩展35.7%"这样的数值 |
| 实时化学数据查询 | 不能查询特定物质的精确物理化学参数 |
| 复杂计算 | 无法进行反应动力学、热力学计算 |

**因此**：边界建议的数值来源于知识库，而非LLM推演。LLM的角色是整合和交互，而非直接计算。

## 技术细节

### 参数边界推荐逻辑

verification agent中会建议调整的边界，其中部分化学规则是写死的，例如固化剂的比例，催化剂的添加量，如需探索其他范围，建议修改agent_zyf\enhanced_verification_tools.py中的语句以及化学知识库agent_zyf\chemistry_knowledge_base.py。
```python
def _get_ratio_bounds_from_kb(column_name, current_range, kb_suggestions):
    """
    从知识库获取比例参数的建议边界
    
    策略：
    1. 优先使用知识库中该反应类型的典型范围
    2. 结合当前数据范围，取并集以扩大探索空间
    3. 应用安全约束（如比例必须在0-1之间）
    """
    # 从知识库获取典型范围
    kb_bounds = knowledge_base.get_bounds(reaction_type)
    
    # 取并集扩大探索空间
    suggested_min = min(current_min, kb_min)
    suggested_max = max(current_max, kb_max)
    
    # 应用硬约束
    suggested_min = max(0.0, suggested_min)
    suggested_max = min(1.0, suggested_max)
    
    return (suggested_min, suggested_max), reasoning
```

### 边界来源标记

所有建议都会标记来源，便于追踪和审计：

```python
suggestions[col] = {
    "current_range": [0.3, 0.5],
    "suggested_bounds": (0.2, 0.6),
    "reasoning": "基于环氧固化反应的典型配比范围...",
    "source": "knowledge_base",  # 标明来源
    "requires_user_confirmation": True  # 需要用户确认
}
```

## 化合物名称自动映射

系统会自动从原始数据中提取 SMILES → 化合物名称 的映射关系，并在推荐结果和模板中显示友好的化合物名称。但对于实际实验中，如果某一类组分的化合物没有变化，则需要手动添加名称列。

### 工作原理

1. **数据验证阶段**：`enhanced_verification` 工具会扫描原始 CSV 中的 `*_molecule` 和 `*_name` 列配对，自动提取映射关系并保存到会话状态。

2. **推荐生成阶段**：`generate_recommendations` 和 `generate_result_template` 会自动为每个分子参数列添加对应的名称列。

### 输出示例

**原始推荐（仅 SMILES）**：
| SubstanceA_molecule | SubstanceA_ratio |
|---------------------|------------------|
| c1cc(OCC2CO2)ccc1... | 0.5 |

**添加名称后**：
| SubstanceA_molecule | SubstanceA_name | SubstanceA_ratio |
|---------------------|-----------------|------------------|
| c1cc(OCC2CO2)ccc1... | DGEBF | 0.5 |

### 注意事项

- 映射关系从原始数据自动提取，无需额外配置
- 如果原始数据没有 `*_name` 列，则不会生成名称列
- 上传实验结果时，`*_name` 列会被自动忽略，不影响 BayBE 优化

---

## Telemetry Notice (BayBE)

运行时可能会看到以下日志：
```
ERROR - exporter.py:340 - Failed to export metrics to public.telemetry.baybe.p.uptimize.merckgroup.com:4317, error code: StatusCode.PERMISSION_DENIED
```

这是 BayBE 的匿名遥测功能被网络/防火墙阻止，**不影响任何功能**。

### 禁用遥测

如需消除此日志，可在启动前设置环境变量：

**PowerShell**:
```powershell
$env:BAYBE_DISABLE_TELEMETRY="1"
adk web
```

**Bash/Zsh**:
```bash
export BAYBE_DISABLE_TELEMETRY=1
adk web
```

---

## 统一实验记录表模式

### 设计理念

为了降低用户的理解成本和操作复杂度，系统实现了**统一实验记录表模式**：

1. **格式一致性**：推荐表格完全复刻首次上传的数据记录表格格式
   - 保持原始列顺序
   - 保持原始列名格式
   - 包含所有原始列（包括元数据列）

2. **统一管理**：所有实验数据记录在同一个表格中
   - 避免多次迭代生成多个命名类似的表格
   - 支持在表格中直接调整推荐参数
   - 自动追踪实验状态（pending/completed）

### 工作流程

```
1. 首次上传数据
   ↓
   enhanced_verification 保存原始表格格式
   ↓
   创建统一实验记录表 (experiment_log.csv)
   ↓
2. 生成推荐
   ↓
   generate_recommendations 按照原始格式生成推荐
   ↓
   追加到统一实验记录表（状态：pending）
   ↓
3. 填写实验结果
   ↓
   用户在统一表格中填写目标值
   ↓
   （可选：适度调整推荐参数）
   ↓
4. 上传结果
   ↓
   upload_experimental_results 从统一表格读取
   ↓
   自动识别新完成的实验
   ↓
   更新状态为 completed
   ↓
   添加到 BayBE Campaign
```

### 关键特性

#### 1. 格式复刻

- **原始格式保存**：在 `enhanced_verification` 阶段保存原始表格的列顺序和格式
- **智能列匹配**：推荐生成时自动匹配参数列、目标列、元数据列
- **名称列支持**：自动添加化合物名称列（`*_name`），便于阅读

#### 2. 统一表格管理

- **文件位置**：`{session_dir}/experiment_log.csv`
- **状态列**：`experiment_status`（pending/completed）
- **轮次标记**：`optimization_round`（记录实验所属的优化轮次）

#### 3. 参数调整容错

- **允许调整**：用户可以在表格中适度调整推荐参数
- **范围验证**：系统会验证调整后的参数是否在搜索空间内
- **警告提示**：如果调整较大，系统会给出警告但不阻止上传

#### 4. 自动状态管理

- **自动识别**：上传时自动识别已填写目标值的待实验行
- **状态更新**：上传成功后自动将状态更新为 `completed`
- **增量上传**：只处理新完成的实验，已完成的实验不会重复上传

### 使用示例

#### 生成推荐（自动追加到统一表格）

```python
# 推荐会自动按照原始格式生成，并追加到 experiment_log.csv
result = build_campaign_and_recommend("5", tool_context)
```

#### 提取待实验模板（可选）

```python
# 从统一表格中提取待实验的行，生成独立模板文件
template = generate_result_template(tool_context)
# 输出：experiment_template_round_1.csv
```

#### 填写并上传（推荐方式）

```python
# 方式一：直接在统一表格中填写，然后上传统一表格
result = upload_experimental_results("experiment_log.csv", tool_context)

# 方式二：填写模板文件后上传
result = upload_experimental_results("experiment_template_round_1.csv", tool_context)
```

### 优势

1. **降低理解成本**：用户看到的是熟悉的表格格式
2. **减少文件混乱**：所有数据在一个表格中，便于管理
3. **支持参数调整**：允许用户根据实际情况微调推荐参数
4. **自动状态追踪**：系统自动管理实验状态，避免重复上传
5. **增量更新**：每次只处理新完成的实验，提高效率

### 注意事项

- **参数调整建议**：虽然允许调整，但建议仅在特殊情况下（如实验条件限制、安全约束）才修改推荐参数
- **格式一致性**：统一表格会保持原始格式，包括列顺序和列名
- **状态管理**：`experiment_status` 列由系统自动管理，用户无需手动修改


### TODOs：

- ADK 前端页面改造：更清晰的多步骤向导式流程
- 交互式实验表格：支持批量编辑、筛选、状态标记（pending/completed）
- 实时收敛折线图：显示目标值趋势与收敛判定
- 推荐结果对比视图：展示“推荐 vs 已完成实验”的差异
- 参数约束可视化编辑器：比例/线性约束可视化配置
- 获取函数选择器：qEI / qUCB / qNEI / qPI 的可视化切换
- 模型诊断面板：不确定度、特征重要性、预测区间
- 实验日志版本管理：轮次、时间戳、导出与回滚
- 数据质量提示：缺失值、异常值、列名污染的即时告警
- 可复制报告导出：一键导出优化总结（PDF/Markdown）