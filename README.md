# BoMAS Agent 🧪🤖

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-Enabled-green)](https://developers.google.com/adk)
[![BayBE](https://img.shields.io/badge/BayBE-Optimization-orange)](https://github.com/emdgroup/baybe)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

[中文](#bomas-项目描述) | [English](#bomas-project-description)

---

<a id="bomas-项目描述"></a>

# BoMAS 项目描述

**BoMAS** (Chemical Bayesian Optimization Multi-Agent System) 是一个基于 **Google Agent Development Kit (ADK)** 和 **BayBE** 贝叶斯优化框架构建的智能化学实验优化系统。该系统通过多智能体协作架构，实现从数据验证、SMILES分子结构处理、实验条件推荐、结果分析到持续迭代优化的完整闭环工作流程。**采用 MAS 的原因**在于：将验证、推荐、拟合等职责拆分给专业化子智能体，显著降低单一智能体的上下文负担与错误传播，增强可维护性、可扩展性与可复现性，并便于在不同环节引入领域规则与工具链。

系统内置化学知识库，能够为实验参数提供专业的边界建议和安全约束，支持单目标、多目标和帕累托前沿优化。通过贝叶斯优化算法，BoMAS能够以最少的实验次数找到最优实验条件，特别适用于环氧固化反应、聚合反应、催化合成、材料配方和工艺参数等化学实验优化场景。

### 与传统方法的区别

| 特性 | 传统 DOE | 传统 BO 库 | BoMAS |
|------|----------|-----------|-------|
| 实验设计 | 一次性固定设计（全因子/正交） | 序贯自适应 | 序贯自适应 + 智能边界建议 |
| 用户交互 | 需统计学背景 | 需编程能力 | 自然语言对话 |
| 数据验证 | 手动检查 | 无/手动 | 自动质量门控 |
| 约束处理 | 手动建模 | 代码硬编码 | 自动检测 + 用户确认 |
| 可复现性 | 依赖文档 | 依赖代码版本 | 完整上下文记录 |

### 为什么 BO 优化需要 AI Agent？

传统 BO 库以"黑箱"形式提供，用户需手动完成数据预处理、参数配置、结果解读等环节。**引入 AI Agent 的核心价值**：

| 传统 BO 库 | BoMAS (AI Agent 驱动) |
|------------|----------------------|
| 用户需手动编写代码调用 API | 自然语言交互，零代码上手 |
| 参数边界/约束需硬编码 | Agent 结合知识库自动建议 |
| 采集函数需专业知识选择 | Agent 根据优化阶段自适应推荐 |
| 数据质量问题导致优化失败 | 验证智能体自动拦截问题数据 |
| 约束冲突需人工排查 | Agent 自动检测并提示修正 |
| 结果解读需额外脚本 | 自动生成轨迹图、特征重要性 |

**技术细节**：
- **采集函数自适应**：根据轮次与改进幅度在探索型（qUCB）与利用型（qEI）间动态切换
- **约束自动生成**：比例列自动生成"和为1"约束，温度列自动施加安全上限
- **数据质量门控**：检测表头污染、非数字字符、SMILES有效性、目标值异常
- **上下文记忆**：维护完整优化历史，便于复现与审计
- **错误恢复**：推荐失败时给出明确修复建议

## 🚀 核心特点

*   **🤖 多智能体协作**：4个专业化智能体协同工作，实现全流程自动化。
*   **🧪 智能实验推荐**：基于BayBE贝叶斯优化框架，高效探索实验空间。
*   **🧠 智能边界建议**：内置化学知识库，为实验参数提供合理的边界建议和安全约束。
*   **📊 多目标优化**：支持单目标、多目标和帕累托前沿优化。
*   **🔄 自适应策略**：根据优化进展动态调整推荐策略（探索 vs 利用）。
*   **📈 结果可视化**：提供模型性能分析、特征重要性和收敛性分析。
*   **🛡️ 智能验证**：自动验证数据质量、CSV表头污染拦截和SMILES分子结构有效性。
*   **📝 统一数据管理**：自动管理实验记录表，追踪实验状态。

## 🛠️ 架构设计

BoMAS采用了一套模块化的参数边界推荐架构：

```mermaid
graph TD
    User[用户 / 化学家] -->|上传数据| VA[增强验证智能体]
    VA -->|验证数据 & 约束| RA[推荐智能体]
    RA -->|实验推荐| User
    User -->|上传结果| RA
    RA -->|更新 Campaign| BayBE[BayBE 核心]
    BayBE -->|新推荐| RA
    RA -->|数据分析| FA[拟合智能体]
    FA -->|模型洞察| User
```

### 智能参数顾问

```
[知识库 (KB)] --(硬约束)--> [顾问]
[RDKit 工具]  --(分子属性)--> [顾问] --> [用户确认] --> [BayBE SearchSpace]
[LLM]        --(意图 & 上下文)--> [顾问]
```

## ⚡ 快速开始

### 前置要求

*   Python 3.12+
*   Google ADK

### 安装步骤

1.  **克隆与设置**：
    ```bash
    git clone https://github.com/githuweeee/ChemBoMAS.git
    cd ChemBoMAS
    python -m venv .venv
    source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
    ```

2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

3.  **配置环境**：
    创建 `.env` 文件：
    ```ini
    GOOGLE_API_KEY=your_api_key
    LOG_LEVEL=INFO
    ```

### 启动 Agent

**默认 (Localhost:8000)**:
```bash
adk web
```

**自定义端口 / 外部访问**:
```bash
# 允许在 8080 端口外部访问
adk web --host 0.0.0.0 --port 8080
```

> **注意**: 如需禁用 BayBE 遥测，请在运行前设置 `BAYBE_DISABLE_TELEMETRY=1`。

## 📖 使用指南

1.  **启动会话**：启动 Web 界面并上传您的初始实验数据 (CSV)。**增强验证智能体**将验证您的数据并建议参数边界。
2.  **获取推荐**：Agent 将构建 BayBE Campaign 并生成第一批实验条件。
3.  **执行实验并上传**：下载提供的模板或使用统一的 `experiment_log.csv`。填写结果并上传回聊天界面。
4.  **分析与迭代**：系统自动更新模型。您可以随时请求可视化或检查收敛状态。

---

<a id="bomas-project-description"></a>

# BoMAS Project Description

**BoMAS** (Chemical Bayesian Optimization Multi-Agent System) is an intelligent chemical experiment optimization system built on the **Google Agent Development Kit (ADK)** and the **BayBE** Bayesian optimization framework. The system employs a multi-agent collaborative architecture to achieve a complete closed-loop workflow from data validation, SMILES molecular structure processing, experimental condition recommendation, result analysis, to continuous iterative optimization.

With a built-in chemistry knowledge base, the system provides professional boundary suggestions and safety constraints for experimental parameters, supporting single-objective, multi-objective, and Pareto frontier optimization. Through Bayesian optimization algorithms, ChemBoMAS can find optimal experimental conditions with minimal experiments, making it particularly suitable for optimization scenarios such as epoxy curing reactions, polymerization reactions, catalytic synthesis, material formulation, and process parameter optimization.

## 🚀 Key Features

*   **🤖 Multi-Agent Collaboration**: Four specialized agents work together to achieve full process automation.
*   **🧪 Intelligent Recommendation**: Efficiently explores the experimental space based on the BayBE Bayesian optimization framework.
*   **🧠 Intelligent Boundaries**: Built-in chemistry knowledge base provides reasonable boundary suggestions and safety constraints for experimental parameters.
*   **📊 Multi-Objective Optimization**: Supports single-objective, multi-objective, and Pareto frontier optimization.
*   **🔄 Adaptive Strategy**: Dynamically adjusts recommendation strategies (exploration vs. exploitation) based on optimization progress.
*   **📈 Result Visualization**: Provides model performance analysis, feature importance, and convergence analysis.
*   **🛡️ Intelligent Verification**: Automatically validates data quality, intercepts CSV header pollution, and checks SMILES validity.
*   **📝 Unified Data Management**: Automatically manages experimental logs and tracks experiment status.

## 🛠️ Architecture

ChemBoMAS employs a modular architecture for parameter boundary recommendation:

```mermaid
graph TD
    User[User / Chemist] -->|Uploads Data| VA[Enhanced Verification Agent]
    VA -->|Validated Data & Constraints| RA[Recommender Agent]
    RA -->|Experimental Recommendations| User
    User -->|Uploads Results| RA
    RA -->|Update Campaign| BayBE[BayBE Core]
    BayBE -->|New Recommendations| RA
    RA -->|Data for Analysis| FA[Fitting Agent]
    FA -->|Model Insights| User
```

### Intelligent Parameter Advisor

```
[Knowledge Base] --(Hard Constraints)--> [Advisor]
[RDKit Tools]    --(Molecule Props)----> [Advisor] --> [User Confirmation] --> [BayBE SearchSpace]
[LLM]            --(Intent & Context)--> [Advisor]
```

## ⚡ Quick Start

### Prerequisites

*   Python 3.12+
*   Google ADK

### Installation

1.  **Clone & Setup**:
    ```bash
    git clone https://github.com/githuweeee/ChemBoMAS.git
    cd ChemBoMAS
    python -m venv .venv
    source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file:
    ```ini
    GOOGLE_API_KEY=your_api_key
    LOG_LEVEL=INFO
    ```

### Running the Agent

**Default (Localhost:8000)**:
```bash
adk web
```

**Custom Port / External Access**:
```bash
# Allow external access on port 8080
adk web --host 0.0.0.0 --port 8080
```

> **Note**: To disable BayBE telemetry, set `BAYBE_DISABLE_TELEMETRY=1` before running.

## 📖 Usage Guide

1.  **Start Session**: Launch the Web interface and upload your initial experiment data (CSV). The **Enhanced Verification Agent** will validate your data and suggest parameter boundaries.
2.  **Get Recommendations**: The Agent will build a BayBE Campaign and generate the first batch of experimental conditions.
3.  **Run Experiments & Upload**: Download the provided template or use the unified `experiment_log.csv`. Fill in results and upload back to the chat interface.
4.  **Analyze & Iterate**: The system automatically updates the model. You can request visualization or check convergence status at any time.

### Detailed Operation Manual

#### 验证安装

安装依赖后，运行以下脚本验证环境配置是否正确：

```python
# 运行此验证脚本
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

#### 环境配置

在项目根目录创建 `.env` 文件并添加以下配置：

```bash
# Google ADK 配置
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/chembonas.log
```

#### 实验结果上传

收到实验推荐后，请按以下步骤上传结果：

**方法 1：使用自动生成的模板（推荐）**

```python
# 1. 生成推荐
recommendations = generate_recommendations(batch_size=5)

# 2. 生成结果上传模板
template = generate_result_template()
# 这将创建文件：result_template_[session_id]_[timestamp].csv

# 3. 进行实验并在模板中填写测量值

# 4. 上传结果（文件路径）
result = upload_experimental_results("result_template_filled.csv")

# 5. 检查优化进度
progress = check_convergence()
```

**方法 2：直接上传 CSV 内容**

```python
# 直接粘贴 CSV 内容上传
csv_content = """
SubstanceA_molecule,SubstanceA_ratio,SubstanceB_molecule,SubstanceB_ratio,Target_yield,Target_quality
CC(C)O,0.6,NCCCN,0.4,87.5,4.2
CCO,0.7,NCCCCN,0.3,89.2,4.5
"""

result = upload_experimental_results(csv_content)
```

#### 系统健康检查

您可以随时检查系统状态：

```python
health = check_agent_health()
# 输出：系统状态、Campaign 就绪情况、优化轮次等
```

#### 完整优化循环示例

```python
# 第一轮
recommendations = generate_recommendations("3")  # 获取 3 个实验推荐
template = generate_result_template()            # 生成模板
# ... 进行实验 ...
upload_experimental_results("results.csv")       # 上传结果
check_convergence()                              # 检查进度

# 第二轮（如未收敛）
recommendations = generate_recommendations("3")  # 获取新推荐
# ... 重复循环 ...
```

---

## 📂 Documentation / 文档

详细文档请参阅 `docs/` 目录。主要包括：
- `docs/examples/` - 示例数据与用例
- `快速开始指南.md` - 快速入门教程

---

## 🤝 Contributing / 贡献

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) (if available) and check the `docs/` folder for architectural details.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** - see the [LICENSE](LICENSE) file for details.

This means that if you run a modified version of this software as a network service (e.g., as a web application or SaaS), you must disclose the source code to the users of that service.

本项目采用 **GNU Affero 通用公共许可证 v3.0 (AGPLv3)** 授权。

这意味着，如果您将本软件的修改版本作为网络服务运行（例如作为 Web 应用程序或 SaaS），您必须向该服务的用户公开源代码。基于此项目进行的开发都需要开源。
