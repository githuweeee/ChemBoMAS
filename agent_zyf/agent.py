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

"""The main Orchestrator Agent for the ChemBoMAS system."""

import os
from google.adk.agents import LlmAgent

# Import sub-agents
from .sub_agents.searchspace_construction.agent import searchspace_construction_agent
from .sub_agents.recommender.agent import recommender_agent
from .sub_agents.fitting.agent import fitting_agent

# Import orchestrator-specific components
from .prompts import return_instructions_orchestrator, return_instructions_enhanced_verification
from . import tools
from .enhanced_verification_tools import enhanced_verification, collect_optimization_goals, diagnose_data_types

# Enhanced Verification Agent - 合并了原来的verification和descriptor功能
enhanced_verification_agent = LlmAgent(
    name="enhanced_verification_agent",
    model="gemini-2.5-flash",
    instruction=return_instructions_enhanced_verification(),
    description="Enhanced data verification, SMILES validation, and user interaction for optimization configuration",
    tools=[
        enhanced_verification,           # 主要的增强验证功能（7个任务）
        collect_optimization_goals,      # 收集用户优化目标
        diagnose_data_types,            # 诊断数据类型问题
        tools.verification,              # 保留原有的基础验证作为备用
    ],
    output_key="enhanced_verification"
)


# Define the main Orchestrator Agent
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="orchestrator_agent",
    instruction=return_instructions_orchestrator(),
    sub_agents=[
        enhanced_verification_agent,     # 数据验证、SMILES验证、用户交互
        searchspace_construction_agent,  # BayBE搜索空间构建  
        recommender_agent,               # 实验推荐和迭代优化
        fitting_agent,                   # 模型分析与可视化
    ],
    tools=[
        tools.handle_file_upload,
    ],
) 