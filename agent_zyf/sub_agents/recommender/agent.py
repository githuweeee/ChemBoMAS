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

"""The Unified Recommender Agent for the ChemBoMAS system.

This agent combines the functionality of:
- SearchSpace Construction Agent (Campaign building)
- Recommender Agent (Experiment recommendation)

Simplified architecture: Enhanced Verification → Recommender → Fitting
"""

import os
import logging
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions_recommender
from . import tools

# 导入缓存函数
from .tools import _get_campaign_from_cache

RECOMMENDER_AGENT_MODEL = os.getenv("RECOMMENDER_AGENT_MODEL", "gemini-2.5-flash")

def check_prerequisites(callback_context: CallbackContext) -> None:
    """检查Unified Recommender Agent的前提条件
    
    前提条件比原来更宽松：
    - 只需要Enhanced Verification Agent完成即可
    - Campaign可以在本Agent内部构建
    """
    # 检查Enhanced Verification Agent是否完成
    verification_status = callback_context.state.get("verification_status")
    
    # 如果已有Campaign（从缓存检查），直接放行
    session_id = callback_context.state.get("session_id", "unknown")
    if _get_campaign_from_cache(session_id) is not None:
        return
    
    # 如果没有Campaign，需要检查验证是否完成
    valid_statuses = ["completed_with_user_input", "configuration_complete", "ready_for_optimization"]
    
    if verification_status not in valid_statuses:
        # 检查是否有基本的验证结果
        if callback_context.state.get("verification_results"):
            # 有验证结果就可以继续
            return
        
        raise ValueError(
            f"Enhanced Verification Agent尚未完成。\n"
            f"当前状态: {verification_status}\n"
            f"请先完成数据验证和优化目标配置。"
        )

recommender_agent = LlmAgent(
    model=RECOMMENDER_AGENT_MODEL,
    name="recommender_agent", 
    instruction=return_instructions_recommender(),
    description="Build BayBE Campaign and generate experimental recommendations with iterative Bayesian optimization",
    tools=[
        # Campaign构建 + 首次推荐 (合并工具)
        tools.build_campaign_and_recommend,
        # 后续推荐生成
        tools.generate_recommendations,
        # 实验结果上传模板
        tools.generate_result_template,
        # 实验结果处理
        tools.upload_experimental_results,
        # 收敛性检查
        tools.check_convergence,
        # Campaign信息查询
        tools.get_campaign_info,
        # 系统健康检查
        tools.check_agent_health,
        # 代码执行工具已禁用（避免绕过标准流程）
    ],
    before_agent_callback=check_prerequisites,
)
