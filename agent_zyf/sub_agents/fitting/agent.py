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

"""The Enhanced Fitting Agent for the ChemBoMAS system."""

import os
import logging
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions_fitting
from . import tools

FITTING_AGENT_MODEL = os.getenv("FITTING_AGENT_MODEL", "gemini-2.5-flash")

def check_prerequisites(callback_context: CallbackContext) -> None:
    """检查Enhanced Fitting Agent的前提条件"""
    # 尝试从缓存恢复campaign（避免state未保存对象导致的误报）
    campaign = tools._ensure_campaign_in_state(callback_context.state)
    if campaign is None:
        raise ValueError("Fitting Agent前提条件不满足，缺少: baybe_campaign")
    
    # 检查是否有足够的实验数据进行分析
    if campaign and hasattr(campaign, 'measurements'):
        measurement_count = len(campaign.measurements)
        if measurement_count < 1:
            raise ValueError("需要至少1轮实验数据才能进行分析")

fitting_agent = LlmAgent(
    model=FITTING_AGENT_MODEL,
    name="fitting_agent", 
    instruction=return_instructions_fitting(),
    description="Analyze BayBE Campaign performance, create interpretable models, and generate optimization reports",
    tools=[
        tools.analyze_campaign_performance,
        tools.create_interpretable_model,
        tools.generate_optimization_report,
    ],
    before_agent_callback=check_prerequisites,
)