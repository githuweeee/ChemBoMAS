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

"""The Enhanced Recommender Agent for the ChemBoMAS system."""

import os
import logging
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions_recommender
from . import tools

RECOMMENDER_AGENT_MODEL = os.getenv("RECOMMENDER_AGENT_MODEL", "gemini-2.5-flash")

def check_prerequisites(callback_context: CallbackContext) -> None:
    """检查Enhanced Recommender Agent的前提条件"""
    required_keys = [
        "baybe_campaign",
        "ready_for_optimization"
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in callback_context.state:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Recommender Agent前提条件不满足，缺少: {', '.join(missing_keys)}")
    
    # 检查SearchSpace Construction Agent是否完成
    if not callback_context.state.get("ready_for_optimization", False):
        raise ValueError("SearchSpace Construction Agent尚未完成Campaign构建")

recommender_agent = LlmAgent(
    model=RECOMMENDER_AGENT_MODEL,
    name="recommender_agent", 
    instruction=return_instructions_recommender(),
    description="Generate experimental recommendations and manage iterative Bayesian optimization cycle",
    tools=[
        tools.generate_recommendations,
        tools.upload_experimental_results,
        tools.check_convergence,
    ],
    before_agent_callback=check_prerequisites,
)