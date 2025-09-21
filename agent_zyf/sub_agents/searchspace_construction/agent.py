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

"""The SearchSpace Construction Agent for the ChemBoMAS system."""

import os
import logging
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from .prompts import return_instructions_searchspace_construction
from . import tools

SEARCHSPACE_AGENT_MODEL = os.getenv("SEARCHSPACE_AGENT_MODEL", "gemini-2.5-flash")

def check_prerequisites(callback_context: CallbackContext) -> None:
    """
    检查SearchSpace Construction Agent的前提条件
    """
    required_keys = [
        "verification_results",
        "baybe_campaign_config",
        "verification_status"
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in callback_context.state:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"SearchSpace Construction前提条件不满足，缺少: {', '.join(missing_keys)}")
    
    # 检查Enhanced Verification Agent是否完成
    verification_status = callback_context.state.get("verification_status")
    if verification_status != "completed_with_user_input":
        raise ValueError(f"Enhanced Verification Agent尚未完成。当前状态: {verification_status}")

searchspace_construction_agent = LlmAgent(
    model=SEARCHSPACE_AGENT_MODEL,
    name="searchspace_construction_agent",
    instruction=return_instructions_searchspace_construction(),
    description="Construct BayBE SearchSpace and Campaign from validated data and user preferences",
    tools=[
        tools.construct_searchspace_and_campaign,
        tools.validate_campaign_readiness,
        tools.get_campaign_info,
    ],
    before_agent_callback=check_prerequisites,
)
