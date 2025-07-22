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

"""The Descriptor Optimization Agent for the ChemBoMAS system."""

import os
import logging
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from .prompts import return_instructions_descriptor_optimization
from . import tools

DESCRIPTOR_AGENT_MODEL = os.getenv("DESCRIPTOR_AGENT_MODEL", "gemini-1.5-pro-001")

def check_prerequisites(callback_context: CallbackContext) -> None:
    """Checks if the required data path exists in the session state."""
    if "verified_data_path" not in callback_context.state:
        raise ValueError("`verified_data_path` not found in session state.")

descriptor_optimization_agent = Agent(
    model=os.getenv("OAI_MODEL", "gpt-4-turbo"),
    name="descriptor_optimization_agent",
    instruction=return_instructions_descriptor_optimization(),
    tools=[tools.generate_and_optimize_descriptors],
    before_agent_callback=check_prerequisites,
) 