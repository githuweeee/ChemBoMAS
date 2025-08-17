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

# Import orchestrator-specific components
from .prompts import return_instructions_orchestrator,return_instructions_verification, return_instructions_descriptor
from . import tools

descriptor_agent = LlmAgent(
    name = "verification_agent", 
    model = "gemini-2.5-flash",
    instruction = return_instructions_descriptor(),
    description="verify data",
    tools = [tools.generate_descriptor],
    output_key="generate descriptor"
)

verification_agent = LlmAgent(
    name = "verification_agent",
    model = "gemini-2.5-flash",
    instruction = return_instructions_verification(),
    description="verify data",
    tools = [tools.verification],
    output_key="verification"
)


# Define the main Orchestrator Agent
root_agent = LlmAgent(
    model= "gemini-2.5-flash",
    name="orchestrator_agent",
    instruction=return_instructions_orchestrator(),
    sub_agents=[verification_agent,descriptor_agent],
    tools=[
        #verification_agent,
        #descriptor_optimization_agent,
        #recommender_agent,
        #fitting_agent,
        tools.handle_file_upload,
    ],
) 