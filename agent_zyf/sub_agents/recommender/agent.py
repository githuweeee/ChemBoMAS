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

"""The Recommender Agent for the ChemBoMAS system."""

import os
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from .prompts import return_instructions_recommender
from . import tools


def check_prerequisites(callback_context: CallbackContext) -> None:
    """Checks if the required data paths exist in the session state."""
    required_keys = ["descriptors_path", "current_data_path", "experiment_round"]
    for key in required_keys:
        if key not in callback_context.state:
            raise ValueError(f"`{key}` not found in session state.")


recommender_agent = Agent(
    model=os.getenv("OAI_MODEL", "gpt-4-turbo"),
    name="recommender_agent",
    instruction=return_instructions_recommender(),
    tools=[tools.run_recommendation_step],
    before_agent_callback=check_prerequisites,
) 