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

"""Verification Agent: A ToolAgent for data validation and summarization."""

import os
import logging
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from .prompts import return_instructions_verification
from . import tools

# It is recommended to use environment variables to configure the model.
# This allows for greater flexibility and avoids hardcoding model names.
VERIFICATION_AGENT_MODEL = os.getenv("VERIFICATION_AGENT_MODEL", "gemini-2.5-flash")


def check_prerequisites(callback_context: CallbackContext) -> None:
    """Checks if the required data path exists in the session state."""
    if "current_data_path" not in callback_context.state:
        raise ValueError("`current_data_path` not found in session state.")


verification_agent = Agent(
    name="verification_agent",
    model=VERIFICATION_AGENT_MODEL,
    instruction=return_instructions_verification(),
    tools=[tools.verify_and_summarize_data],
    before_agent_callback=check_prerequisites,
) 