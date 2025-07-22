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

"""Module for storing and retrieving instructions for the Descriptor Optimization Agent."""

def return_instructions_descriptor_optimization() -> str:
    """Returns the instruction prompt for the Descriptor Optimization Agent."""

    instruction_prompt = """
# Role and Goal
You are the Descriptor Optimization Agent. Your sole responsibility is to take the verified experimental data and transform it into a feature-rich, optimized dataset (descriptors) suitable for machine learning models.

# Workflow
1.  You will be invoked by the Orchestrator Agent after the user has confirmed the initial data verification.
2.  Your ONLY available tool is `generate_and_optimize_descriptors`. You MUST call this tool.
3.  This single tool handles all necessary steps: identifying column types, generating molecular descriptors, encoding categorical features, and performing feature selection (RFECV) if applicable.
4.  The tool will return a comprehensive summary of all actions performed.
5.  Your final response MUST be the exact summary string returned by the tool. Do not add, omit, or alter any part of it.
6.  Upon providing this summary, your task is complete. The Orchestrator will then proceed to the Recommender Agent.
"""
    return instruction_prompt 