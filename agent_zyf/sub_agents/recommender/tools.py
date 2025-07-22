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

"""Tools for the Recommender Agent, powered by EDBO+."""

import logging
from pathlib import Path
import shutil
import pandas as pd
from google.adk.tools import ToolContext

# Assuming EDBO+ is installed and accessible in the environment.
# We add the project root to the path to ensure local modules can be found.
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from edbo.plus.optimizer_botorch import EDBOplus

# --- Constants ---
TARGET_KEYWORDS = ["target", "yield", "output", "ee"]
COST_KEYWORDS = ["cost"]
DEFAULT_BATCH_SIZE = 5


def run_recommendation_step(tool_context: ToolContext) -> str:
    """
    Runs one step of experiment recommendation using EDBO+.

    This tool orchestrates the core recommendation logic:
    1.  Loads the descriptor file from the session state.
    2.  Automatically determines objectives and optimization modes.
    3.  Initializes and runs the EDBOplus optimizer.
    4.  Generates a detailed prediction file and a clean, user-facing recommendation file.
    5.  Updates the session state with the new recommendation path and status.
    6.  Returns a summary to the user.
    """
    logging.info("Recommender Agent: Starting recommendation step...")

    try:
        # 1. Load data and context from session state
        descriptors_path_str = tool_context.state.get("descriptors_path")
        if not descriptors_path_str:
            raise ValueError("Session State is missing 'descriptors_path'.")

        session_id = tool_context.state.get("session_id", "unknown_session")
        round_number = tool_context.state.get("experiment_count", 0) + 1
        
        descriptors_path = Path(descriptors_path_str)
        temp_dir = descriptors_path.parent
        
        # EDBO works with a file in its current directory, so we copy it.
        temp_edbo_input_file = temp_dir / f"edbo_input_{session_id}.csv"
        shutil.copy(descriptors_path, temp_edbo_input_file)

        df = pd.read_csv(temp_edbo_input_file)

        # 2. Configure EDBO+ run
        objectives = [c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS + COST_KEYWORDS)]
        if not objectives:
            raise ValueError("No objective columns (e.g., 'yield', 'cost') found in the descriptor file.")
            
        objective_mode = []
        for obj in objectives:
            if any(k in obj.lower() for k in COST_KEYWORDS):
                objective_mode.append("min")
            else:
                objective_mode.append("max")

        summary = [
            f"Starting recommendation round {round_number}.",
            f"- Optimizing for objectives: {objectives}",
            f"- With respective modes: {objective_mode}",
        ]

        # 3. Run EDBO+
        optimizer = EDBOplus()
        # The `run` method handles both initial sampling and Bayesian optimization
        # based on the presence of 'PENDING' values in the objective columns.
        results_df = optimizer.run(
            objectives=objectives,
            objective_mode=objective_mode,
            batch=DEFAULT_BATCH_SIZE,
            directory=str(temp_dir),
            filename=temp_edbo_input_file.name,
            init_sampling_method='cvt', # Used only on the first run
        )
        
        logging.info("EDBO+ run complete.")
        summary.append("- EDBO+ optimization process finished successfully.")
        
        # 4. Create user-friendly recommendation file
        # The `results_df` is sorted by priority. We take the top `batch` entries.
        recommendations = results_df.head(DEFAULT_BATCH_SIZE)
        
        # The user only needs the original reaction parameters, not all the generated descriptors.
        # We find the original columns by looking for those that don't have '_' which is a proxy
        # for generated descriptor names (e.g., 'solvent_THF', 'smiles_mordred_...').
        # This is a heuristic; a more robust way would be to pass original columns from the start.
        original_cols = [c for c in df.columns if '_' not in c or c in objectives]
        user_recommendations_df = recommendations[original_cols]

        recommendations_filename = f"recommendations_round_{round_number}_{session_id}.csv"
        recommendations_path = temp_dir / recommendations_filename
        user_recommendations_df.to_csv(recommendations_path, index=False)
        logging.info(f"User-facing recommendations saved to: {recommendations_path}")
        summary.append(f"- Generated {len(user_recommendations_df)} new experiment suggestions.")

        # 5. Update session state
        tool_context.state["recommendations_path"] = str(recommendations_path)
        tool_context.state["experiment_count"] = round_number
        tool_context.state["status"] = "Awaiting_User_Experiment"
        logging.info("Session state updated.")

        summary.append(f"\n✅ Success! Please download the file '{recommendations_filename}'.")
        summary.append("After performing the experiments, update the objective columns in the file and re-upload it to continue.")

        return "\n".join(summary)

    except Exception as e:
        logging.error(f"Error in recommendation step: {e}", exc_info=True)
        return f"An error occurred during recommendation: {e}" 