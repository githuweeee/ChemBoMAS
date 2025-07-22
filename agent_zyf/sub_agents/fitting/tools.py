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

"""Tools for the Fitting Agent for analysis and visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google.adk.tools import ToolContext
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# --- Constants ---
TARGET_KEYWORDS = ["target", "yield", "output", "ee", "cost"]
TOP_N_FEATURES = 15  # Number of top features to display in the importance plot

def analyze_and_visualize_results(tool_context: ToolContext) -> str:
    """
    Analyzes and visualizes the results of completed experiments.

    This tool performs the following steps:
    1.  Loads the latest descriptor file containing all experimental data.
    2.  Trains a RandomForestRegressor model on the completed experiments.
    3.  Generates and saves a "Predicted vs. Actual" plot to evaluate model fit.
    4.  Generates and saves a "Feature Importance" plot to provide insights.
    5.  Updates the session state with the paths to the generated plots.
    6.  Returns a text summary of the analysis.
    """
    logging.info("Fitting Agent: Starting analysis and visualization...")

    try:
        # 1. Load data from session state
        descriptors_path_str = tool_context.state.get("descriptors_path")
        if not descriptors_path_str:
            raise ValueError("Session State is missing 'descriptors_path'.")

        df = pd.read_csv(descriptors_path_str)
        
        # Filter out rows with pending experiments for model training
        completed_df = df.dropna()
        if len(completed_df) == 0:
            return "No completed experiments found to analyze. Please provide experimental results."

        # 2. Separate features and targets
        target_cols = [c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS)]
        if not target_cols:
            raise ValueError("No target columns found for analysis.")
        
        # For simplicity, we analyze the first target column
        primary_target = target_cols[0]
        features = completed_df.drop(columns=target_cols)
        target = completed_df[primary_target]

        # 3. Train a RandomForest model for analysis
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model trained for analysis. R² score: {r2:.3f}")

        # --- Plotting ---
        session_id = tool_context.state.get("session_id", "unknown_session")
        plots_dir = Path(descriptors_path_str).parent
        sns.set_theme(style="whitegrid")

        # 4. "Predicted vs. Actual" Plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs. Actual ({primary_target})\nR² = {r2:.3f}")
        pred_vs_actual_path = plots_dir / f"predicted_vs_actual_{session_id}.png"
        plt.savefig(pred_vs_actual_path)
        plt.close()
        logging.info(f"Predicted vs. Actual plot saved to {pred_vs_actual_path}")

        # 5. "Feature Importance" Plot
        importances = model.feature_importances_
        indices = np.argsort(importances)[-TOP_N_FEATURES:]
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {TOP_N_FEATURES} Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
        plt.xlabel("Relative Importance")
        feature_importance_path = plots_dir / f"feature_importance_{session_id}.png"
        plt.savefig(feature_importance_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Feature Importance plot saved to {feature_importance_path}")

        # 6. Update session state with plot paths
        tool_context.state["analysis_plots"] = {
            "predicted_vs_actual": str(pred_vs_actual_path),
            "feature_importance": str(feature_importance_path),
        }
        tool_context.state["status"] = "Analysis_Complete"
        logging.info("Session state updated with plot paths.")

        # 7. Return summary
        summary = [
            "✅ Analysis Complete.",
            f"A model was trained on your {len(completed_df)} completed experiments with an R² score of {r2:.3f}.",
            "Below are the generated analysis plots:",
            f"- **Predicted vs. Actual Plot**: Shows how well the model's predictions match the real experimental outcomes.",
            f"- **Feature Importance Plot**: Highlights the top {TOP_N_FEATURES} reaction parameters that most significantly impact the '{primary_target}'.",
            "\nYou can now review these insights or start another round of recommendations."
        ]
        return "\n".join(summary)

    except Exception as e:
        logging.error(f"Error in analysis and visualization: {e}", exc_info=True)
        return f"An error occurred during results analysis: {e}" 