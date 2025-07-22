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

"""Tools for the Descriptor Optimization Agent."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from google.adk.tools import ToolContext
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# --- Constants ---

# Keywords to identify target columns, which will be excluded from feature generation.
TARGET_KEYWORDS = ["target", "yield", "output", "ee", "cost"]
# Keyword to identify SMILES columns for molecular descriptor generation.
SMILES_KEYWORD = "smiles"
# Minimum number of experiments required to trigger RFECV.
RFECV_THRESHOLD = 10


def _encode_smiles(df: pd.DataFrame, smiles_cols: list[str]) -> pd.DataFrame:
    """Generates molecular descriptors from SMILES columns using mordred."""
    logging.info(f"Generating molecular descriptors for columns: {smiles_cols}")
    calc = Calculator(descriptors, ignore_3D=True)
    all_descriptors = []

    for col in smiles_cols:
        mols = [Chem.MolFromSmiles(smi) for smi in df[col]]
        # Calculate descriptors, returning a pandas DataFrame
        descriptor_df = calc.pandas(mols, quiet=True)
        # Add prefix to column names to avoid collisions
        descriptor_df = descriptor_df.add_prefix(f"{col}_")
        all_descriptors.append(descriptor_df)

    # Concatenate all descriptor dataframes
    if not all_descriptors:
        return pd.DataFrame()

    final_df = pd.concat(all_descriptors, axis=1)
    # Convert all descriptor columns to numeric, coercing errors to NaN
    final_df = final_df.apply(pd.to_numeric, errors="coerce")
    # Impute NaN values with the column mean
    final_df.fillna(final_df.mean(), inplace=True)

    logging.info(f"Generated {len(final_df.columns)} molecular descriptors.")
    return final_df


def _encode_categorical(
    df: pd.DataFrame, categorical_cols: list[str]
) -> pd.DataFrame:
    """Applies one-hot encoding to categorical columns."""
    if not categorical_cols:
        return pd.DataFrame()
    logging.info(f"Applying one-hot encoding to: {categorical_cols}")
    return pd.get_dummies(df[categorical_cols], prefix=categorical_cols)


def _select_features_rfecv(
    features: pd.DataFrame, target: pd.DataFrame
) -> (pd.DataFrame, str):
    """Performs Recursive Feature Elimination with Cross-Validation."""
    logging.info("Starting RFECV for feature selection...")
    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    # Use the first target column for feature selection if multiple targets exist
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=3,  # 3-fold cross-validation
        scoring="neg_mean_squared_error",
        n_jobs=-1,  # Use all available CPU cores
    )
    selector.fit(features, target.iloc[:, 0])

    selected_features = features.loc[:, selector.support_]
    num_initial = features.shape[1]
    num_selected = selected_features.shape[1]
    summary = (
        "Recursive Feature Elimination (RFECV) complete. "
        f"Selected {num_selected} features out of {num_initial}."
    )
    logging.info(summary)
    return selected_features, summary


def generate_and_optimize_descriptors(tool_context: ToolContext) -> str:
    """
    Main tool function to orchestrate descriptor generation and optimization.

    This function performs a multi-step process:
    1.  Loads verified data.
    2.  Identifies columns by type (SMILES, categorical, numerical, target).
    3.  Generates molecular descriptors for SMILES columns.
    4.  Applies one-hot encoding for categorical columns.
    5.  Combines all features into a single matrix.
    6.  If data is sufficient, runs RFECV to select the most important features.
    7.  Saves the final descriptor matrix and updates the session state.
    8.  Returns a summary of the operations performed.
    """
    logging.info("Descriptor Optimization Agent: Starting...")

    try:
        # 1. Load data and context from session state
        verified_path_str = tool_context.state.get("verified_data_path")
        if not verified_path_str:
            raise ValueError("Session State is missing 'verified_data_path'.")
        df = pd.read_csv(verified_path_str)
        exp_count = tool_context.state.get("experiment_count", len(df))

        # 2. Identify column types
        target_cols = [
            c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS)
        ]
        smiles_cols = [
            c for c in df.columns if SMILES_KEYWORD in c.lower()
        ]
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Exclude target columns from numerical features
        numerical_cols = [c for c in numerical_cols if c not in target_cols]

        # Categorical columns are non-numeric, non-SMILES, and non-target
        categorical_cols = [
            c for c in df.columns
            if c not in smiles_cols + numerical_cols + target_cols
        ]
        
        summary = [
            "Descriptor generation process started.",
            f"- Identified {len(target_cols)} target column(s): {target_cols}",
            f"- Identified {len(smiles_cols)} SMILES column(s): {smiles_cols}",
            f"- Identified {len(categorical_cols)} categorical column(s): {categorical_cols}",
            f"- Identified {len(numerical_cols)} numerical column(s): {numerical_cols}",
        ]

        # 3. Feature Engineering
        smiles_features = _encode_smiles(df, smiles_cols)
        categorical_features = _encode_categorical(df, categorical_cols)
        numerical_features = df[numerical_cols]
        
        # Keep targets separate
        targets = df[target_cols]

        # Combine all features
        all_features = pd.concat(
            [numerical_features, categorical_features, smiles_features], axis=1
        )
        summary.append(f"-> Generated a total of {all_features.shape[1]} features.")

        # 4. Conditional Feature Selection
        rfecv_summary = ""
        if exp_count >= RFECV_THRESHOLD:
            final_features, rfecv_summary = _select_features_rfecv(
                all_features, targets
            )
            summary.append(f"- {rfecv_summary}")
        else:
            final_features = all_features
            summary.append(
                f"- Skipping feature selection (RFECV) as experiment count ({exp_count}) is below threshold ({RFECV_THRESHOLD})."
            )

        # 5. Save results and update state
        final_df_to_save = pd.concat([final_features, targets], axis=1)

        session_id = tool_context.state.get("session_id", "unknown_session")
        verified_path = Path(verified_path_str)
        descriptors_dir = verified_path.parent
        
        descriptors_filename = f"descriptors_{session_id}.csv"
        descriptors_path = descriptors_dir / descriptors_filename
        
        final_df_to_save.to_csv(descriptors_path, index=False)
        logging.info(f"Final descriptors saved to: {descriptors_path}")

        tool_context.state["descriptors_path"] = str(descriptors_path)
        tool_context.state["status"] = "Descriptors_Generated"
        
        summary.append(f"\n✅ Success! Descriptors file created at '{descriptors_path}'.")
        summary.append("Proceeding to the next step: Recommender Agent.")

        return "\n".join(summary)

    except Exception as e:
        logging.error(f"Error in descriptor generation: {e}", exc_info=True)
        return f"An error occurred during descriptor generation: {e}" 