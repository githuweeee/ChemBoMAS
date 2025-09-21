# ChemBoMAS Agent

This agent is designed to assist in chemical experiment optimization. It uses a multi-agent system built with the Google Agent Development Kit (ADK) to guide a user through a cycle of data submission, feature engineering, experiment recommendation, and results analysis.

## Workflow

The agent follows these main steps:

1.  **Data Upload**: The user uploads a CSV file containing chemical reaction data. This includes substance names, their SMILES strings, ratios, and target experimental outcomes (e.g., yield, cost).

2.  **Data Verification**: The system first verifies the uploaded data to ensure it conforms to the required format, checking for correctly named columns for substances and targets.

3.  **Descriptor Generation and SearchSpace Construction**:
    *   Molecular descriptors are calculated from the SMILES strings for each substance using `rdkit` and `mordred`.
    *   The system identifies experimental parameters and constructs a BayBE-compatible search space.
    *   Parameter constraints and bounds are automatically defined based on chemical experiment rules.

4.  **Bayesian Optimization and Experiment Recommendation**:
    *   Using the constructed search space, the agent employs BayBE (Bayesian Optimization for Black-box Experiments) to recommend the next batch of experiments.
    *   BayBE's internal algorithms handle feature optimization and descriptor processing automatically.
    *   The goal is to efficiently explore the experimental space to find optimal conditions with minimal experiments.

5.  **Analysis and Visualization**:
    *   After the user performs the recommended experiments and uploads the results, the `fitting` agent takes over.
    *   It trains a Random Forest model on the completed experimental data.
    *   It generates two key visualizations:
        *   A **Predicted vs. Actual** plot to assess model accuracy.
        *   A **Feature Importance** plot to show which parameters have the most significant impact on the experimental outcomes.

This iterative cycle allows for continuous learning and optimization of the chemical process.

## Setup and Installation

To run this agent, you need to install the required Python dependencies.

### System Requirements
- **Python**: 3.8 - 3.11 (推荐 3.10)
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Memory**: 最低 8GB RAM (推荐 16GB+)

### Installation Steps

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Current Package Versions (Tested)

The following package versions are currently installed and tested in the development environment:

```
pandas==2.2.2
numpy==1.26.4
rdkit==2024.9.3
mordred==1.2.0
scikit-learn==1.6.1
matplotlib==3.9.2
seaborn==0.13.2
setuptools==75.1.0
```

### Verify Installation

After installing the dependencies, verify that everything is working correctly:

```python
# Run this verification script
python -c "
import pandas as pd
import numpy as np
import rdkit
import mordred
import sklearn
import matplotlib
import seaborn

print('✓ Package verification successful!')
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
print(f'rdkit: {rdkit.__version__}')
print(f'mordred: {mordred.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print(f'seaborn: {seaborn.__version__}')
print('All dependencies are correctly installed!')
"
```

### Environment Configuration

Create a `.env` file in the project root with the following configuration:

```bash
# Google ADK Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/chembonas.log
```

## Usage

Once the dependencies are installed, you can interact with the main orchestrator agent, which will guide you through the workflow described above. Start by providing an initial data file when prompted.
