# Ford Fleet AI Copilot

Ford Fleet AI Copilot is a data science and AI application for fleet health monitoring. The project combines vehicle telematics, service history, failure labels, support tickets, and maintenance documentation to support exploratory analysis, predictive maintenance modeling, retrieval augmented generation, and a Streamlit user interface.

## Project Goals

- Analyze fleet telematics and service data for maintenance patterns.
- Build features for vehicle health, usage, and failure risk.
- Train models that predict likely vehicle failures or maintenance needs.
- Use maintenance manuals and support documents for RAG-based explanations.
- Provide an interactive Streamlit app for fleet insights and decision support.

## Repository Structure

```text
ford-fleet-ai-copilot/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── docs/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── rag/
├── requirements.txt
├── README.md
└── .gitignore
```

## Data

Raw project data lives in `data/raw/`:

- `vehicles.csv`: vehicle metadata.
- `telematics.csv`: operational telemetry and sensor readings.
- `service_history.csv`: historical maintenance events.
- `failure_labels.csv`: target labels for model training.
- `support_tickets.csv`: user-reported issues and service requests.

Reference documents live in `data/docs/`:

- `battery_manual.txt`
- `brake_manual.txt`
- `engine_manual.txt`
- `oil_pressure_manual.txt`
- `transmission_manual.txt`

Processed datasets should be written to `data/processed/`.

## Notebook Workflow

Use the notebooks in order:

1. `01_data_generation.ipynb`: create or refresh sample project data.
2. `02_eda.ipynb`: inspect distributions, missing values, relationships, and failure trends.
3. `03_feature_engineering.ipynb`: create model-ready features.
4. `04_model_training.ipynb`: train predictive maintenance models.
5. `05_model_evaluation.ipynb`: evaluate model quality and explain results.

## Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using VS Code or Jupyter, select the project environment as your notebook kernel:

```text
ford-fleet-ai-copilot/.venv/bin/python
```

## Running the App

Start the Streamlit app from the project root:

```bash
streamlit run app/streamlit_app.py
```

## Core Dependencies

- `pandas` and `numpy` for data processing.
- `matplotlib` and `seaborn` for visualization.
- `scikit-learn` for machine learning.
- `jupyter` for notebook development.
- `streamlit` for the app interface.

## Development Notes

- Keep raw source files in `data/raw/`.
- Save cleaned and model-ready files in `data/processed/`.
- Put reusable data loading code in `src/data/`.
- Put feature transformations in `src/features/`.
- Put model training and inference utilities in `src/models/`.
- Put retrieval and document search logic in `src/rag/`.
- Put API or service endpoints in `src/api/`.
- Put drift checks, health metrics, and operational monitoring in `src/monitoring/`.

## Current Status

This project is in the early build stage. The folder architecture, starter notebooks, raw data files, document files, dependencies, and Streamlit entrypoint are in place.
