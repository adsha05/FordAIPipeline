# Ford Fleet AI Copilot

Ford Fleet AI Copilot is an end-to-end predictive maintenance project for fleet health monitoring. It uses vehicle telematics, service history, failure labels, support tickets, and maintenance documents to predict 30-day failure risk and support maintenance prioritization through notebooks, a trained model, a FastAPI service, a Streamlit dashboard, and a RAG document retrieval workflow.

## What This Project Does

- Generates and analyzes synthetic fleet operations data.
- Builds vehicle-level features from telematics, service, ticket, and failure data.
- Trains an XGBoost-based failure risk model.
- Saves model artifacts for repeatable inference.
- Exposes vehicle risk predictions through a FastAPI API.
- Provides a Streamlit app for selecting vehicles and ranking high-risk assets.
- Builds a vector store from maintenance manuals for future RAG-based support.

## Repository Structure

```text
ford-fleet-ai-copilot/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── docs/
│   ├── processed/
│   └── raw/
├── models/
│   ├── failure_risk_model.pkl
│   ├── feature_columns.pkl
│   └── xgb_explainability_model.pkl
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── rag/
│       └── build_vector_store.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Assets

Raw data lives in `data/raw/`:

- `vehicles.csv`: fleet vehicle metadata.
- `telematics.csv`: vehicle telemetry and operating signals.
- `service_history.csv`: historical maintenance activity.
- `failure_labels.csv`: supervised learning target labels.
- `support_tickets.csv`: driver or support-reported issues.

Processed data lives in `data/processed/`:

- `eda_vehicle_level.csv`: vehicle-level dataset for exploration.
- `model_training_table.csv`: feature table used for training and inference.
- `model_comparison.csv`: model evaluation comparison output.
- `test_predictions.csv`: test-set prediction results.
- `vehicle_risk_explanations.csv`: vehicle-level explanation and recommendation output.

Maintenance documents live in `data/docs/` and are used by the RAG vector-store builder.

## Model Artifacts

The trained model files are stored in `models/`:

- `failure_risk_model.pkl`: primary failure risk classifier used by the API.
- `feature_columns.pkl`: ordered feature list required during inference.
- `xgb_explainability_model.pkl`: XGBoost model artifact used for explainability workflows.

## Notebook Workflow

Run the notebooks in order:

1. `01_data_generation.ipynb`: generate raw fleet, telematics, service, ticket, and failure data.
2. `02_eda.ipynb`: perform exploratory data analysis.
3. `03_feature_engineering.ipynb`: create model-ready vehicle-level features.
4. `04_model_training.ipynb`: train and save failure risk models.
5. `05_model_evaluation.ipynb`: evaluate predictions and generate model explanations.

## Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For VS Code or Jupyter, select this notebook kernel:

```text
ford-fleet-ai-copilot/venv/bin/python
```

## Running the API

Start the FastAPI service from the project root:

```bash
venv/bin/uvicorn src.api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /`: API status and available endpoints.
- `GET /health`: model and data health check.
- `POST /predict/vehicle`: predict risk for a known `vehicle_id`.
- `POST /predict/features`: predict risk from a custom feature payload.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict/vehicle" \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id": "V0001"}'
```

## Running the Streamlit App

Start the API first, then run Streamlit in a second terminal:

```bash
venv/bin/streamlit run app/streamlit_app.py
```

The app lets you:

- Select a vehicle and request a failure risk prediction.
- View failure probability and risk level.
- Scan a subset of the fleet.
- Rank vehicles by predicted risk.
- Visualize risk level counts and probability distribution.

## Building the RAG Vector Store

The RAG builder reads text files from `data/docs/`, embeds them with `all-MiniLM-L6-v2`, and saves a FAISS vector index plus document metadata under `rag_store/`.

Run:

```bash
venv/bin/python src/rag/build_vector_store.py
```

Expected outputs:

```text
rag_store/maintenance_docs.index
rag_store/documents.pkl
```

## Core Dependencies

- `pandas`, `numpy`, and `scikit-learn` for data processing and modeling.
- `xgboost` for failure risk modeling.
- `matplotlib` and `seaborn` for visualization.
- `jupyter` for notebook workflows.
- `fastapi`, `uvicorn`, and `pydantic` for the prediction API.
- `streamlit` for the dashboard.

The RAG vector-store workflow also uses FAISS and Sentence Transformers.

## Current Status

The project now includes the full local prototype workflow:

- Raw and processed fleet datasets.
- Completed notebook workflow through model evaluation.
- Saved model artifacts.
- FastAPI prediction service.
- Streamlit dashboard connected to the API.
- Initial RAG vector-store builder for maintenance documentation.

Next likely work includes adding API tests, improving model monitoring, wiring RAG retrieval into the app or API, and adding deployment configuration.
