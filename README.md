# Uncertainty-Aware Explainable Multimodal Deep Learning Framework for Differential Diagnosis of Brain Encephalitis

## Overview

This project is a full-stack automated diagnostic system designed specifically for the differential diagnosis of brain encephalitis using a multimodal approach. It combines imaging data (MRI/CT scans) and clinical metrics (CSF profiles, age, symptoms) to predict the encephalitis etiology (Amoebic, Viral, Bacterial, or Non-infectious). 

Not only does this system provide predictions, but it is built to be **uncertainty-aware** and **explainable**:
- Tracks **Prediction Uncertainty** using Monte Carlo (MC) Dropout to give doctors a measure of confidence.
- Provides **Grad-CAM Heatmaps** that highlight focal brain regions in the uploaded scans driving the model's decision.
- Incorporates **SHAP Values** to reveal how clinical variables (e.g., CSF protein) contributed individually to the diagnosis.

## Tech Stack
- **Frontend**: React (Vite), Vanilla CSS premium styling
- **Backend / API**: Python FastAPI
- **Machine Learning**: PyTorch (Multimodal: CNN Image Encoder + Dense Clinical Encoder)
- **Deployment**: Local Uvicorn server, suitable for Docker/production mapping.

---

## Folder Structure

```
.
├── backend
│   ├── main.py              # FastAPI server and routing endpoints
│   ├── ml_model.py          # PyTorch models, Monte Carlo inference, wrappers
│   ├── schemas.py           # Pydantic schemas for data validation
│   └── requirements.txt     # Python backend dependencies
├── frontend
│   ├── package.json         # React dependencies
│   ├── vite.config.js       # Vite bundler configuration
│   └── src
│       ├── App.jsx          # Main application layout
│       ├── index.css        # Premium dashboard styling
│       ├── main.jsx         # React DOM entry
│       └── components
│           ├── Dashboard.jsx        # Clinical input and upload interface
│           └── PredictionResult.jsx # Visual explanations and scores UI
└── README.md
```

## Setup Instructions

### 1. Backend Setup

Launch the FastAPI backend module to serve the inferences.

```bash
cd backend
# Optional: Create a virtual environment
# python -m venv venv
# source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install libraries
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Note: Ensure an active python environment for PyTorch (>=3.8 recommended).*

The backend should now run at `http://localhost:8000`. You can verify deployment at `http://localhost:8000/health` or `http://localhost:8000/docs` (Swagger UI).

### 2. Frontend Setup

Run the Vite React frontend. Open another terminal in the root project folder:

```bash
cd frontend

# Install Node dependencies (only required the first time)
npm install

# Start the Vite development server
npm run dev
```

The interface will be hosted locally at `http://localhost:5173`. Open this URL in the browser.

---

## Model Explanations & Details 🧠

1. **Architecture Overview**: The model fuses imaging features extracted from a `DummyImageEncoder` (CNN prototype) and a `DummyClinicalEncoder` (MLP wrapper over clinical metrics) before routing through a unified classification multi-layer perceptron.
2. **Explainability Elements**:
   - `mc_dropout_predict()`: Makes consecutive stochastic predictions (via enabled Dropout layers during eval) to assess output variability/uncertainty score.
   - `generate_gradcam()`: Prototype hook implementation to identify focus pixels in convolution sequences.
   - `generate_shap()`: Exposes which clinical variables (e.g. glucose levels) pushed the class prediction positively or negatively.
*(Note: Since raw pretrained weights were not supplied, a complete dummy/synthetic PyTorch scaffold is initialized, providing identical endpoint formats ready for the production weights replacement.)*
