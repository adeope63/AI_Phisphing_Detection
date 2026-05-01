# AI-Driven Phishing Detection

A simple machine learning project that detects phishing emails from text fields (`sender`, `subject`, `body`, and `urls`).

## Project Structure

- `backend/EDA.py` - exploratory data analysis and chart generation
- `backend/model_selection.py` - train/evaluate models with CV, leakage checks, temporal validation, and engineered phishing features
- `backend/features.py` - text combiner + engineered URL/sender/body features + preprocessing pipeline
- `backend/best_model.py` - inference helpers (`predict_email`, `predict_email_with_confidence`) and optional best-model evaluation
- `backend/evaluate_best_model.py` - evaluate saved best model (no import side effects)
- `frontend/app.py` - Flask app for interactive prediction UI
- `frontend/index.html`, `frontend/style.css` - frontend template and styling

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Training Pipeline

Train and compare models, then save the best model bundle:

```bash
python backend/model_selection.py
```

This generates artifacts in `backend/output/`, including:

- `best_phishing_model.pkl`
- `model_results.csv` (includes CV and temporal metrics)
- `overfitting_summary.csv`
- `leakage_report.csv`
- plots for model comparison and confusion matrix

## Evaluate Saved Best Model

```bash
python backend/evaluate_best_model.py
```

## Run Web App

```bash
python frontend/app.py
```

Open the app in your browser at the local Flask URL shown in terminal.

## API Endpoints

- `GET /health` - health check endpoint
- `POST /predict` - JSON prediction endpoint

Example request:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "alerts@bank.com",
    "subject": "Verify account",
    "body": "Urgent! confirm your account now",
    "urls": "http://example.com/login"
  }'
```

The API returns `label`, numeric `prediction`, and `confidence`.

## Tests

Run automated tests with:

```bash
pytest -q
```

## Docker (One-Command Startup)

Build and run:

```bash
docker build -t phishing-detector .
docker run --rm -p 5001:5001 phishing-detector
```

Then open `http://127.0.0.1:5001`.

## Notes

- The dataset file is `backend/CEAS_08.csv`.
- Ensure `backend/output/best_phishing_model.pkl` exists before using the web app.
- Phase 3 adds engineered signals such as URL counts, suspicious TLD usage, IP-host URLs, sender-vs-URL domain matching, and urgency markers.
