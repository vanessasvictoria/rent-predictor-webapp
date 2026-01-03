# Rent Predictor Web App (Basel demo)

Small end-to-end demo: a simple web UI that predicts rent prices from a few listing features.
Built to show a clean “model → API → frontend” flow (not to maximize accuracy).

## What’s inside
- **Backend:** Python (FastAPI/Flask) serving a prediction endpoint
- **Model:** scikit-learn (toy demo model)
- **Frontend:** HTML + CSS (simple form)

## Repo structure
- `app/` – backend app (routes + model loading)
- `templates/` – HTML templates
- `static/` – CSS
- `data/` – tiny sample CSVs for demo
- `models/` – saved model artifact (demo)

## Quickstart (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app/main.py