# Rent Predictor Web App (Basel demo)

Small end-to-end demo: a simple web UI that predicts rent prices from a few listing features.
Built to show a clean “model → API → frontend” flow (not to maximize accuracy).

**Live demo** https://rent-predictor-webapp.onrender.com/

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

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000
API docs: http://127.0.0.1:8000/docs

Deployment notes (Render)
Free tier sleeps → first request can take ~30–60s (cold start).

Then commit + push:

```bash
git add README.md
git commit -m "Add local run + Render notes"
git push
```