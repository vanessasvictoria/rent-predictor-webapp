from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import predict_one, get_model, MODEL_PATH

app = FastAPI(title="Rent Predictor Web App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for demo; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "model_ready": MODEL_PATH.exists()},
    )


@app.post("/predict")
def predict(payload: dict):
    # expects keys: size_m2, rooms, zip_code, area, balcony, elevator, furnished
    pred = predict_one(payload)
    return {"predicted_price_chf": pred}


@app.post("/train")
def train():
    # Forces model to load/train and confirms readiness
    _ = get_model()
    return {"status": "ok", "model_saved": str(MODEL_PATH.exists())}
