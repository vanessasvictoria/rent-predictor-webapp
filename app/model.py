from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

TARGET = "price_chf"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_listings.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_and_save_model() -> Pipeline:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Training CSV must include '{TARGET}'")

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    pipe = _build_pipeline(X)
    pipe.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


_model_cache: Pipeline | None = None


def get_model() -> Pipeline:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if MODEL_PATH.exists():
        _model_cache = joblib.load(MODEL_PATH)
        return _model_cache

    _model_cache = train_and_save_model()
    return _model_cache


def predict_one(features: dict) -> float:
    model = get_model()
    X = pd.DataFrame([features])
    pred = float(model.predict(X)[0])
    return float(np.round(pred, 2))
