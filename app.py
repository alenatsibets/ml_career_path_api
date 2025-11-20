# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import json
import numpy as np
import pandas as pd
from util.logger import get_logger

logger = get_logger(__name__, log_file="app.log")

# Load model, encoder, feature list
try:
    with open("model/logreg_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("model/feature_list.json", "r") as f:
        feature_list = json.load(f)

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Could not load model files")


app = FastAPI(title="Career Path Prediction API", version="1.0")


# Request schema
class UserRIASEC(BaseModel):
    """User provides 48 RIASEC inputs"""
    features: list[float] = Field(
        ..., description="List of 48 RIASEC feature values (0–1)."
    )

    def as_dataframe(self):
        return pd.DataFrame([self.features], columns=feature_list)


# Routes
@app.get("/")
def root():
    return {"message": "Career Path Prediction API is running!"}


@app.post("/predict")
def predict_major(data: UserRIASEC):
    try:
        x = data.as_dataframe()

        # Prediction
        pred = model.predict(x)[0]
        pred_label = encoder.inverse_transform([pred])[0]

        # Top-5 classes
        probas = model.predict_proba(x)[0]
        top5_idx = np.argsort(probas)[-5:][::-1]
        top5_labels = encoder.inverse_transform(top5_idx).tolist()
        top5_probs = probas[top5_idx].round(3).tolist()

        response = {
            "predicted_major": pred_label,
            "top_5_predictions": [
                {"major": m, "probability": p}
                for m, p in zip(top5_labels, top5_probs)
            ]
        }

        logger.info(f"Prediction success | Input={data.features} | Output={response}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Invalid input format.")
