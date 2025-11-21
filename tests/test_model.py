# tests/test_model.py
import os
import json
import pickle
import numpy as np
import pytest


MODEL_PATH = "model/logreg_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
FEATURES_PATH = "model/feature_list.json"


def test_model_files_exist():
    assert os.path.exists(MODEL_PATH), "Model file not found"
    assert os.path.exists(ENCODER_PATH), "Label encoder file missing"


@pytest.mark.parametrize("path", [MODEL_PATH, ENCODER_PATH, FEATURES_PATH])
def test_artifacts_exist(path):
    assert os.path.exists(path), f"Missing artifact: {path}"


def test_model_load_and_predict():
    # Load artifacts
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        feature_list = json.load(f)

    # Check feature length
    assert len(feature_list) == 48, "Expected 48 features in feature_list.json"

    # Dummy input (one sample, 48 features)
    x_dummy = np.zeros((1, 48))
    y_pred = model.predict(x_dummy)

    # Check prediction shape and class mapping
    assert y_pred.shape == (1,)
    label = encoder.inverse_transform(y_pred)[0]
    assert isinstance(label, str)
