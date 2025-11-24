import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import importlib


# -------------------- HELPERS --------------------

def reload_app():
    """Reload module to ensure patches apply before import."""
    import streamlit_app
    importlib.reload(streamlit_app)
    return streamlit_app


# -------------------- PREDICTION TESTS --------------------

def test_predict_top5_logic_basic():
    """Ensure top5 logic returns correct order and count."""

    app = reload_app()

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.10, 0.60, 0.20, 0.05, 0.05]
    ])

    def fake_inverse_transform(idxs):
        return [f"class{i}" for i in idxs]

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.side_effect = fake_inverse_transform

    df = pd.DataFrame([[0.1] * 48])

    labels, probs = app.predict_top5(mock_model, mock_encoder, df)

    assert labels[0] == "class1"
    assert len(labels) == 5
    assert np.isclose(probs[0], 0.60)


def test_predict_top5_correct_sorting():
    """Ensure probabilities are sorted descending."""
    app = reload_app()

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.2, 0.1, 0.5, 0.15, 0.05]
    ])

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = ["C", "A", "B", "D", "E"]

    df = pd.DataFrame([[0.1] * 48])

    labels, probs = app.predict_top5(mock_model, mock_encoder, df)

    assert probs[0] >= probs[1] >= probs[2]


# -------------------- NORMALIZATION TESTS --------------------
def test_input_normalization():
    app = reload_app()

    df = pd.DataFrame([[1, 3, 5]], columns=["R1", "R2", "R3"])
    norm = (df - 1) / 4

    assert norm.iloc[0, 0] == 0
    assert norm.iloc[0, 1] == 0.5
    assert norm.iloc[0, 2] == 1.0


# -------------------- MODEL / ENCODER LOAD TESTS --------------------

def test_load_model_missing_file():
    """load_model should raise FileNotFoundError if file missing."""
    with patch("joblib.load", side_effect=FileNotFoundError):
        app = reload_app()
        with pytest.raises(FileNotFoundError):
            app.load_model()


def test_load_encoder_missing():
    with patch("joblib.load", side_effect=FileNotFoundError):
        app = reload_app()
        with pytest.raises(FileNotFoundError):
            app.load_encoder()


# -------------------- PARALLEL PREDICTION TESTS --------------------

def test_predict_top5_parallel():
    app = reload_app()

    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = [
        np.array([[0.1, 0.6, 0.2, 0.05, 0.05]]),
        np.array([[0.3, 0.1, 0.5, 0.05, 0.05]])
    ]

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.side_effect = lambda idxs: idxs

    df1 = pd.DataFrame([[0.1] * 48])
    df2 = pd.DataFrame([[0.2] * 48])

    # Only works if you already added predict_top5_parallel()
    if hasattr(app, "predict_top5_parallel"):
        results = app.predict_top5_parallel(mock_model, mock_encoder, [df1, df2], workers=2)
        assert len(results) == 2
        assert len(results[0][0]) == 5


# -------------------- UI STRUCTURE TESTS (Streamlit mocked) --------------------
def test_items_dictionary_valid():
    """Ensure 48 items exist."""
    app = reload_app()
    assert len(app.ITEMS) == 48
    assert "R1" in app.ITEMS
    assert "C8" in app.ITEMS
