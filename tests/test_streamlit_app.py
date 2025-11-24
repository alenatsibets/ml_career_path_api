import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import importlib


def test_predict_top5_logic():
    import streamlit_app
    importlib.reload(streamlit_app)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([
        [0.1, 0.6, 0.2, 0.05, 0.05]
    ])

    def fake_inverse_transform(idxs):
        mapping = {0: "c0", 1: "c1", 2: "c2", 3: "c3", 4: "c4"}
        return [mapping[i] for i in idxs]

    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.side_effect = fake_inverse_transform

    df = pd.DataFrame([[0.1] * 48])

    labels, probs = streamlit_app.predict_top5(mock_model, mock_encoder, df)

    assert labels[0] == "c1"
    assert len(labels) == 5


def test_shap_pipeline_runs():
    """Test SHAP logic with mocks."""
    import streamlit_app
    importlib.reload(streamlit_app)

    mock_model = MagicMock()
    fake_input = pd.DataFrame([[0.2] * 48])

    with patch("shap.LinearExplainer") as mock_exp:
        mock_inst = MagicMock()
        mock_inst.shap_values.return_value = np.zeros((1, 48))
        mock_inst.expected_value = 0.123
        mock_exp.return_value = mock_inst

        value, shap_vals = streamlit_app.compute_shap(mock_model, fake_input)

        assert shap_vals.shape == (1, 48)
        assert value == 0.123
