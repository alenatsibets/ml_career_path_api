import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import json
import functools


@functools.lru_cache()
def load_model():
    return joblib.load("model/logreg_model.pkl")


@functools.lru_cache()
def load_encoder():
    return joblib.load("model/label_encoder.pkl")


@functools.lru_cache()
def load_features():
    try:
        with open("model/feature_list.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return [f"F{i}" for i in range(48)]

# ---------------- RIASEC Items ----------------
ITEMS = {
    "R1": "Test the quality of parts before shipment",
    "R2": "Lay brick or tile",
    "R3": "Work on an offshore oil-drilling rig",
    "R4": "Assemble electronic parts",
    "R5": "Operate a grinding machine in a factory",
    "R6": "Fix a broken faucet",
    "R7": "Assemble products in a factory",
    "R8": "Install flooring in houses",

    "I1": "Study the structure of the human body",
    "I2": "Study animal behavior",
    "I3": "Do research on plants or animals",
    "I4": "Develop a new medical treatment or procedure",
    "I5": "Conduct biological research",
    "I6": "Study whales and other marine life",
    "I7": "Work in a biology lab",
    "I8": "Make a map of the bottom of an ocean",

    "A1": "Conduct a musical choir",
    "A2": "Direct a play",
    "A3": "Design artwork for magazines",
    "A4": "Write a song",
    "A5": "Write books or plays",
    "A6": "Play a musical instrument",
    "A7": "Perform stunts for a movie or TV show",
    "A8": "Design sets for plays",

    "S1": "Give career guidance to people",
    "S2": "Do volunteer work at a non-profit organization",
    "S3": "Help people with drugs or alcohol problems",
    "S4": "Teach an individual an exercise routine",
    "S5": "Help people with family problems",
    "S6": "Supervise activities of children at a camp",
    "S7": "Teach children how to read",
    "S8": "Help elderly people with daily activities",

    "E1": "Sell restaurant franchises to individuals",
    "E2": "Sell merchandise at a department store",
    "E3": "Manage the operations of a hotel",
    "E4": "Operate a beauty salon or barber shop",
    "E5": "Manage a department in a company",
    "E6": "Manage a clothing store",
    "E7": "Sell houses",
    "E8": "Run a toy store",

    "C1": "Generate monthly payroll checks for an office",
    "C2": "Inventory supplies using a hand-held computer",
    "C3": "Use a computer program to generate bills",
    "C4": "Maintain employee records",
    "C5": "Compute and record statistical data",
    "C6": "Operate a calculator",
    "C7": "Handle customer bank transactions",
    "C8": "Keep shipping and receiving records"
}


def predict_top5(model, encoder, input_norm):
    """Pure function: given normalized input -> return labels, probs."""
    probs = model.predict_proba(input_norm)[0]
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_labels = encoder.inverse_transform(top5_idx)
    top5_probs = probs[top5_idx]
    return top5_labels, top5_probs


def compute_shap(model, input_norm):
    """Pure SHAP pipeline (testable with mocks)."""
    explainer = shap.LinearExplainer(model, input_norm)
    shap_values = explainer.shap_values(input_norm)
    return explainer.expected_value, shap_values


# ------------------ STREAMLIT UI (NOT RUN IN TESTS) ------------------ #
def run_app():
    st.set_page_config(page_title="Career Path Predictor", layout="wide")

    st.title("🎓 Career Path Prediction App")

    features = load_features()
    model = load_model()
    encoder = load_encoder()

    ITEMS = {f"R{i}": f"Question {i}" for i in range(1, 49)}  # simplified here

    cols = st.columns(3)
    user_data = {}

    for idx, (item, desc) in enumerate(ITEMS.items()):
        col = cols[idx % 3]
        with col:
            user_data[item] = st.slider(f"{item}: {desc}", 1, 5, 3)

    input_df = pd.DataFrame([user_data])
    input_norm = (input_df - 1) / 4

    if st.button("Predict"):
        labels, probs = predict_top5(model, encoder, input_norm)

        for l, p in zip(labels, probs):
            st.write(f"{l}: {p:.2f}")

        exp_value, shap_vals = compute_shap(model, input_norm)
        st.write("SHAP explanation:")
        st.write(shap_vals)


if __name__ == "__main__":
    run_app()