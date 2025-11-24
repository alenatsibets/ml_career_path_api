import streamlit as st
import pandas as pd
import numpy as np
import joblib
import functools


@functools.lru_cache()
def load_model():
    return joblib.load("model/logreg_model.pkl")


@functools.lru_cache()
def load_encoder():
    return joblib.load("model/label_encoder.pkl")


# ---------------- REAL RIASEC ITEM DESCRIPTIONS ---------------- #
ITEMS = {
    # Realistic
    "R1": "Test the quality of parts before shipment",
    "R2": "Lay brick or tile",
    "R3": "Work on an offshore oil-drilling rig",
    "R4": "Assemble electronic parts",
    "R5": "Operate a grinding machine in a factory",
    "R6": "Fix a broken faucet",
    "R7": "Assemble products in a factory",
    "R8": "Install flooring in houses",

    # Investigative
    "I1": "Study the structure of the human body",
    "I2": "Study animal behavior",
    "I3": "Do research on plants or animals",
    "I4": "Develop a new medical treatment or procedure",
    "I5": "Conduct biological research",
    "I6": "Study whales or marine life",
    "I7": "Work in a biology lab",
    "I8": "Make a map of the bottom of an ocean",

    # Artistic
    "A1": "Conduct a musical choir",
    "A2": "Direct a play",
    "A3": "Design artwork for magazines",
    "A4": "Write a song",
    "A5": "Write books or plays",
    "A6": "Play a musical instrument",
    "A7": "Perform stunts for a movie or TV show",
    "A8": "Design sets for plays",

    # Social
    "S1": "Give career guidance to people",
    "S2": "Do volunteer work at a non-profit organization",
    "S3": "Help people with alcohol or drug problems",
    "S4": "Teach an individual an exercise routine",
    "S5": "Help people with family problems",
    "S6": "Supervise activities of children at a camp",
    "S7": "Teach children how to read",
    "S8": "Help elderly people with daily activities",

    # Enterprising
    "E1": "Sell restaurant franchises",
    "E2": "Sell merchandise at a department store",
    "E3": "Manage hotel operations",
    "E4": "Operate a beauty salon or barber shop",
    "E5": "Manage a department within a company",
    "E6": "Manage a clothing store",
    "E7": "Sell houses",
    "E8": "Run a toy store",

    # Conventional
    "C1": "Generate monthly payroll checks",
    "C2": "Inventory supplies using a handheld computer",
    "C3": "Use a computer program to generate bills",
    "C4": "Maintain employee records",
    "C5": "Compute and record statistical data",
    "C6": "Operate a calculator",
    "C7": "Handle customer bank transactions",
    "C8": "Keep shipping and receiving records",
}


def predict_top5(model, encoder, input_norm):
    """Pure function: normalized input → top5 labels + probabilities."""
    probs = model.predict_proba(input_norm)[0]
    idx = np.argsort(probs)[-5:][::-1]
    return encoder.inverse_transform(idx), probs[idx]


# ---------------- STREAMLIT UI (NOT USED IN UNIT TESTS) ---------------- #
def run_app():
    st.set_page_config(page_title="Career Path Predictor", layout="wide")

    st.title("Career Path Prediction App")

    st.markdown("""
    ### Instructions  
    The following items were rated on a **1–5 scale** based on how much  
    you would like to perform that task:  

    - **1 = Dislike**  
    - **3 = Neutral**  
    - **5 = Enjoy**  
    """)

    model = load_model()
    encoder = load_encoder()

    st.subheader("Rate Each Activity:")

    cols = st.columns(3)
    user_data = {}

    # Sliders default to 1 as requested
    for idx, (item, desc) in enumerate(ITEMS.items()):
        col = cols[idx % 3]
        with col:
            user_data[item] = st.slider(f"{item}: {desc}", 1, 5, 1)

    input_df = pd.DataFrame([user_data])
    input_norm = (input_df - 1) / 4

    if st.button("Predict Career Path"):
        st.subheader("Top-5 Predicted Career Paths")

        labels, probs = predict_top5(model, encoder, input_norm)

        for label, p in zip(labels, probs):
            st.write(f"**{label}** — {p*100:.2f}%")


if __name__ == "__main__":
    run_app()
