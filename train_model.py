# train_model.py
import pandas as pd
import pickle
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from util.logger import get_logger

logger = get_logger(__name__, log_file="train_model.log")

DATA_PATH = "data/final_data_48.csv"
MODEL_PATH = "model/logreg_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
FEATURES_PATH = "model/feature_list.json"


def main():

    logger.info("==== Starting model training pipeline ====")
    start_time = time.time()

    # Load dataset
    try:
        logger.info(f"Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded successfully — rows={len(df)}, cols={len(df.columns)}")
    except Exception as e:
        logger.exception(f"Failed to load dataset: {e}")
        return

    # Select features
    feature_cols = [c for c in df.columns if c.startswith(tuple("RIASEC"))]
    logger.info(f"Detected {len(feature_cols)} RIASEC features")

    x = df[feature_cols]
    y = df["major_standard"]

    # Encode labels
    try:
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        logger.info(f"Label encoding complete — classes={len(encoder.classes_)}")
    except Exception as e:
        logger.exception(f"Label encoding failed: {e}")
        return

    # Split data
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        logger.info(f"Data split complete — train={len(x_train)}, test={len(x_test)}")
    except Exception as e:
        logger.exception(f"Data splitting failed: {e}")
        return

    # Train model
    try:
        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500
        )
        model.fit(x_train, y_train)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.exception(f"Model training failed: {e}")
        return

    # Save model + encoder + features
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")

        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)
        logger.info(f"Label encoder saved to {ENCODER_PATH}")

        with open(FEATURES_PATH, "w") as f:
            json.dump(feature_cols, f)
        logger.info(f"Feature list saved to {FEATURES_PATH}")

    except Exception as e:
        logger.exception(f"Saving model artifacts failed: {e}")
        return

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"==== Training pipeline completed in {elapsed} seconds ====")


if __name__ == "__main__":
    main()
