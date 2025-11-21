# prepare_data.py
import os
import pandas as pd
from rapidfuzz import process, fuzz
from util.logger import get_logger
from util.major_mapping import major_mapping
from util.categories_list import standardized_categories

logger = get_logger(__name__, log_file="prepare_data.log")


def run_prepare_data(input_path="data/data.csv", output_path="data/final_data.csv", test_mode=False):
    """
    Full data cleaning pipeline.
    Returns final cleaned dataframe.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load data
    df = pd.read_csv(input_path, sep="\t")
    logger.info(f"Loaded dataset: {df.shape}")

    # 2. Remove duplicate raw rows
    raw_dups = df.duplicated().sum()
    if raw_dups > 0:
        logger.warning(f"Found {raw_dups} duplicates. Removing...")
        df = df.drop_duplicates()

    # 3. Expected RIASEC columns
    riasec_columns = {
        'R': ['R1','R2','R3','R4','R5','R6','R7','R8'],
        'I': ['I1','I2','I3','I4','I5','I6','I7','I8'],
        'A': ['A1','A2','A3','A4','A5','A6','A7','A8'],
        'S': ['S1','S2','S3','S4','S5','S6','S7','S8'],
        'E': ['E1','E2','E3','E4','E5','E6','E7','E8'],
        'C': ['C1','C2','C3','C4','C5','C6','C7','C8']
    }
    expected_cols = [c for cols in riasec_columns.values() for c in cols] + ["major"]

    # keep only existing
    existing_cols = [c for c in expected_cols if c in df.columns]
    df = df[existing_cols]

    # 4. Drop missing rows
    df = df.dropna()
    logger.info(f"After dropping missing: {df.shape}")

    # 5. Compute percentages (1–5 → 0–1)
    for key, cols in riasec_columns.items():
        cols = [c for c in cols if c in df.columns]
        if cols:
            df[key + "_pct"] = (df[cols].mean(axis=1) - 1) / 4

    df = df[[c for c in df.columns if c.endswith("_pct")] + ["major"]]

    # 6. Clean major
    df["major"] = (
        df["major"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.strip()
    )

    # 7. Dictionary mapping
    df["major_standard"] = df["major"].map(major_mapping).fillna(df["major"])

    # 8. Fuzzy matching
    if not test_mode:
        def fuzzy_match(x):
            match, score, _ = process.extractOne(x, standardized_categories, scorer=fuzz.WRatio)
            return match if score >= 70 else "Other"

        unmatched = df["major_standard"] == df["major"]
        df.loc[unmatched, "major_standard"] = df.loc[unmatched, "major"].apply(fuzzy_match)

    # 9. Remove "Other" + rare classes
    if not test_mode:
        df = df[df["major_standard"] != "Other"]
        vc = df["major_standard"].value_counts()
        df = df[df["major_standard"].isin(vc[vc > 2].index)]

    # 10. Final cleanup
    df = df.drop_duplicates()

    # 11. Save result
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path}")

    return df


if __name__ == "__main__":
    run_prepare_data()
