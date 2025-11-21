# prepare_data_48.py
import os
import pandas as pd
from rapidfuzz import process, fuzz
from util.logger import get_logger
from util.major_mapping import major_mapping
from util.categories_list import standardized_categories

logger = get_logger(__name__, "prepare_data_48.log")


def run_prepare_data_48(input_path="data/data.csv", output_path="data/final_data_48.csv", test_mode=False):
    df = pd.read_csv(input_path, sep="\t")
    df = df.drop_duplicates()

    riasec_items = [f"{c}{i}" for c in "RIASEC" for i in range(1,9)]
    existing = [c for c in riasec_items + ["major"] if c in df.columns]
    df = df[existing].dropna()

    present_items = [c for c in riasec_items if c in df.columns]
    df[present_items] = (df[present_items] - 1) / 4

    df["major"] = (
        df["major"].astype(str)
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.strip()
    )
    df = df[df["major"] != ""]

    df["major_standard"] = df["major"].map(major_mapping).fillna(df["major"])
    unchanged = df[df["major_standard"] == df["major"]]

    if not test_mode:
        def fuzz_match(x):
            match, score, _ = process.extractOne(x, standardized_categories, scorer=fuzz.WRatio)
            return match if score >= 70 else "Other"
        df.loc[unchanged.index, "major_standard"] = unchanged["major"].apply(fuzz_match)

    if not test_mode:
        df = df[df["major_standard"] != "Other"]
        vc = df["major_standard"].value_counts()
        df = df[df["major_standard"].isin(vc[vc > 2].index)]

    df = df.drop_duplicates()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    run_prepare_data_48()
