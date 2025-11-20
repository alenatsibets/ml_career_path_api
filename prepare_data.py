# prepare_data.py
import pandas as pd
from rapidfuzz import process, fuzz
from util.logger import get_logger
from util.major_mapping import major_mapping
from util.categories_list import standardized_categories


logger = get_logger(__name__, log_file="prepare_data.log")

# --- 1. Load data ---
try:
    df = pd.read_csv("data/data.csv", sep="\t")
    logger.info(f"Data loaded successfully with shape {df.shape}")
except FileNotFoundError:
    logger.error("Data file not found. Make sure 'data/data.csv' exists.")
    raise


# --- 2. Check and remove duplicate rows ---
logger.info("Checking for duplicate rows in the raw dataset...")

total_rows = len(df)
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    logger.warning(f"Found {duplicate_count} duplicate rows in the raw data. Removing duplicates...")
    df = df.drop_duplicates()
    logger.info(f"Removed {duplicate_count} duplicates. Dataset reduced to {len(df)} rows.")
else:
    logger.info("No duplicate rows found in the raw dataset.")


# --- 3. Choose RIASEC columns ---
riasec_columns = [
    # Realistic
    'R1','R2','R3','R4','R5','R6','R7','R8',
    # Investigative
    'I1','I2','I3','I4','I5','I6','I7','I8',
    # Artistic
    'A1','A2','A3','A4','A5','A6','A7','A8',
    # Social
    'S1','S2','S3','S4','S5','S6','S7','S8',
    # Enterprising
    'E1','E2','E3','E4','E5','E6','E7','E8',
    # Conventional
    'C1','C2','C3','C4','C5','C6','C7','C8'
]
columns_to_keep = riasec_columns + ['major']
df_filtered = df[columns_to_keep]
logger.info(f"Retained {len(df_filtered.columns)} columns for processing")


# --- 4. Clean empty values ---
missing_before = df_filtered['major'].isna().sum()
logger.info(f"Found {missing_before} rows with missing 'major' values.")
df_cleaned = df_filtered.dropna(subset=['major'])
logger.info(f"Dropped missing majors: {len(df)} → {len(df_cleaned)} rows remaining.")


# --- 5. Calculate RIASEC percentage ---
riasec_columns = {
    'R': ['R1','R2','R3','R4','R5','R6','R7','R8'],
    'I': ['I1','I2','I3','I4','I5','I6','I7','I8'],
    'A': ['A1','A2','A3','A4','A5','A6','A7','A8'],
    'S': ['S1','S2','S3','S4','S5','S6','S7','S8'],
    'E': ['E1','E2','E3','E4','E5','E6','E7','E8'],
    'C': ['C1','C2','C3','C4','C5','C6','C7','C8']
}


def calculate_percentage(row, cols):
    return (row[cols].mean() - 1) / 4  # normalize from 1–5 scale to 0–1


logger.info("Calculating RIASEC percentage scores...")
for key, cols in riasec_columns.items():
    df_cleaned[key + '_pct'] = df_cleaned.apply(lambda row: calculate_percentage(row, cols), axis=1)
columns_to_keep = [k + '_pct' for k in riasec_columns.keys()] + ['major']
df_final = df_cleaned[columns_to_keep]
logger.info("RIASEC percentage columns added.")


# --- 6. Cleaning ---
logger.info("Cleaning and normalizing major names...")
df_final['major'] = df_final['major'].astype(str).str.lower().str.strip()
df_final['major'] = df_final['major'].str.replace(r'[^a-z ]', '', regex=True)
df_cleaned_m = df_final[df_final['major'].str.strip() != ''].copy()
logger.info(f"Size after cleaning: {len(df_cleaned_m)} rows")


# --- 7. Dictionary Data Mapping ---
logger.info("Applying direct mapping from major dictionary...")
df_cleaned_m['major_standard'] = df_cleaned_m['major'].map(major_mapping)
df_cleaned_m['major_standard'] = df_cleaned_m['major_standard'].fillna(df_cleaned_m['major'])

df_unchanged = df_cleaned_m[df_cleaned_m['major_standard'] == df_cleaned_m['major']]
logger.info(f"Unmatched after dictionary mapping: {len(df_unchanged)} entries")


# --- 8. Fuzzy Mapping ---
logger.info("Starting fuzzy matching for remaining unmatched majors...")


def fuzzy_match_major(major, categories, threshold=70):
    if pd.isna(major) or not str(major).strip():
        return None
    match, score, _ = process.extractOne(major, categories, scorer=fuzz.WRatio)
    return match if score >= threshold else "Other"


df_unchanged['major_fuzzy'] = df_unchanged['major'].apply(lambda x: fuzzy_match_major(x, standardized_categories))
df_cleaned_m.loc[df_cleaned_m['major_standard'] == df_cleaned_m['major'], 'major_standard'] = df_unchanged['major_fuzzy'].values

unmatched_final = df_cleaned_m[df_cleaned_m['major_standard'].isin(['Other', None])]
logger.info(f"Still unmatched after fuzzy: {len(unmatched_final)} rows")


# --- 9. Drop unmatched ---
logger.info("Removing uncategorized or unmatched entries...")
df_final = df_cleaned_m[~df_cleaned_m['major_standard'].isin(['Other', None])]
df_final = df_final.dropna(subset=['major_standard'])
df_final = df_final.drop(columns=['major'])
logger.info(f"Pre-final dataset size: {len(df_final)} rows, Columns: {list(df_final.columns)}")


# --- 10. Drop rear ---
value_counts = df_final['major_standard'].value_counts()

# Keep only majors with at least 3 samples
valid_classes = value_counts[value_counts > 2].index
df_filtered = df_final[df_final['major_standard'].isin(valid_classes)]

logger.info(f"Removed {len(df_final) - len(df_filtered)} rows belonging to rare majors.")
logger.info(f"Remaining majors after filtering: {len(valid_classes)} unique categories.")

df_filtered.to_csv("data/final_data.csv", index=False)
logger.info(f"Final dataset saved → data/final_data.csv")
logger.info(f"Final dataset size: {len(df_filtered)} rows, Columns: {list(df_filtered.columns)}")

unique_majors = df_filtered['major_standard'].unique()
logger.info(f"Unique standardized majors: {len(unique_majors)}")

logger.info("Data preparation completed successfully!")