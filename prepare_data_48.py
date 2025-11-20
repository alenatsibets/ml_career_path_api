# prepare_data_48.py
import pandas as pd
from rapidfuzz import process, fuzz
from util.logger import get_logger
from util.major_mapping import major_mapping
from util.categories_list import standardized_categories

logger = get_logger(__name__, log_file="prepare_data_48.log")

# --- 1. Load data ---
try:
    df = pd.read_csv("data/data.csv", sep="\t")
    logger.info(f"Data loaded successfully with shape {df.shape}")
except FileNotFoundError:
    logger.error("Data file not found. Make sure 'data/data.csv' exists.")
    raise

# --- 1.5 Remove duplicate raw rows ---
logger.info("Checking for duplicate rows in the raw dataset...")
raw_dups = df.duplicated().sum()
if raw_dups > 0:
    logger.warning(f"Found {raw_dups} duplicate rows. Removing...")
    df = df.drop_duplicates()
    logger.info(f"Raw duplicates removed. New shape: {df.shape}")
else:
    logger.info("No raw duplicates found.")

# --- 2. Select 48 RIASEC item columns + major ---
riasec_items = [
    # R
    'R1','R2','R3','R4','R5','R6','R7','R8',
    # I
    'I1','I2','I3','I4','I5','I6','I7','I8',
    # A
    'A1','A2','A3','A4','A5','A6','A7','A8',
    # S
    'S1','S2','S3','S4','S5','S6','S7','S8',
    # E
    'E1','E2','E3','E4','E5','E6','E7','E8',
    # C
    'C1','C2','C3','C4','C5','C6','C7','C8'
]

columns_to_keep = riasec_items + ['major']
df = df[columns_to_keep]
logger.info(f"Retained {len(df.columns)} columns (48 items + major).")

# --- 3. Drop rows with missing major or missing item responses ---
missing_major = df['major'].isna().sum()
logger.info(f"Found {missing_major} rows with missing 'major'. Dropping...")
df = df.dropna(subset=['major'])

missing_items = df[riasec_items].isna().any(axis=1).sum()
logger.info(f"Found {missing_items} rows with at least one missing RIASEC item. Dropping...")
df = df.dropna(subset=riasec_items)
logger.info(f"Shape after dropping missing: {df.shape}")

# --- 4. Normalize item scores from 1–5 to 0–1 ---
logger.info("Normalizing RIASEC items from 1–5 scale to 0–1...")
df[riasec_items] = (df[riasec_items] - 1) / 4.0

# --- 5. Clean and normalize 'major' text ---
logger.info("Cleaning and normalizing major names...")
df['major'] = df['major'].astype(str).str.lower().str.strip()
df['major'] = df['major'].str.replace(r'[^a-z ]', '', regex=True)
df = df[df['major'].str.strip() != '']
logger.info(f"Shape after major cleaning: {df.shape}")

# --- 6. Dictionary mapping ---
logger.info("Applying direct mapping from major dictionary...")
df['major_standard'] = df['major'].map(major_mapping)
df['major_standard'] = df['major_standard'].fillna(df['major'])

unchanged = df[df['major_standard'] == df['major']]
logger.info(f"Unmatched after dictionary mapping: {len(unchanged)} entries.")

# --- 7. Fuzzy mapping for remaining unmatched majors ---
logger.info("Starting fuzzy matching for remaining unmatched majors...")

def fuzzy_match_major(major, categories, threshold=70):
    if pd.isna(major) or not str(major).strip():
        return None
    match, score, _ = process.extractOne(major, categories, scorer=fuzz.WRatio)
    return match if score >= threshold else "Other"

unchanged['major_fuzzy'] = unchanged['major'].apply(
    lambda x: fuzzy_match_major(x, standardized_categories)
)
df.loc[df['major_standard'] == df['major'], 'major_standard'] = unchanged['major_fuzzy'].values

unmatched_final = df[df['major_standard'].isin(['Other', None])]
logger.info(f"Still unmatched after fuzzy: {len(unmatched_final)} rows")

# --- 8. Drop unmatched/Other and original major ---
logger.info("Removing uncategorized or unmatched entries...")
df = df[~df['major_standard'].isin(['Other', None])]
df = df.dropna(subset=['major_standard'])
df = df.drop(columns=['major'])
logger.info(f"Shape after removing unmatched & dropping 'major': {df.shape}")

# --- 9. Drop rare majors (< 3 samples) ---
value_counts = df['major_standard'].value_counts()
valid_classes = value_counts[value_counts > 2].index
rows_before = len(df)
df = df[df['major_standard'].isin(valid_classes)]
logger.info(f"Removed {rows_before - len(df)} rows belonging to majors with <= 2 samples.")
logger.info(f"Remaining majors: {len(valid_classes)} unique categories.")

# --- 10. Final duplicate check (just in case) ---
logger.info("Checking for duplicate rows in the final dataset...")
final_dups = df.duplicated().sum()
if final_dups > 0:
    logger.warning(f"Found {final_dups} duplicate rows in final data. Removing...")
    df = df.drop_duplicates()
    logger.info(f"Final duplicates removed. New shape: {df.shape}")
else:
    logger.info("No duplicates found in final dataset.")

# --- 11. Save final 48-feature dataset ---
df.to_csv("data/final_data_48.csv", index=False)
logger.info(f"Final 48-feature dataset saved → data/final_data_48.csv")
logger.info(f"Final dataset size: {len(df)} rows, Columns: {list(df.columns)}")
logger.info("Data preparation (48-feature) completed successfully!")
