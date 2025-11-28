import os
import pandas as pd
import numpy as np
from utils.config import dir_data, dir_cleancsv
from utils.cleandata import split_and_clean, make_citations_unique  # import new function

# --- Load data ---
data_path = dir_data / "20251128_Elicitrevised.csv"
df = pd.read_csv(data_path, sep=";")
print(f"Loaded {len(df)} records from {data_path.name}")

# --- Normalize column names ---
df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)

# --- Clean and standardize text fields ---
text_cols = [
    "cohort", "gait_task", "dual_layer_cap",
    "type_of_eeg_electrodes", "gait_measurement_system",
    "artifactrej_methods", "step_keywords", "outcome_keywords_script"
]

for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

# --- Replace invalid placeholders with NaN ---
df.replace(["", "nan", "none", "NaN", "None", "NULL"], np.nan, inplace=True)

# --- Drop empty or exact duplicate rows ---
df = df.drop_duplicates().dropna(how="all")

# --- Make citations unique to preserve multiple studies by same author/year ---
if "citation" in df.columns:
    df["citation"] = make_citations_unique(df["citation"])
else:
    print("Warning: 'Citation' column not found; uniqueness not applied.")

print("Data standardized, cleaned, and citations made unique.")

# --- Apply split-and-clean function ---
df_artifact = split_and_clean(df, "artifactrej_methods")
df_steps = split_and_clean(df, "step_keywords")
df_outcomes = split_and_clean(df, "outcome_keywords_script")

# --- Summary of outputs ---
print("\nSummary of extracted entries:")
print(f"Artifact rejection entries: {len(df_artifact)}")
print(f"Step keyword entries:       {len(df_steps)}")
print(f"Outcome keyword entries:    {len(df_outcomes)}")

# --- Export cleaned tables ---
cleaned_files = {
    "Artifact_Methods_cleaned.csv": df_artifact,
    "Step_Keywords_cleaned.csv": df_steps,
    "Outcome_Keywords_cleaned.csv": df_outcomes,
}

for fname, table in cleaned_files.items():
    out_path = os.path.join(dir_cleancsv, fname)
    table.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

print("\nAll cleaned CSVs successfully exported.")

