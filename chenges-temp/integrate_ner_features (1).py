"""
integrate_ner_features.py
==========================
Merges the aggregated NER features with the project's primary training dataset.
Performs a left-join on (censuscode, iso_year, iso_week) and fills missing values with 0.

Usage:
    python integrate_ner_features.py

Output:
    ../Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting-main/Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting-main/data/training_dataset_with_ner.csv
"""

import pandas as pd
import os
import sys

# Avoid unicode print errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Configuration ──────────────────────────────────────────────────
# ner_features.csv lives in the same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NER_FEATURES_PATH = os.path.join(BASE_DIR, "ner_features.csv")

# The project’s data directory is one level up from ner-pipeline/
REAL_DATASET_DIR     = os.path.join(BASE_DIR, "..", "data")
PRIMARY_DATASET_PATH = os.path.join(REAL_DATASET_DIR, "training_dataset_enhanced_v3.csv")
OUTPUT_DATASET_PATH  = os.path.join(REAL_DATASET_DIR, "training_dataset_with_ner.csv")


def main():
    # Verify inputs exist
    if not os.path.exists(NER_FEATURES_PATH):
        print(f"❌ NER features file not found: {NER_FEATURES_PATH}")
        print("Please run ner_aggregator.py first.")
        sys.exit(1)

    if not os.path.exists(PRIMARY_DATASET_PATH):
        print(f"❌ Primary dataset not found: {PRIMARY_DATASET_PATH}")
        print("Please verify the project path or environment configuration.")
        sys.exit(1)

    print("Loading primary training dataset...")
    primary_df = pd.read_csv(PRIMARY_DATASET_PATH)
    print(f"Loaded {len(primary_df):,} rows from primary dataset.")
    print(f"Primary columns: {list(primary_df.columns)}")

    print("\nLoading NER features...")
    ner_df = pd.read_csv(NER_FEATURES_PATH)
    print(f"Loaded {len(ner_df):,} rows from NER features.")
    print(f"NER columns: {list(ner_df.columns)}")

    # Ensure keys have consistent types (e.g. integer)
    merge_keys = ["censuscode", "iso_year", "iso_week"]
    for key in merge_keys:
        primary_df[key] = primary_df[key].astype(int)
        ner_df[key] = ner_df[key].astype(int)

    print(f"\nMerging datasets on {merge_keys} using LEFT join...")
    merged_df = pd.merge(primary_df, ner_df, on=merge_keys, how="left")

    # NER columns to fill
    ner_cols = ["ner_symptoms", "ner_diseases", "ner_pathogens", "ner_travel", "ner_total_notes"]
    
    # Check NaN counts before filling
    print("\nMissing (NaN) value counts in merged NER columns before fillna:")
    for col in ner_cols:
        nan_count = merged_df[col].isna().sum()
        print(f"  {col}: {nan_count:,} / {len(merged_df):,} rows ({nan_count/len(merged_df)*100:.1f}%)")

    # Fill NaNs with 0 (since no notes means 0 clinical mentions of that category)
    merged_df[ner_cols] = merged_df[ner_cols].fillna(0)

    # Convert NER features to integer for clean representation
    for col in ner_cols:
        merged_df[col] = merged_df[col].astype(int)

    # Basic safety checks
    assert len(merged_df) == len(primary_df), "ERROR: Merged row count doesn't match original primary dataset!"

    print("\nMerged NER columns statistics:")
    print(merged_df[ner_cols].describe().loc[["mean", "min", "max"]])

    # Save output
    os.makedirs(REAL_DATASET_DIR, exist_ok=True)
    merged_df.to_csv(OUTPUT_DATASET_PATH, index=False)
    print(f"\n✅ Merged dataset successfully saved to: {OUTPUT_DATASET_PATH}")
    print(f"  Total rows: {len(merged_df):,}")
    print(f"  Total columns: {len(merged_df.columns)} (added {len(ner_cols)} NER features)")
    print(f"  File size: {os.path.getsize(OUTPUT_DATASET_PATH) / 1e6:.2f} MB")

if __name__ == "__main__":
    main()
