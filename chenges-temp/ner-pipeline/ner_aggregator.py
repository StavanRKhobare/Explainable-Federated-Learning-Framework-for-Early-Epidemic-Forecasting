"""
ner_aggregator.py
=================
Loads synthetic clinical notes, processes them with NERExtractor,
and aggregates the extracted features per (censuscode, iso_year, iso_week).

Usage:
    python ner_aggregator.py

Output:
    ner_features.csv  — aggregated counts per district-week
"""

import pandas as pd
import os
import sys
from collections import Counter

# Avoid unicode print errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from ner_pipeline import NERExtractor

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_CSV_PATH = os.path.join(BASE_DIR, "synthetic_notes.csv")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "ner_features.csv")

def main():
    if not os.path.exists(NOTES_CSV_PATH):
        print(f"❌ Synthetic notes not found at {NOTES_CSV_PATH}")
        print("Please run generate_synthetic_notes.py first.")
        sys.exit(1)

    print("Loading synthetic notes...")
    df = pd.read_csv(NOTES_CSV_PATH)
    print(f"Loaded {len(df):,} notes.")

    print("Initializing NER Extractor...")
    try:
        extractor = NERExtractor()
    except Exception as e:
        print(f"❌ Failed to load NERExtractor: {e}")
        sys.exit(1)

    print("Extracting entities from notes (this might take a few minutes)...")
    
    # Store extracted counts
    symptom_counts = []
    disease_counts = []
    pathogen_counts = []
    travel_counts = []
    
    # Process each note
    total_notes = len(df)
    for idx, row in df.iterrows():
        note_text = str(row["note"])
        counts = extractor.extract(note_text)
        
        symptom_counts.append(counts.get("SYMPTOM", 0))
        disease_counts.append(counts.get("DISEASE", 0))
        pathogen_counts.append(counts.get("PATHOGEN", 0))
        travel_counts.append(counts.get("TRAVEL", 0))
        
        # Periodic logging
        if (idx + 1) % 100 == 0 or (idx + 1) == total_notes:
            print(f"  Processed {idx + 1}/{total_notes} notes...")

    # Assign raw features back to DataFrame
    df["ner_symptoms"] = symptom_counts
    df["ner_diseases"] = disease_counts
    df["ner_pathogens"] = pathogen_counts
    df["ner_travel"] = travel_counts
    # Count each note as 1 to aggregate note volume
    df["ner_total_notes"] = 1

    print("\nAggregating entity counts by censuscode, iso_year, and iso_week...")
    # Group by censuscode, iso_year, and iso_week and sum all counts
    agg_df = df.groupby(["censuscode", "iso_year", "iso_week"], as_index=False).agg({
        "ner_symptoms": "sum",
        "ner_diseases": "sum",
        "ner_pathogens": "sum",
        "ner_travel": "sum",
        "ner_total_notes": "sum"
    })

    print(f"Aggregation complete. Result has {len(agg_df):,} district-week rows.")
    
    # Show preview
    print("\nFeature aggregations preview:")
    print(agg_df.head(10))

    # Save aggregated features
    agg_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Aggregated features saved to: {OUTPUT_CSV_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_CSV_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
