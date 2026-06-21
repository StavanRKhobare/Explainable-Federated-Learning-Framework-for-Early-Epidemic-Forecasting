import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "final_datasets/master_dataset_clean.csv"
OUTPUT_FILE = "final_datasets/master_dataset_synthetic_dense.csv"
LOOKBACK = 4
NUM_TRUE_NEGATIVES = 10000

print("="*50)
print("Option B: Synthetic Data Generation (Local)")
print("="*50)

# 1. Load Original Data
print("Loading original dataset...")
df = pd.read_csv(INPUT_FILE)
original_count = len(df)
print(f"Original events: {original_count}")

# 2. Extract District Climate Profiles
print("Calculating seasonal climate profiles for synthesis...")
# Group by district and week to get the "average" weather for that week historically
# We use week_sin and week_cos if week is not explicitly available, but iso_week is.
climate_profile = df.groupby(['censuscode', 'iso_week']).agg({
    'temp_k': 'mean',
    'preci_mm': 'mean',
    'LAI': 'mean',
    'population_2024': 'first',
    'pop_density_per_km2_2024': 'first',
    'district': 'first',
    'state': 'first',
    'lat': 'first',
    'lon': 'first',
    'week_sin': 'first',
    'week_cos': 'first',
    'is_monsoon': 'mean' # will be rounded
}).reset_index()

# If some districts are missing data for some weeks, we will use district-level global averages as fallback
fallback_profile = df.groupby('censuscode').agg({
    'temp_k': 'mean', 'preci_mm': 'mean', 'LAI': 'mean',
    'population_2024': 'first', 'pop_density_per_km2_2024': 'first',
    'district': 'first', 'state': 'first', 'lat': 'first', 'lon': 'first'
}).reset_index()

# 3. Create a set of existing (district, year, week) to avoid overwriting real data
existing_records = set(zip(df['censuscode'], df['iso_year'], df['iso_week']))

# Helper to generate previous week (handles year boundaries roughly)
def get_prev_week(year, week, lag):
    new_week = week - lag
    new_year = year
    while new_week <= 0:
        new_year -= 1
        new_week += 52
    return new_year, new_week

# 4. Generate Lookbacks
print(f"Generating {LOOKBACK} weeks of lookback data for every event...")
lookback_records = []
for _, row in df.iterrows():
    c_code = row['censuscode']
    y = row['iso_year']
    w = row['iso_week']
    
    for lag in range(1, LOOKBACK + 1):
        prev_y, prev_w = get_prev_week(y, w, lag)
        if (c_code, prev_y, prev_w) not in existing_records:
            lookback_records.append({
                'censuscode': c_code,
                'iso_year': prev_y,
                'iso_week': prev_w,
                'cases': 0.0,
                'deaths': 0.0,
                'is_outbreak': 0.0,
                'cases_lag1': 0.0, 'cases_lag2': 0.0, 'cases_lag3': 0.0, 'cases_roll4w': 0.0,
                'growth_rate': 0.0, 'cases_per_100k': 0.0
            })
            existing_records.add((c_code, prev_y, prev_w))

# 5. Generate True Negatives
print(f"Generating {NUM_TRUE_NEGATIVES} random True Negatives (zero-cases background)...")
unique_codes = df['censuscode'].unique()
min_year, max_year = df['iso_year'].min(), df['iso_year'].max()

tn_records = []
while len(tn_records) < NUM_TRUE_NEGATIVES:
    c_code = np.random.choice(unique_codes)
    y = np.random.randint(min_year, max_year + 1)
    w = np.random.randint(1, 53)
    
    if (c_code, y, w) not in existing_records:
        tn_records.append({
            'censuscode': c_code,
            'iso_year': y,
            'iso_week': w,
            'cases': 0.0,
            'deaths': 0.0,
            'is_outbreak': 0.0,
            'cases_lag1': 0.0, 'cases_lag2': 0.0, 'cases_lag3': 0.0, 'cases_roll4w': 0.0,
            'growth_rate': 0.0, 'cases_per_100k': 0.0
        })
        existing_records.add((c_code, y, w))

# 6. Combine all new records
new_df = pd.DataFrame(lookback_records + tn_records)

# 7. Merge Synthetic Weather into the new records
print("Applying synthetic climate data to new records...")
# First try merging by district AND week
new_df = pd.merge(new_df, climate_profile, on=['censuscode', 'iso_week'], how='left')

# For rows that missed, merge from fallback (district global average)
missing_mask = new_df['temp_k'].isna()
if missing_mask.any():
    fallback_merge = pd.merge(new_df[missing_mask][['censuscode', 'iso_year', 'iso_week', 'cases', 'is_outbreak']], 
                              fallback_profile, on='censuscode', how='left')
    
    # Recalculate simple week_sin/cos for missing
    fallback_merge['week_sin'] = np.sin(2 * np.pi * fallback_merge['iso_week'] / 52.0)
    fallback_merge['week_cos'] = np.cos(2 * np.pi * fallback_merge['iso_week'] / 52.0)
    fallback_merge['is_monsoon'] = fallback_merge['iso_week'].apply(lambda w: 1 if 24 <= w <= 36 else 0)
    
    # Update back
    for col in fallback_merge.columns:
        if col in new_df.columns and col not in ['censuscode', 'iso_year', 'iso_week', 'cases', 'is_outbreak']:
            new_df.loc[missing_mask, col] = fallback_merge[col].values

# Ensure is_monsoon is rounded
if 'is_monsoon' in new_df.columns:
    new_df['is_monsoon'] = new_df['is_monsoon'].round()

# 8. Combine with Original
final_df = pd.concat([df, new_df], ignore_index=True)

# Sort strictly by time and district
final_df = final_df.sort_values(["censuscode", "iso_year", "iso_week"]).reset_index(drop=True)

print(f"Done! Final dataset has {len(final_df)} rows (Original: {original_count}, Added: {len(new_df)}).")
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")
