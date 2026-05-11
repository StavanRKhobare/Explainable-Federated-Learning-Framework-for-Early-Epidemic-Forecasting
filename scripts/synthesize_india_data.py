import pandas as pd
import numpy as np

print("Loading datasets...")
district_file = r'c:\4th sem el\code\final_datasets\training_dataset_real_weather.csv'
state_file = r'C:\4th sem el\one more trial India\masterDB_V1.2.csv'
output_file = r'c:\4th sem el\code\final_datasets\training_dataset_enhanced_v2.csv'

df_dist = pd.read_csv(district_file)
df_state = pd.read_csv(state_file)

# Ensure cases is numeric
df_dist['cases'] = pd.to_numeric(df_dist['cases'], errors='coerce').fillna(0)

# 1. State Mapping to match names
state_map = {
    'Orissa': 'Odisha',
    'J & K': 'Jammu and Kashmir',
    'J  &  K': 'Jammu and Kashmir',
    'Pondicherry': 'Puducherry',
    'Uttrakhand': 'Uttarakhand',
    'D&N Haveli': 'Dadra and Nagar Haveli',
    'Daman & Diu': 'Daman and Diu',
    'Arunachal\nPradesh': 'Arunachal Pradesh',
    'A&N Island': 'Andaman and Nicobar'
}

# 2. Process State Data (masterDB)
df_state['dengue_total'] = pd.to_numeric(df_state['dengue_total'], errors='coerce').fillna(0)
df_state['Year'] = pd.to_datetime(df_state['calendar_start_date'], errors='coerce').dt.year
df_state = df_state.dropna(subset=['Year', 'adm_1_name'])
df_state['Year'] = df_state['Year'].astype(int)
df_state['State'] = df_state['adm_1_name'].replace(state_map)

# Get the official maximum total for each State-Year
official_totals = df_state.groupby(['Year', 'State'])['dengue_total'].max().reset_index()
official_totals.columns = ['iso_year', 'state', 'official_total']

# 3. Calculate "Risk Score" for each district-week in the training data
print("Calculating Climate-Population Risk Scores...")
temp_factor = np.exp(-((df_dist['temp_k'] - 300) / 7)**2) 
precip_factor = np.log1p(df_dist['preci_mm']) 
pop_factor = np.log1p(df_dist['pop_density_per_km2_2024'])

df_dist['risk_score'] = temp_factor * precip_factor * pop_factor
df_dist['risk_score'] = df_dist['risk_score'].fillna(0).clip(lower=0)

# 4. Aggregate current district cases per State-Year
current_totals = df_dist.groupby(['iso_year', 'state'])['cases'].sum().reset_index()
current_totals.columns = ['iso_year', 'state', 'current_total']

# Merge to find the Gap
merged_totals = pd.merge(current_totals, official_totals, on=['iso_year', 'state'], how='left')
merged_totals['official_total'] = merged_totals['official_total'].fillna(0)
merged_totals['gap'] = merged_totals['official_total'] - merged_totals['current_total']
merged_totals['gap'] = merged_totals['gap'].clip(lower=0)

# 5. Distribute the Gap based on Risk Score
print("Distributing missing cases based on Risk Scores...")
sum_risk_scores = df_dist.groupby(['iso_year', 'state'])['risk_score'].sum().reset_index()
sum_risk_scores.columns = ['iso_year', 'state', 'sum_risk_score']

df_enhanced = pd.merge(df_dist, merged_totals[['iso_year', 'state', 'gap']], on=['iso_year', 'state'], how='left')
df_enhanced = pd.merge(df_enhanced, sum_risk_scores, on=['iso_year', 'state'], how='left')

df_enhanced['distributed_cases'] = np.where(
    df_enhanced['sum_risk_score'] > 0,
    (df_enhanced['risk_score'] / df_enhanced['sum_risk_score']) * df_enhanced['gap'],
    0
)

df_enhanced['cases'] = np.round(df_enhanced['cases'] + df_enhanced['distributed_cases'])

# 6. Recalculate derived columns (cases_per_100k, is_outbreak)
print("Updating derived metrics (cases_per_100k, is_outbreak)...")
df_enhanced['cases_per_100k'] = (df_enhanced['cases'] / df_enhanced['population_2024']) * 100000
df_enhanced['is_outbreak'] = np.where((df_enhanced['cases'] > 10) | (df_enhanced['cases_per_100k'] > 5), 1.0, 0.0)

# --- RECALCULATE TEMPORAL LAG FEATURES ---
print("Recalculating temporal lag features (cases_lag1, cases_lag2, etc.)...")
df_enhanced = df_enhanced.sort_values(by=['district', 'iso_year', 'iso_week']).reset_index(drop=True)

df_enhanced['cases_lag1'] = df_enhanced.groupby('district')['cases'].shift(1).fillna(0)
df_enhanced['cases_lag2'] = df_enhanced.groupby('district')['cases'].shift(2).fillna(0)
df_enhanced['cases_lag3'] = df_enhanced.groupby('district')['cases'].shift(3).fillna(0)
df_enhanced['cases_roll4w'] = df_enhanced.groupby('district')['cases'].transform(lambda x: x.rolling(window=4, min_periods=1).mean()).fillna(0)

# Clean up temporary columns
df_enhanced = df_enhanced.drop(columns=['risk_score', 'gap', 'sum_risk_score', 'distributed_cases'])

# 7. Save the Enhanced Dataset
print(f"Saving enhanced dataset to {output_file}...")
df_enhanced.to_csv(output_file, index=False)

print("Done!")
