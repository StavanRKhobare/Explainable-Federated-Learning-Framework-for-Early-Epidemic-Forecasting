import pandas as pd
import numpy as np
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "final_datasets/master_dataset_clean.csv"
OUTPUT_FILE = "final_datasets/master_dataset_real_dense.csv"
LOOKBACK = 4
NUM_TRUE_NEGATIVES = 10000

print("="*50)
print("Option A: Real Weather Data Generation (API)")
print("="*50)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
# Increased backoff factor to handle rate limits automatically
retry_session = retry(cache_session, retries=5, backoff_factor=5.0)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 1. Load Original Data
print("Loading original dataset...")
df = pd.read_csv(INPUT_FILE)
original_count = len(df)
print(f"Original events: {original_count}")

# 2. Extract District Info
print("Extracting unique districts and coordinates...")
district_info = df.groupby('censuscode').agg({
    'lat': 'first',
    'lon': 'first',
    'population_2024': 'first',
    'pop_density_per_km2_2024': 'first',
    'district': 'first',
    'state': 'first',
    'LAI': 'mean'  # LAI is hard to fetch free, so we use historical district average
}).reset_index()

# 3. Create a set of existing (district, year, week)
existing_records = set(zip(df['censuscode'], df['iso_year'], df['iso_week']))

def get_prev_week(year, week, lag):
    new_week = week - lag
    new_year = year
    while new_week <= 0:
        new_year -= 1
        new_week += 52
    return new_year, new_week

# 4. Generate Dense Targets (Lookbacks + True Negatives)
print(f"Generating {LOOKBACK} weeks of lookback + {NUM_TRUE_NEGATIVES} True Negatives...")
lookback_records = []
for _, row in df.iterrows():
    c_code = row['censuscode']
    y = row['iso_year']
    w = row['iso_week']
    for lag in range(1, LOOKBACK + 1):
        prev_y, prev_w = get_prev_week(y, w, lag)
        if (c_code, prev_y, prev_w) not in existing_records:
            lookback_records.append({'censuscode': c_code, 'iso_year': prev_y, 'iso_week': prev_w, 'cases': 0.0, 'is_outbreak': 0.0})
            existing_records.add((c_code, prev_y, prev_w))

unique_codes = df['censuscode'].unique()
min_year, max_year = df['iso_year'].min(), df['iso_year'].max()

tn_records = []
while len(tn_records) < NUM_TRUE_NEGATIVES:
    c_code = np.random.choice(unique_codes)
    y = np.random.randint(min_year, max_year + 1)
    w = np.random.randint(1, 53)
    if (c_code, y, w) not in existing_records:
        tn_records.append({'censuscode': c_code, 'iso_year': y, 'iso_week': w, 'cases': 0.0, 'is_outbreak': 0.0})
        existing_records.add((c_code, y, w))

new_df = pd.DataFrame(lookback_records + tn_records)

# 5. Add Static Features to New Data
new_df = pd.merge(new_df, district_info, on='censuscode', how='left')
new_df['week_sin'] = np.sin(2 * np.pi * new_df['iso_week'] / 52.0)
new_df['week_cos'] = np.cos(2 * np.pi * new_df['iso_week'] / 52.0)
new_df['is_monsoon'] = new_df['iso_week'].apply(lambda w: 1 if 24 <= w <= 36 else 0)

# Fill other tracking vars
for col in ['deaths', 'cases_lag1', 'cases_lag2', 'cases_lag3', 'cases_roll4w', 'growth_rate', 'cases_per_100k']:
    new_df[col] = 0.0

# 6. Fetch Historical Weather per District
print("Fetching real historical weather data from Open-Meteo API...")
print("This script now supports RESUMING. If interrupted, run it again to continue.")

TEMP_DIR = "final_datasets/temp_weather"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

START_DATE = f"{min_year}-01-01"
END_DATE = f"{max_year}-12-31"

total = len(district_info)
for idx, row in district_info.iterrows():
    c_code = row['censuscode']
    lat = row['lat']
    lon = row['lon']
    
    # Check if we already have this district's data
    temp_path = os.path.join(TEMP_DIR, f"{c_code}.csv")
    if os.path.exists(temp_path):
        continue
        
    print(f"  Fetching district {idx+1}/{total} (Code: {c_code})...")
        
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "timezone": "auto"
    }
    
    success = False
    retries = 0
    while not success and retries < 3:
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            
            # We need to construct a dataframe from the daily data
            date_range = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            )
            
            temp_data = daily.Variables(0).ValuesAsNumpy()
            precip_data = daily.Variables(1).ValuesAsNumpy()
            
            weather_df = pd.DataFrame({
                "date": date_range,
                "temp_c": temp_data,
                "preci_mm": precip_data
            })
            
            # Convert date to iso_year and iso_week
            weather_df['iso_year'] = weather_df['date'].dt.isocalendar().year
            weather_df['iso_week'] = weather_df['date'].dt.isocalendar().week
            
            # Aggregate daily to weekly
            weekly_weather = weather_df.groupby(['iso_year', 'iso_week']).agg({
                'temp_c': 'mean',
                'preci_mm': 'sum'
            }).reset_index()
            
            # Convert Celsius to Kelvin
            weekly_weather['temp_k'] = weekly_weather['temp_c'] + 273.15
            weekly_weather = weekly_weather.drop(columns=['temp_c'])
            weekly_weather['censuscode'] = c_code
            
            # SAVE to temp file immediately
            weekly_weather.to_csv(temp_path, index=False)
            success = True
            
        except Exception as e:
            err_str = str(e)
            if 'Minutely API request limit exceeded' in err_str or '429' in err_str:
                print(f"  [Rate Limit Hit] Sleeping for 65 seconds before retrying...")
                time.sleep(65)
                retries += 1
            elif 'Hourly API request limit exceeded' in err_str:
                print(f"  [CRITICAL: Hourly Limit Hit] Stopping for now. Run this script again in 1 hour to resume.")
                exit(0)
            else:
                print(f"  Warning: Failed to fetch for district {c_code}: {e}")
                break
    
    # Sleep to pace the requests
    time.sleep(2.5)

# 7. Merge all fetched data
print("Merging all downloaded weather files...")
all_weather_dfs = []
for filename in os.listdir(TEMP_DIR):
    if filename.endswith(".csv"):
        all_weather_dfs.append(pd.read_csv(os.path.join(TEMP_DIR, filename)))

if all_weather_dfs:
    final_weather = pd.concat(all_weather_dfs, ignore_index=True)
    
    # Merge the API weather into our new_df
    print("Combining weather with the dense index...")
    new_df = pd.merge(new_df, final_weather, on=['censuscode', 'iso_year', 'iso_week'], how='left')
    
    # Fill any API failures with global means just in case
    new_df['temp_k'] = new_df['temp_k'].fillna(new_df['temp_k'].mean())
    new_df['preci_mm'] = new_df['preci_mm'].fillna(0.0)

else:
    print("FATAL ERROR: No weather data found in temp_weather/ folder.")
    exit(1)

# 8. Combine and Save
final_df = pd.concat([df, new_df], ignore_index=True)
final_df = final_df.sort_values(["censuscode", "iso_year", "iso_week"]).reset_index(drop=True)

print(f"Done! Final dataset has {len(final_df)} rows (Original: {original_count}, Added: {len(new_df)}).")
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to {OUTPUT_FILE}")
print(f"You can now delete the '{TEMP_DIR}' folder if you wish.")
