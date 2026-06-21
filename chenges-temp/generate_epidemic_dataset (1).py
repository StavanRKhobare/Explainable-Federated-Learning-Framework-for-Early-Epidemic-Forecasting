import pandas as pd
import numpy as np
import os
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
INPUT_FILE = "data/raw/master_dataset_clean.csv"
EDGES_FILE = "data/graph/graph_edges.csv"
OUTPUT_FILE = "data/training_dataset_enhanced_v3.csv"

print("="*60)
print("Epidemic Data Generator (Ross-Macdonald + GNN Spatio-Temporal)")
print("="*60)

# 1. Load Original Data & Demographics
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing master clean dataset: {INPUT_FILE}")
if not os.path.exists(EDGES_FILE):
    raise FileNotFoundError(f"Missing graph edges: {EDGES_FILE}")

print("Loading original clean dataset...")
df_orig = pd.read_csv(INPUT_FILE)
original_count = len(df_orig)
print(f"Loaded {original_count} original records.")

# Extract unique districts (284 districts)
print("Extracting unique districts and metadata...")
district_info = df_orig.groupby('censuscode').agg({
    'district': 'first',
    'state': 'first',
    'lat': 'first',
    'lon': 'first',
    'population_2024': 'first',
    'pop_density_per_km2_2024': 'first'
}).reset_index()

unique_censuscodes = sorted(district_info['censuscode'].unique())
num_districts = len(unique_censuscodes)
print(f"Found {num_districts} unique districts.")

# 2. Build Continuous Timeline Index (2009 - 2022)
# We will create exactly 52 weeks per year for 14 years: 2009 to 2022 inclusive.
years = list(range(2009, 2023))
weeks = list(range(1, 53))

timeline = []
for y in years:
    for w in weeks:
        timeline.append((y, w))
num_weeks = len(timeline)
print(f"Generated timeline of {num_weeks} weeks (14 years, 2009-2022).")

# 3. Calculate District-Week Seasonal Weather Profiles
print("Computing district-specific seasonal weather profiles...")
# We use both master_dataset_clean and any other source to build average profiles.
profile_data = df_orig.groupby(['censuscode', 'iso_week']).agg({
    'temp_k': 'mean',
    'preci_mm': 'mean',
    'LAI': 'mean'
}).reset_index()

# Build fallback profiles at district-level (in case some weeks of the year are missing)
fallback_district = df_orig.groupby('censuscode').agg({
    'temp_k': 'mean',
    'preci_mm': 'mean',
    'LAI': 'mean'
}).reset_index()

# Global fallback just in case
global_temp = df_orig['temp_k'].mean()
global_preci = df_orig['preci_mm'].mean()
global_lai = df_orig['LAI'].mean()

# Create lookup dictionaries for fast access
weather_lookup = {}
for row in profile_data.itertuples():
    weather_lookup[(row.censuscode, row.iso_week)] = (row.temp_k, row.preci_mm, row.LAI)

fallback_lookup = {}
for row in fallback_district.itertuples():
    fallback_lookup[row.censuscode] = (row.temp_k, row.preci_mm, row.LAI)

# 4. Load the Shared Border Graph (Adjacency List)
print("Loading border graph...")
df_edges = pd.read_csv(EDGES_FILE)
# Filter edges to only keep contacts between our 284 districts
census_set = set(unique_censuscodes)
df_edges_filtered = df_edges[
    df_edges['source_censuscode'].isin(census_set) & 
    df_edges['target_censuscode'].isin(census_set)
].copy()

# Build adjacency list: node -> list of (neighbor_node, weight)
# We will normalize weights so that for each district, the weights of its neighbors sum to 1.
adj_list = {code: [] for code in unique_censuscodes}
for row in df_edges_filtered.itertuples():
    s = int(row.source_censuscode)
    d = int(row.target_censuscode)
    wt = float(row.shared_border_km) if pd.notna(row.shared_border_km) else 1.0
    adj_list[s].append((d, wt))
    adj_list[d].append((s, wt)) # Undirected graph

# Normalize weights
for code in unique_censuscodes:
    neighbors = adj_list[code]
    if neighbors:
        total_wt = sum(n[1] for n in neighbors)
        adj_list[code] = [(n[0], n[1] / total_wt) for n in neighbors]

# 5. Pre-generate continuous grid and meteorological data
print("Generating continuous meteorological baseline...")
grid_records = []
for code in unique_censuscodes:
    info = district_info[district_info['censuscode'] == code].iloc[0]
    pop = info['population_2024']
    density = info['pop_density_per_km2_2024']
    dist_name = info['district']
    state_name = info['state']
    lat = info['lat']
    lon = info['lon']
    
    # Retrieve district fallbacks
    f_temp, f_preci, f_lai = fallback_lookup.get(code, (global_temp, global_preci, global_lai))
    
    for y, w in timeline:
        # Retrieve seasonal profiles
        t_val, p_val, l_val = weather_lookup.get((code, w), (f_temp, f_preci, f_lai))
        
        # Fill missing values if NaN
        if pd.isna(t_val): t_val = f_temp
        if pd.isna(p_val): p_val = f_preci
        if pd.isna(l_val): l_val = f_lai
        
        # Add realistic noise/variability (inter-annual and weekly fluctuations)
        # Temperature (Kelvin): profile value + random offset
        t_val_noisy = t_val + np.random.normal(0, 0.7)
        # Precipitation (mm): profile value * scale multiplier
        p_val_noisy = max(0.0, p_val * (1.0 + np.random.normal(0, 0.2)))
        # LAI: profile value + small noise
        l_val_noisy = max(0.1, l_val + np.random.normal(0, 0.05))
        
        # Time encodings
        week_sin = np.sin(2 * np.pi * w / 52.0)
        week_cos = np.cos(2 * np.pi * w / 52.0)
        is_monsoon = 1 if 24 <= w <= 42 else 0
        
        grid_records.append({
            'censuscode': code,
            'district': dist_name,
            'state': state_name,
            'lat': lat,
            'lon': lon,
            'iso_year': y,
            'iso_week': w,
            'temp_k': t_val_noisy,
            'preci_mm': p_val_noisy,
            'LAI': l_val_noisy,
            'population_2024': pop,
            'pop_density_per_km2_2024': density,
            'week_sin': week_sin,
            'week_cos': week_cos,
            'is_monsoon': is_monsoon,
            'cases': 0.0,
            'deaths': 0.0,
            'is_outbreak': 0.0
        })

df_grid = pd.DataFrame(grid_records)
# Sort by censuscode and date
df_grid = df_grid.sort_values(['censuscode', 'iso_year', 'iso_week']).reset_index(drop=True)
print(f"Baseline grid generated with shape: {df_grid.shape}")

# 6. Execute Epidemic Simulation (Ross-Macdonald Equation + Spatio-Temporal Propagation)
print("Running Ross-Macdonald epidemic simulation...")

# Helper structures
dist_to_idx = {code: i for i, code in enumerate(unique_censuscodes)}
idx_to_dist  = {i: code for i, code in enumerate(unique_censuscodes)}

cases_mat = np.zeros((num_weeks, num_districts))
temp_mat  = np.zeros((num_weeks, num_districts))
preci_mat = np.zeros((num_weeks, num_districts))
lai_mat   = np.zeros((num_weeks, num_districts))
vc_mat    = np.zeros((num_weeks, num_districts))

# Build a fast O(1) timeline lookup
timeline_to_idx = {(y, w): i for i, (y, w) in enumerate(timeline)}

# Vectorised fill of meteorological matrices
df_grid['_t_idx'] = df_grid.apply(lambda r: timeline_to_idx[(r.iso_year, r.iso_week)], axis=1)
df_grid['_d_idx'] = df_grid['censuscode'].map(dist_to_idx)
temp_mat [df_grid['_t_idx'].values, df_grid['_d_idx'].values] = df_grid['temp_k'].values
preci_mat[df_grid['_t_idx'].values, df_grid['_d_idx'].values] = df_grid['preci_mm'].values
lai_mat  [df_grid['_t_idx'].values, df_grid['_d_idx'].values] = df_grid['LAI'].values

# ── Calibrated Simulation Constants ────────────────────────────────────────────
# CM=100 yields ~5% outbreak prevalence, consistent with dengue epidemiology.
CM    = 100.0   # Vector abundance scaling factor (calibrated)
DELTA = 0.60    # Autoregressive case retention between weeks
GAMMA = 0.04    # Local transmission efficiency multiplier
ETA   = 0.10    # Spatial border diffusion coefficient

# Spark parameters: probability and magnitude of stochastic importation events
SPARK_PROB_MONSOON = 0.08   # Probability during monsoon weeks (24-42)
SPARK_PROB_DRY     = 0.005  # Probability during dry season

# Population-proportional case ceiling: realistic max ~1% of population
pop_arr   = district_info.set_index('censuscode').loc[unique_censuscodes, 'population_2024'].values
CASE_CEIL = np.maximum(200, (pop_arr * 0.01).astype(int))   # minimum 200 cases ceiling

# ── Run iteration week by week ──────────────────────────────────────────────────
for t in range(num_weeks):
    _y, _w = timeline[t]
    is_monsoon_t = (24 <= _w <= 42)

    for d in range(num_districts):
        code = idx_to_dist[d]

        # A. Ross-Macdonald Vectorial Capacity ────────────────────────────────
        temp_c = temp_mat[t, d] - 273.15  # Kelvin → Celsius

        # 1. Biting rate (a) — Brière thermal performance curve
        a = 0.0
        if 13.3 < temp_c < 38.5:
            a = 0.00014 * temp_c * (temp_c - 13.3) * np.sqrt(38.5 - temp_c)

        # 2. Daily mortality rate (μ) and daily survival probability (p)
        mu = max(0.05, 0.00224 * temp_c**2 - 0.113 * temp_c + 1.545)
        p  = np.exp(-mu)

        # 3. Extrinsic Incubation Period (n days)
        n = 21.0
        if 12.2 < temp_c < 38.5:
            Y = 0.000035 * temp_c * (temp_c - 12.2) * np.sqrt(38.5 - temp_c)
            if Y > 0:
                n = max(5.0, min(21.0, 1.0 / Y))

        # 4. Larval pool: weighted sum of past 4 weeks rainfall (mosquito dev. cycle)
        lag_weights = [0.1, 0.4, 0.4, 0.1]
        l_pool = sum(
            lag_weights[lag - 1] * preci_mat[t - lag, d]
            if t - lag >= 0
            else 0.25 * preci_mat[t, d]
            for lag in range(1, 5)
        )

        # 5. Mosquito-to-human ratio (m): driven by larval pool and vegetation
        density = district_info.iloc[d]['pop_density_per_km2_2024']
        m = CM * (l_pool * lai_mat[t, d]) / np.log(density + 2.0)

        # 6. Vectorial Capacity
        vc = (m * (a ** 2) * (p ** n)) / mu if mu > 0 else 0.0
        vc_mat[t, d] = vc

        # B. Case Transmission Model ──────────────────────────────────────────
        if t == 0:
            # Seed sparse initial cases
            cases_mat[t, d] = float(
                random.choices([0, 1, 2, 5], weights=[0.80, 0.15, 0.04, 0.01])[0]
            )
        else:
            # Spatial pressure from border-sharing neighbours
            spatial_pres = sum(
                nb_wt * cases_mat[t - 1, dist_to_idx[nb_code]]
                for nb_code, nb_wt in adj_list[code]
            )

            retained     = DELTA * cases_mat[t - 1, d]
            local_growth = GAMMA * vc * cases_mat[t - 1, d]
            imported     = ETA   * spatial_pres

            # Stochastic spark: VC-proportional importation event
            spark = 0.0
            spark_prob = SPARK_PROB_MONSOON if is_monsoon_t else SPARK_PROB_DRY
            if vc > 1.0 and random.random() < spark_prob:
                # Spark magnitude scales with VC (hotter climate = larger seed)
                spark = float(random.randint(1, max(1, int(vc * 2))))

            expected_cases = retained + local_growth + imported + spark

            # Stochastic realisation (Negative Binomial approximation via Poisson)
            # Clamp to per-district ceiling to avoid explosive runaway
            actual_cases = min(
                int(CASE_CEIL[d]),
                int(np.random.poisson(max(0.0, expected_cases)))
            )
            cases_mat[t, d] = float(actual_cases)

# 7. Merge cases back into the grid dataframe
print("Merging simulation outputs back to dataframe...")
cases_list = []
vc_list = []
for row in df_grid.itertuples():
    t_idx = timeline.index((row.iso_year, row.iso_week))
    d_idx = dist_to_idx[row.censuscode]
    cases_list.append(cases_mat[t_idx, d_idx])
    vc_list.append(vc_mat[t_idx, d_idx])

df_grid['cases'] = cases_list
df_grid['vectorial_capacity'] = vc_list # Store vectorial capacity for reference

# Fill deaths: simple fatality rate (e.g. ~0.1% to 0.5% of cases during outbreaks)
df_grid['deaths'] = df_grid['cases'].apply(lambda c: round(c * np.random.uniform(0.001, 0.005)) if c > 5 else 0.0)

# Calculate cases_per_100k
df_grid['cases_per_100k'] = (df_grid['cases'] / df_grid['population_2024']) * 100000

# Define is_outbreak using the project's standard criteria
df_grid['is_outbreak'] = np.where((df_grid['cases'] > 10) | (df_grid['cases_per_100k'] > 5.0), 1.0, 0.0)

# 8. Calculate Mathematically Correct Lag Features
print("Calculating mathematically correct lag features (cases_lag1, cases_lag2, cases_lag3, cases_roll4w)...")
df_grid = df_grid.sort_values(by=['censuscode', 'iso_year', 'iso_week']).reset_index(drop=True)

df_grid['cases_lag1'] = df_grid.groupby('censuscode')['cases'].shift(1).fillna(0.0)
df_grid['cases_lag2'] = df_grid.groupby('censuscode')['cases'].shift(2).fillna(0.0)
df_grid['cases_lag3'] = df_grid.groupby('censuscode')['cases'].shift(3).fillna(0.0)
# Rolling 4 weeks mean (cases in t-1, t-2, t-3, t-4)
df_grid['cases_roll4w'] = df_grid.groupby('censuscode')['cases'].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
).fillna(0.0)

# Calculate growth rate: (cases - cases_lag1) / (cases_lag1 + 1e-5)
df_grid['growth_rate'] = (df_grid['cases'] - df_grid['cases_lag1']) / (df_grid['cases_lag1'] + 1e-5)
df_grid['growth_rate'] = df_grid['growth_rate'].clip(lower=-10.0, upper=10.0).fillna(0.0)

# Ensure columns align with standard schema
df_grid = df_grid[[
    'iso_year', 'iso_week', 'censuscode', 'district', 'state', 'lat', 'lon',
    'cases', 'deaths', 'is_outbreak', 'cases_lag1', 'cases_lag2', 'cases_lag3',
    'cases_roll4w', 'temp_k', 'preci_mm', 'LAI', 'population_2024', 'pop_density_per_km2_2024',
    'week_sin', 'week_cos', 'is_monsoon', 'growth_rate', 'cases_per_100k'
]]

# Save final dataset
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_grid.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Pipeline Complete! New continuous dataset saved to: {OUTPUT_FILE}")
print(f"  Total rows: {len(df_grid):,} (284 districts * 728 weeks)")
print(f"  Outbreak distribution: \n{df_grid['is_outbreak'].value_counts()}")
print(f"  Average cases per district-week: {df_grid['cases'].mean():.3f}")
print(f"  Max cases in a single week: {df_grid['cases'].max()}")
print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.2f} MB")
print("="*60)
