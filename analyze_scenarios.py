import json
import pandas as pd

# Load predictions
with open("predictions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load graph edges to know neighbors
edges_df = pd.read_csv("data/graph/graph_edges.csv")
neighbors_map = {}
for row in edges_df.itertuples():
    s, d = row.source_censuscode, row.target_censuscode
    if s not in neighbors_map: neighbors_map[s] = []
    if d not in neighbors_map: neighbors_map[d] = []
    neighbors_map[s].append(d)
    neighbors_map[d].append(s)

# Load dataset to easily access cases
df = pd.read_csv("data/training_dataset_with_ner.csv")
# Create lookup: (year, week, censuscode) -> cases, cases_lag1, etc
lookup = {}
for row in df.itertuples():
    lookup[(row.iso_year, row.iso_week, row.censuscode)] = {
        "cases": row.cases,
        "cases_lag1": row.cases_lag1,
        "cases_lag2": row.cases_lag2,
        "cases_lag3": row.cases_lag3,
        "is_outbreak": row.is_outbreak
    }

# Build flat list of all predictions
flat_preds = []
for t in data["timeline"]:
    y, w = t["year"], t["week"]
    for p in t["predictions"]:
        c = p["censuscode"]
        info = lookup.get((y, w, c), {})
        flat_preds.append({
            "year": y, "week": w, "censuscode": c,
            "prob": p["prob"], "true": p["true"],
            "cases": info.get("cases", 0),
            "cases_lag1": info.get("cases_lag1", 0),
            "cases_lag2": info.get("cases_lag2", 0),
            "cases_lag3": info.get("cases_lag3", 0),
            "is_outbreak": info.get("is_outbreak", 0)
        })

pdf = pd.DataFrame(flat_preds)

print("=== REAL WORLD SCENARIO ANALYSIS ===")

# Scenario 4: District with just 2, 3, or 4 cases
s4 = pdf[(pdf["cases"].between(2, 4)) & (pdf["cases_lag1"] < 5)]
if not s4.empty:
    sample = s4.sample(1, random_state=42).iloc[0]
    print(f"\nScenario: District with just {sample.cases} cases (Low Risk)")
    print(f"  Cases history (lag3->lag2->lag1->now): {sample.cases_lag3} -> {sample.cases_lag2} -> {sample.cases_lag1} -> {sample.cases}")
    print(f"  Predicted Outbreak Probability: {sample.prob*100:.1f}%")

# Scenario 3: District with reducing cases
s3 = pdf[(pdf["cases_lag3"] > 20) & (pdf["cases_lag2"] > 10) & (pdf["cases_lag1"] < 10) & (pdf["cases"] < 5)]
if not s3.empty:
    sample = s3.sample(1, random_state=42).iloc[0]
    print(f"\nScenario: Reducing Lag Cases")
    print(f"  Cases history (lag3->lag2->lag1->now): {sample.cases_lag3} -> {sample.cases_lag2} -> {sample.cases_lag1} -> {sample.cases}")
    print(f"  Predicted Outbreak Probability: {sample.prob*100:.1f}%")

# Scenario 1 & 2: Neighbors
# Find pairs of neighbors at the same time
# To make it fast, we can just iterate over a subset of rows where cases are high
high_cases = pdf[pdf["cases"] > 25]

found_s1 = False
found_s2 = False

for _, row in high_cases.iterrows():
    y, w, c = row["year"], row["week"], row["censuscode"]
    neighs = neighbors_map.get(c, [])
    
    # Check neighbors at the same time
    for n in neighs:
        n_data = pdf[(pdf["year"] == y) & (pdf["week"] == w) & (pdf["censuscode"] == n)]
        if n_data.empty: continue
        n_row = n_data.iloc[0]
        
        # Scenario 1: One has high cases, neighbor has 0
        if not found_s1 and n_row["cases"] == 0 and row["cases"] > 25:
            print(f"\nScenario: Node A has {row['cases']} cases, Neighbor B has 0 cases")
            print(f"  Node A (Cases: {row['cases']}) Predicted Prob: {row['prob']*100:.1f}%")
            print(f"  Node B (Cases: {n_row['cases']}) Predicted Prob: {n_row['prob']*100:.1f}%")
            found_s1 = True
            
        # Scenario 2: Both have high cases
        if not found_s2 and n_row["cases"] > 25 and row["cases"] > 25:
            print(f"\nScenario: Both Node A and Neighbor B have High Cases (Outbreak)")
            print(f"  Node A (Cases: {row['cases']}) Predicted Prob: {row['prob']*100:.1f}%")
            print(f"  Node B (Cases: {n_row['cases']}) Predicted Prob: {n_row['prob']*100:.1f}%")
            found_s2 = True
            
        if found_s1 and found_s2:
            break
    if found_s1 and found_s2:
        break
