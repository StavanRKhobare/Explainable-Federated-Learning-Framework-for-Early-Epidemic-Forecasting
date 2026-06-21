"""
generate_synthetic_notes.py
============================
Generates realistic synthetic clinical notes for dengue surveillance.
Each note is keyed to a (censuscode, iso_year, iso_week) tuple that matches
the FedXGNN project's training_dataset_enhanced_v2.csv schema.

Usage:
    python generate_synthetic_notes.py

Output:
    synthetic_notes.csv  — columns: censuscode, iso_year, iso_week, note
"""

import random
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Avoid unicode print errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Configuration ─────────────────────────────────────────────────────────────
# Path to the new Ross-Macdonald simulated dataset (v3)
REAL_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "training_dataset_enhanced_v3.csv",
)

NUM_WEEKS = 52           # Weeks per year
YEARS = list(range(2009, 2023))   # 14-year full coverage
SEED = 42


random.seed(SEED)

# ── Clinical Data Pools ───────────────────────────────────────────────────────
# Dengue-specific symptom vocabulary (WHO Clinical Guide)
SYMPTOMS = [
    "fever",
    "high fever",
    "severe headache",
    "retro-orbital pain",
    "pain behind the eyes",
    "joint pain",
    "muscle pain",
    "myalgia",
    "arthralgia",
    "skin rash",
    "maculopapular rash",
    "nausea",
    "vomiting",
    "mild bleeding",
    "petechiae",
    "fatigue",
    "loss of appetite",
    "abdominal pain",
]

DISEASES = [
    "dengue fever",
    "dengue",
    "severe dengue",
    "dengue hemorrhagic fever",
    "malaria",
    "chikungunya",
    "viral fever",
    "typhoid",
    "leptospirosis",
]

PATHOGENS = [
    "Dengue virus",
    "DENV-1",
    "DENV-2",
    "DENV-3",
    "DENV-4",
    "Orthoflavivirus denguei",
    "Aedes aegypti exposure",
]

TRAVEL_PHRASES = [
    "No recent travel history.",
    "Patient traveled to a neighboring district last week.",
    "Recent travel to an urban area reported.",
    "Returned from a rural village 5 days ago.",
    "Traveled to a known dengue-endemic zone recently.",
    "Lives near stagnant water bodies.",
    "No travel outside the district.",
    "Visited a relative in a nearby town last weekend.",
]

CLINICAL_CONTEXTS = [
    "Patient presented to the OPD with",
    "A {age}-year-old {gender} reported with",
    "Chief complaint:",
    "History of present illness includes",
    "Patient was admitted with complaints of",
    "Walk-in patient presenting with",
    "Referral case from PHC with",
    "Emergency presentation with acute onset of",
]

LAB_RESULTS = [
    "NS1 antigen test positive.",
    "NS1 antigen test negative.",
    "IgM ELISA pending.",
    "IgM ELISA positive for dengue.",
    "Platelet count: {platelets}/µL (low).",
    "Platelet count: {platelets}/µL (normal range).",
    "CBC shows leukopenia.",
    "Hematocrit elevated at {hct}%.",
    "Liver enzymes mildly elevated.",
    "Dengue PCR pending.",
    "",  # No lab result mentioned
    "",
]

GENDERS = ["male", "female"]
AGE_RANGE = (1, 85)


# ── Helper Functions ──────────────────────────────────────────────────────────
def load_census_codes():
    """Load real census codes from the project's training dataset."""
    if os.path.exists(REAL_DATASET_PATH):
        df = pd.read_csv(REAL_DATASET_PATH, usecols=["censuscode", "district", "state"])
        districts = df.drop_duplicates("censuscode")[["censuscode", "district", "state"]]
        print(f"✅ Loaded {len(districts)} real district census codes from project data.")
        return districts
    else:
        print(f"⚠️  Real dataset not found at {REAL_DATASET_PATH}")
        print("    Generating with placeholder census codes (100–384).")
        codes = list(range(100, 384))
        return pd.DataFrame({
            "censuscode": codes,
            "district": [f"District_{c}" for c in codes],
            "state": ["Unknown"] * len(codes),
        })


def get_note_volume(week, is_outbreak):
    """
    Simulate note volume per district-week with heavy realistic noise.
    """
    is_monsoon = 24 <= week <= 42
    if is_outbreak == 1.0:
        # 30% chance of missed/low reporting even during an outbreak
        if random.random() < 0.30:
            return random.randint(0, 1)
        return random.randint(2, 5)
    else:
        if is_monsoon:
            # 20% chance of a localized panic or flu outbreak resembling dengue
            if random.random() < 0.20:
                return random.randint(3, 5)
            return random.choices([0, 1, 2], weights=[0.50, 0.30, 0.20])[0]
        else:
            return random.choices([0, 1], weights=[0.70, 0.30])[0]


def generate_note(district_name, state, week, is_outbreak):
    """Generate a single realistic clinical note with heavy noise."""
    age = random.randint(*AGE_RANGE)
    gender = random.choice(GENDERS)
    context = random.choice(CLINICAL_CONTEXTS).format(age=age, gender=gender)

    if is_outbreak == 1.0:
        # 35% of the time during a real outbreak, doctors just diagnose "viral fever" without confirming dengue
        if random.random() < 0.35:
            disease = random.choice(["viral fever", "unknown fever"])
            symptoms = random.sample(["high fever", "headache", "fatigue"], k=random.randint(1, 3))
            pathogen_text = ""
        else:
            n_symptoms = random.randint(1, 3)
            symptoms = random.sample(["high fever", "severe headache", "retro-orbital pain", "joint pain", "maculopapular rash", "mild bleeding", "platelet drop"], k=n_symptoms)
            disease = random.choice(["severe dengue", "dengue hemorrhagic fever", "dengue fever", "suspected dengue"])
            pathogen_text = f" Possible exposure to {random.choice(PATHOGENS)}." if random.random() < 0.3 else ""
    else:
        n_symptoms = random.randint(1, 2)
        symptoms = random.sample(["fever", "headache", "fatigue", "loss of appetite", "nausea", "muscle pain", "abdominal pain"], k=n_symptoms)
        
        # 25% chance of a false dengue diagnosis (False Positive) when there is no actual outbreak
        if random.random() < 0.25:
            disease = "dengue fever"
        else:
            disease = random.choice(["viral fever", "typhoid", "malaria", "chikungunya", "leptospirosis"])
        pathogen_text = ""

    # Travel history
    travel = random.choice(TRAVEL_PHRASES)

    # Lab results (50% chance)
    lab = ""
    if random.random() < 0.5:
        lab_template = random.choice(LAB_RESULTS)
        if lab_template:
            platelet_count = random.randint(20000, 250000) if is_outbreak == 0.0 else random.randint(10000, 100000)
            lab = " " + lab_template.format(
                platelets=platelet_count,
                hct=random.randint(38, 55),
            )

    # Location context
    location = f" District: {district_name}, {state}."

    # Assemble the note
    note = (
        f"{context} {', '.join(symptoms)}. "
        f"Suspected {disease}.{pathogen_text} "
        f"{travel}{lab}{location}"
    )

    return note


# ── Main Generation Loop ─────────────────────────────────────────────────────
def main():
    if os.path.exists(REAL_DATASET_PATH):
        print(f"✅ Loading real dataset for smart simulation: {REAL_DATASET_PATH}")
        df_real = pd.read_csv(REAL_DATASET_PATH)
        df_real = df_real[df_real['iso_year'].isin(YEARS)]
        
        data = []
        for row in df_real.itertuples():
            code = row.censuscode
            year = row.iso_year
            week = row.iso_week
            dist = row.district
            state = row.state
            is_outb = row.is_outbreak
            
            n_notes = get_note_volume(week, is_outbreak=is_outb)
            for _ in range(n_notes):
                note = generate_note(dist, state, week, is_outbreak=is_outb)
                data.append({
                    "censuscode": code,
                    "iso_year": year,
                    "iso_week": week,
                    "note": note,
                })
        
        df = pd.DataFrame(data)
    else:
        print(f"⚠️  Real dataset not found at {REAL_DATASET_PATH}")
        print("    Cannot perform smart simulation without ground truth.")
        return

    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_notes.csv")
    df.to_csv(output_path, index=False)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"SMART SYNTHETIC NOTE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total notes generated  : {len(df):,}")
    print(f"  Districts covered      : {df['censuscode'].nunique()}")
    print(f"  Year(s)                : {YEARS}")
    print(f"  Avg notes/district     : {len(df) / df['censuscode'].nunique():.1f}")
    print(f"  Output file            : {output_path}")
    print(f"  File size              : {os.path.getsize(output_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
