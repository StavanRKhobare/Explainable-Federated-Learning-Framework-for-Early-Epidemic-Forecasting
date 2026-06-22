"""
Generate realistic EHR samples (CSVs + TXT clinical notes) for 3 cities:
 - Bangalore General Hospital  (census 572)
 - Coimbatore Medical College  (census 632)
 - New Delhi Hospital          (census 94)

Run from repo root:
    python scripts/generate_ehr_samples.py
"""

import os, csv, random
from pathlib import Path
from datetime import date, timedelta

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Output base ───────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "ehr_samples"

# ── City profiles ─────────────────────────────────────────────────────────────
CITIES = {
    "bangalore": {
        "hospital": "Bangalore General Hospital",
        "censuscode": 572,
        "address": "K.R. Market Road, Bengaluru – 560002",
        "district": "Bengaluru Urban",
        "state": "Karnataka",
        "endemic": ["dengue", "typhoid", "leptospirosis"],
        "dengue_prevalence": 0.28,  # fraction of positive cases
        "names_m": ["Arjun", "Rohan", "Vikram", "Kiran", "Suresh", "Mahesh", "Ravi", "Deepak"],
        "names_f": ["Priya", "Sneha", "Kavitha", "Anitha", "Rekha", "Deepa", "Lakshmi"],
        "surnames": ["Kumar", "Reddy", "Naidu", "Shetty", "Gowda", "Murthy", "Patel"],
    },
    "coimbatore": {
        "hospital": "Coimbatore Medical College Hospital",
        "censuscode": 632,
        "address": "Trichy Road, Coimbatore – 641018",
        "district": "Coimbatore",
        "state": "Tamil Nadu",
        "endemic": ["dengue", "chikungunya", "malaria"],
        "dengue_prevalence": 0.32,
        "names_m": ["Senthil", "Murugan", "Arun", "Prasath", "Vijay", "Balaji", "Saravanan"],
        "names_f": ["Lakshmi", "Selvi", "Meena", "Suganya", "Kaveri", "Padma", "Geetha"],
        "surnames": ["Muthusamy", "Krishnan", "Subramanian", "Rajan", "Pandian", "Natarajan"],
    },
    "delhi": {
        "hospital": "New Delhi District Hospital",
        "censuscode": 94,
        "address": "Deen Dayal Upadhyay Marg, New Delhi – 110002",
        "district": "New Delhi",
        "state": "Delhi",
        "endemic": ["dengue", "typhoid", "influenza"],
        "dengue_prevalence": 0.21,
        "names_m": ["Rahul", "Amit", "Vivek", "Sanjay", "Rajesh", "Pankaj", "Nikhil"],
        "names_f": ["Pooja", "Neha", "Sunita", "Ananya", "Ritu", "Simran", "Divya"],
        "surnames": ["Sharma", "Gupta", "Verma", "Singh", "Mishra", "Joshi", "Agarwal"],
    },
}

# ── EHR template helpers ──────────────────────────────────────────────────────
DENGUE_POS_SYMPTOMS = [
    'severe joint pain ("breakbone fever"), retroorbital headaches, and maculopapular rash on trunk.',
    'high-grade fever, myalgia, vomiting, and diffuse skin rash since 3 days.',
    'sudden onset fever with chills, severe backache, and retro-orbital pain.',
    'acute febrile illness with petechiae, muscle ache, and fatigue.',
    'dengue-like illness: frontal headache, nausea, joint pains, and positive tourniquet test.',
]
DENGUE_NEG_SYMPTOMS = [
    'mild cough, mild rhinorrhoea, and low-grade fever. No rash or joint pain.',
    'productive cough, sore throat, and mild body ache. Likely viral URTI.',
    'loose stools x3, nausea, and mild abdominal pain. No fever.',
    'fever with headache, no rash. Blood film negative for malaria. Likely viral fever.',
    'bilateral nasal congestion, post-nasal drip, low-grade fever 99.2°F.',
]
NORMAL_DIAGNOSIS = [
    "Acute Viral Respiratory Infection (AVRI)",
    "Viral Gastroenteritis",
    "Allergic Rhinitis with secondary bacterial sinusitis",
    "Undifferentiated Viral Fever",
    "Acute Pharyngitis",
]
DENGUE_TREATMENT = (
    "Prescribed paracetamol 500 mg TDS for fever. Strict bed rest and aggressive oral/IV hydration. "
    "Avoid NSAIDs (aspirin, ibuprofen). Platelet monitoring every 24 hrs. "
    "Patient counselled on warning signs: bleeding, severe abdominal pain, rapid breathing."
)
NORMAL_TREATMENT = (
    "Symptomatic management: antipyretics, ORS for hydration. Advised rest, avoid cold foods. "
    "Follow up if fever persists > 3 days. No antibiotic prescribed."
)


def rand_date(year=2024):
    """Return a random date string within year."""
    start = date(year, 1, 1)
    offset = random.randint(0, 364)
    return (start + timedelta(days=offset)).isoformat()


def rand_name(city_cfg):
    sex = random.choice(["Male", "Female"])
    pool = city_cfg["names_m"] if sex == "Male" else city_cfg["names_f"]
    return f"{random.choice(pool)} {random.choice(city_cfg['surnames'])}", sex


def rand_vitals(is_dengue):
    if is_dengue:
        temp_c = round(random.uniform(38.8, 40.4), 1)
        temp_f = round(temp_c * 9/5 + 32, 1)
        bp_sys = random.randint(90, 120)
        pulse = random.randint(90, 130)
        spo2 = random.randint(94, 99)
    else:
        temp_c = round(random.uniform(36.5, 38.3), 1)
        temp_f = round(temp_c * 9/5 + 32, 1)
        bp_sys = random.randint(110, 130)
        pulse = random.randint(65, 90)
        spo2 = random.randint(97, 100)
    bp_dia = random.randint(60, max(61, bp_sys - 30))
    return temp_c, temp_f, bp_sys, bp_dia, pulse, spo2


def build_ehr_text(city_cfg, patient_id, is_dengue, platelet_low=False):
    name, sex = rand_name(city_cfg)
    age = random.randint(8, 72)
    visit_date = rand_date(2024)
    temp_c, temp_f, bp_sys, bp_dia, pulse, spo2 = rand_vitals(is_dengue)
    symptoms = random.choice(DENGUE_POS_SYMPTOMS if is_dengue else DENGUE_NEG_SYMPTOMS)
    platelet = random.randint(28000, 80000) if is_dengue and platelet_low else random.randint(120000, 280000) if not is_dengue else random.randint(85000, 145000)
    wbc = round(random.uniform(2.1, 4.8) if is_dengue else random.uniform(4.5, 11.0), 1)
    ns1 = "POSITIVE (detected)" if is_dengue else "NEGATIVE"
    igm = "Positive" if is_dengue else "Negative"
    diagnosis = f"Dengue Fever – {'Severe' if platelet < 50000 else 'Non-Severe'} (WHO 2009 criteria)" if is_dengue else random.choice(NORMAL_DIAGNOSIS)
    treatment = DENGUE_TREATMENT if is_dengue else NORMAL_TREATMENT

    lines = [
        f"{'='*56}",
        f"  {city_cfg['hospital'].upper()}",
        f"  CLINICAL ELECTRONIC HEALTH RECORD",
        f"{'='*56}",
        f"Patient ID    : {city_cfg['censuscode']}-{patient_id:04d}",
        f"Patient Name  : {name}",
        f"Age / Sex     : {age} / {sex}",
        f"Address       : {city_cfg['district']}, {city_cfg['state']}",
        f"Visit Date    : {visit_date}",
        f"Ward          : {'Isolation / Dengue Ward' if is_dengue else 'OPD / General Medicine'}",
        "",
        "PRESENTING COMPLAINT:",
        f"  Patient reports {symptoms}",
        "",
        "VITALS ON ADMISSION:",
        f"  Temperature   : {temp_c} °C  ({temp_f} °F)  {'[HIGH FEVER]' if temp_c > 38.5 else '[Normal]'}",
        f"  Blood Pressure: {bp_sys}/{bp_dia} mmHg",
        f"  Heart Rate    : {pulse} bpm  {'[Tachycardia]' if pulse > 100 else ''}",
        f"  SpO₂          : {spo2}%",
        "",
        "LABORATORY INVESTIGATIONS:",
        f"  Dengue NS1 Antigen    : {ns1}",
        f"  Dengue IgM Antibody   : {igm}",
        f"  Platelet Count        : {platelet:,} /µL  {'[CRITICAL – thrombocytopenia]' if platelet < 50000 else '[Low]' if platelet < 100000 else '[Normal]'}",
        f"  WBC Count             : {wbc} × 10³/µL  {'[Leukopenia]' if wbc < 4.5 else ''}",
        f"  Blood Smear (Malaria) : Negative",
        f"  Hematocrit            : {random.randint(30,48)}%",
        "",
        "DIAGNOSIS:",
        f"  {diagnosis}",
        "",
        "TREATMENT PLAN:",
        f"  {treatment}",
        "",
        f"{'─'*56}",
        f"Reporting Physician : Dr. {random.choice(city_cfg['names_m'])} {random.choice(city_cfg['surnames'])}",
        f"Hospital            : {city_cfg['hospital']}",
        f"Census Code         : {city_cfg['censuscode']}",
        f"{'='*56}",
    ]
    return "\n".join(lines)


def build_csv_rows(n_patients, dengue_prevalence):
    """
    Generates richer CSV rows with multiple clinical features (not just temperature).
    """
    rows = []
    for _ in range(n_patients):
        is_dengue = random.random() < dengue_prevalence
        temp_c, temp_f, bp_sys, bp_dia, pulse, spo2 = rand_vitals(is_dengue)
        platelet = random.randint(28000, 80000) if is_dengue and random.random() < 0.4 else random.randint(85000, 145000) if is_dengue else random.randint(120000, 280000)
        wbc = round(random.uniform(2.1, 4.8) if is_dengue else random.uniform(4.5, 11.0), 1)
        rash = 1 if is_dengue and random.random() < 0.75 else 0
        joint_pain = 1 if is_dengue and random.random() < 0.85 else random.randint(0, 1) * (1 if random.random() < 0.05 else 0)
        headache = 1 if is_dengue and random.random() < 0.90 else random.randint(0, 1) * (1 if random.random() < 0.15 else 0)
        vomiting = 1 if is_dengue and random.random() < 0.55 else random.randint(0, 1) * (1 if random.random() < 0.10 else 0)
        ns1_pos = 1 if is_dengue else 0
        severity = "severe" if is_dengue and platelet < 50000 else "moderate" if is_dengue else "mild"
        rows.append({
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "blood_pressure_sys": bp_sys,
            "blood_pressure_dia": bp_dia,
            "heart_rate_bpm": pulse,
            "spo2_pct": spo2,
            "platelet_count": platelet,
            "wbc_count": wbc,
            "rash": rash,
            "joint_pain": joint_pain,
            "headache": headache,
            "vomiting": vomiting,
            "ns1_positive": ns1_pos,
            "dengue_status": 1 if is_dengue else 0,
            "severity": severity,
        })
    return rows


# ── Main generation ───────────────────────────────────────────────────────────
N_EHR_TXT = 12      # clinical narrative TXT files per city
N_PATIENTS = 120    # rows in main CSV per city
N_WEEKLY = 8        # weeks of aggregated weekly CSV

for city_key, cfg in CITIES.items():
    city_dir = BASE / city_key
    city_dir.mkdir(parents=True, exist_ok=True)

    dengue_prev = cfg["dengue_prevalence"]

    # ── 1. Individual Patient CSVs (rich schema) ─────────────────────────────
    rows = build_csv_rows(N_PATIENTS, dengue_prev)
    csv_path = city_dir / f"{city_key}_{N_PATIENTS}_patients.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {csv_path.name}  ({N_PATIENTS} rows)")

    # ── 2. Weekly Aggregated Epidemiology CSV ────────────────────────────────
    weekly_rows = []
    for w in range(N_WEEKLY):
        wk_rows = build_csv_rows(random.randint(40, 90), dengue_prev)
        n_cases = sum(r["dengue_status"] for r in wk_rows)
        n_total = len(wk_rows)
        avg_temp = round(sum(r["temperature_c"] for r in wk_rows) / n_total, 2)
        avg_plt = round(sum(r["platelet_count"] for r in wk_rows) / n_total)
        n_severe = sum(1 for r in wk_rows if r["severity"] == "severe")
        weekly_rows.append({
            "week": w + 1,
            "year": 2024,
            "total_admissions": n_total,
            "dengue_positive": n_cases,
            "dengue_negative": n_total - n_cases,
            "positivity_rate": round(n_cases / n_total, 4),
            "severe_cases": n_severe,
            "avg_temp_c": avg_temp,
            "avg_platelet": avg_plt,
            "hospital": cfg["hospital"],
            "censuscode": cfg["censuscode"],
        })
    weekly_path = city_dir / f"{city_key}_weekly_epidemiology.csv"
    with open(weekly_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=weekly_rows[0].keys())
        writer.writeheader()
        writer.writerows(weekly_rows)
    print(f"[OK] {weekly_path.name}  ({N_WEEKLY} weeks)")

    # ── 3. EHR Narrative TXT Files ───────────────────────────────────────────
    for i in range(N_EHR_TXT):
        pid = random.randint(1000, 9999)
        is_dengue = random.random() < (dengue_prev + 0.1)   # slightly oversample positives
        platelet_low = is_dengue and random.random() < 0.3
        label = "dengue_pos" if is_dengue else "dengue_neg"
        fname = city_dir / f"patient_{i+1:02d}_{label}_{city_key[:3].upper()}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(build_ehr_text(cfg, pid, is_dengue, platelet_low))
        print(f"[OK] {fname.name}")

    print()

print("Done — ehr_samples/ populated for all 3 cities.")
