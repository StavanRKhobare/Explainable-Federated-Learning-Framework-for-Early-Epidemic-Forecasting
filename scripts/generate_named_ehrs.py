"""
Generate named-patient EHR TXT files for Coimbatore and Delhi,
matching the style of the original bangalore named-patient files.
Also generates a 70-patient simple CSV (temperature + dengue_status)
for each city to match the original bangalore_70_patients.csv format.

Run from repo root:
    python scripts/generate_named_ehrs.py
"""

import csv, random
from pathlib import Path

random.seed(99)

BASE = Path(__file__).resolve().parent.parent / "ehr_samples"

# ── Named patient profiles ────────────────────────────────────────────────────
PATIENTS = {
    "coimbatore": [
        # (filename_tag, name, age, sex, is_dengue, temp_c)
        ("senthil_dengue_pos_39C",  "Senthil Muthusamy",  28, "Male",   True,  39.4),
        ("meena_dengue_neg_37C",    "Meena Krishnan",     42, "Female", False, 37.1),
        ("arunkumar_dengue_pos_40C","Arunkumar Rajan",    19, "Male",   True,  40.2),
        ("suganya_dengue_neg_37C",  "Suganya Natarajan",  55, "Female", False, 37.3),
    ],
    "delhi": [
        ("rahul_dengue_pos_39C",    "Rahul Sharma",       31, "Male",   True,  39.7),
        ("pooja_dengue_neg_37C",    "Pooja Verma",        26, "Female", False, 37.0),
        ("vivek_dengue_pos_40C",    "Vivek Singh",        45, "Male",   True,  40.1),
        ("sunita_dengue_neg_37C",   "Sunita Gupta",       60, "Female", False, 37.4),
    ],
    "mysore": [
        ("kiran_dengue_pos_39C",    "Kiran Gowda",        34, "Male",   True,  39.5),
        ("lakshmi_dengue_neg_37C",  "Lakshmi Narayan",    29, "Female", False, 37.2),
        ("prashanth_dengue_pos_40C","Prashanth Shetty",   41, "Male",   True,  40.0),
        ("kavitha_dengue_neg_37C",  "Kavitha Rao",        52, "Female", False, 37.5),
    ],
}

HOSPITALS = {
    "coimbatore": "COIMBATORE MEDICAL COLLEGE HOSPITAL",
    "delhi":      "NEW DELHI DISTRICT HOSPITAL",
    "mysore":     "MYSORE DISTRICT HOSPITAL",
}

DENGUE_POS_SYMPTOMS = [
    'severe joint pain ("breakbone fever"), retroorbital headaches, and maculopapular rash on trunk.',
    'high-grade fever with chills, myalgia, vomiting, and diffuse skin rash since 3 days.',
    'sudden onset fever, severe backache, retroorbital pain, and positive tourniquet test.',
    'acute febrile illness with petechiae, muscle ache, nausea, and marked fatigue.',
]
DENGUE_NEG_SYMPTOMS = [
    'mild cough, rhinorrhoea, and low-grade fever. No rash or joint pain.',
    'productive cough, sore throat, and mild body ache. Likely viral URTI.',
    'loose stools, nausea, and mild abdominal pain. No rash.',
    'bilateral nasal congestion, post-nasal drip, and low-grade fever.',
]

def build_named_ehr(hospital_name, name, age, sex, is_dengue, temp_c, visit_date):
    temp_f = round(temp_c * 9/5 + 32, 1)
    bp_sys = random.randint(90, 118) if is_dengue else random.randint(112, 130)
    bp_dia = random.randint(60, max(61, bp_sys - 30))
    pulse  = random.randint(92, 128) if is_dengue else random.randint(66, 88)
    spo2   = random.randint(94, 98)  if is_dengue else random.randint(97, 100)
    platelet = random.randint(38000, 120000) if is_dengue else random.randint(130000, 270000)
    wbc    = round(random.uniform(2.2, 4.5) if is_dengue else random.uniform(4.5, 10.5), 1)
    symptoms = random.choice(DENGUE_POS_SYMPTOMS if is_dengue else DENGUE_NEG_SYMPTOMS)
    ns1    = "POSITIVE (detected)" if is_dengue else "NEGATIVE"
    igm    = "Positive" if is_dengue else "Negative"
    diagnosis = (
        f"Dengue Fever – {'Severe' if platelet < 50000 else 'Non-Severe'} (WHO 2009 criteria)"
        if is_dengue else "Undifferentiated Viral Fever / Acute Viral URTI"
    )
    treatment = (
        "Paracetamol 500 mg TDS for fever. Strict bed rest, aggressive oral/IV hydration. "
        "Avoid NSAIDs. Platelet monitoring every 24 hrs. Patient counselled on warning signs."
        if is_dengue else
        "Symptomatic management: antipyretics and ORS. Rest advised. "
        "Follow up if symptoms persist beyond 3 days."
    )

    lines = [
        f"{'='*52}",
        f"  {hospital_name}",
        f"  CLINICAL ELECTRONIC HEALTH RECORD",
        f"{'='*52}",
        f"Patient Name  : {name}",
        f"Age / Sex     : {age} / {sex}",
        f"Date          : {visit_date}",
        f"Ward          : {'Isolation / Dengue Ward' if is_dengue else 'OPD / General Medicine'}",
        "",
        "SYMPTOMS & VITALS:",
        f"  Subject reports {symptoms}",
        "  Recorded Vitals:",
        f"    - Blood Pressure : {bp_sys}/{bp_dia} mmHg",
        f"    - Pulse          : {pulse} bpm{'  [Tachycardia]' if pulse > 100 else ''}",
        f"    - Temperature    : {temp_c} °C  ({temp_f} °F)  {'[High Fever]' if temp_c > 38.5 else '[Normal]'}",
        f"    - SpO₂           : {spo2}%",
        "",
        "DIAGNOSIS & LAB RESULTS:",
        f"  Dengue NS1 Antigen Test : {ns1}",
        f"  IgM Antibody            : {igm}",
        f"  Platelet Count          : {platelet:,} /µL  {'[CRITICAL]' if platelet < 50000 else '[Low]' if platelet < 100000 else '[Normal]'}",
        f"  WBC Count               : {wbc} × 10³/µL  {'[Leukopenia]' if wbc < 4.5 else ''}",
        f"  Blood Smear (Malaria)   : Negative",
        "",
        f"  DIAGNOSIS: {diagnosis}",
        "",
        "RECOMMENDATION:",
        f"  {treatment}",
        "",
        f"{'='*52}",
    ]
    return "\n".join(lines)

# ── Write named patient TXT EHRs ─────────────────────────────────────────────
DATES = ["2024-06-10", "2024-07-22", "2024-08-05", "2024-09-14"]

for city, patients in PATIENTS.items():
    city_dir = BASE / city
    city_dir.mkdir(parents=True, exist_ok=True)
    for i, (tag, name, age, sex, is_dengue, temp_c) in enumerate(patients):
        visit_date = DATES[i % len(DATES)]
        content = build_named_ehr(HOSPITALS[city], name, age, sex, is_dengue, temp_c, visit_date)
        fname = city_dir / f"patient_{tag}.txt"
        fname.write_text(content, encoding="utf-8")
        print(f"[OK] {fname.name}")

    # ── Simple 70-patient CSV (temperature + dengue_status) ──────────────────
    simple_csv_path = city_dir / f"{city}_70_patients.csv"
    with open(simple_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "dengue_status"])
        prev = PATIENTS[city][0][2+2]  # is_dengue of first patient
        for _ in range(70):
            is_d = random.random() < (0.30 if city == "coimbatore" else 0.22)
            temp = round(random.uniform(101.2, 104.5), 1) if is_d else round(random.uniform(97.6, 99.6), 1)
            writer.writerow([temp, 1 if is_d else 0])
    print(f"[OK] {simple_csv_path.name}  (70 rows)")
    print()

print("Done — named patient EHRs and simple CSVs added for Coimbatore, Delhi, and Mysore.")
