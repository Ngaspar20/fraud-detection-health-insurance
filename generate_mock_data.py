"""
Generate a realistic mock claims dataset for testing the Health Claims Intelligence Platform.
Injects known fraud patterns so they surface clearly in the app.

Run:
    conda activate ml_env
    python generate_mock_data.py
Output: mock_claims.csv  (same folder)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_NORMAL        = 1800
N_FRAUD         = 120   # injected anomalies
START_DATE      = datetime(2023, 1, 1)
END_DATE        = datetime(2024, 12, 31)

SPECIALTIES     = ["Cardiology", "Orthopedics", "General Practice", "Oncology",
                   "Neurology", "Radiology", "Physical Therapy", "Psychiatry"]
PROCEDURE_CODES = ["99213", "99214", "99215", "27447", "93000", "71046",
                   "99232", "90837", "20610", "70553", "99283", "43239"]
DIAGNOSIS_CODES = ["Z00.00", "I10", "M54.5", "E11.9", "J06.9", "K21.0",
                   "F32.9", "M17.11", "G89.29", "Z12.31", "I25.10", "N39.0"]
DRUG_NAMES      = ["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine",
                   "Omeprazole", "Metoprolol", "Sertraline", "Gabapentin"]

def rand_date(start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def rand_amount(mean=400, std=250, low=50, high=8000):
    return round(np.clip(np.random.normal(mean, std), low, high), 2)

# ── Provider pool ──────────────────────────────────────────────────────────────
N_PROVIDERS = 40
providers = [f"PRV{str(i).zfill(4)}" for i in range(1, N_PROVIDERS + 1)]
provider_specialty = {p: random.choice(SPECIALTIES) for p in providers}

# Flag 3 providers as high-risk
fraud_providers = providers[:3]

# ── Member pool ───────────────────────────────────────────────────────────────
N_MEMBERS = 300
members = [f"MBR{str(i).zfill(5)}" for i in range(1, N_MEMBERS + 1)]
member_ages = {m: random.randint(18, 80) for m in members}
member_genders = {m: random.choice(["M", "F"]) for m in members}

# Flag 5 members as high-risk (multi-provider shoppers)
fraud_members = members[:5]

# ── Normal claims ─────────────────────────────────────────────────────────────
rows = []

for i in range(N_NORMAL):
    claim_date   = rand_date()
    service_date = claim_date - timedelta(days=random.randint(0, 5))
    provider     = random.choice(providers)
    member       = random.choice(members)
    amount       = rand_amount()

    rows.append({
        "claim_id":          f"CLM{str(i+1).zfill(6)}",
        "member_id":         member,
        "provider_id":       provider,
        "claim_date":        claim_date.strftime("%Y-%m-%d"),
        "service_date":      service_date.strftime("%Y-%m-%d"),
        "claim_amount":      amount,
        "paid_amount":       round(amount * random.uniform(0.7, 0.95), 2),
        "diagnosis_code":    random.choice(DIAGNOSIS_CODES),
        "procedure_code":    random.choice(PROCEDURE_CODES),
        "provider_specialty": provider_specialty[provider],
        "member_age":        member_ages[member],
        "member_gender":     member_genders[member],
        "claim_type":        random.choice(["Medical", "Pharmacy", "Dental"]),
        "drug_name":         random.choice(DRUG_NAMES) if random.random() < 0.3 else "",
    })

# ── Inject fraud pattern 1: High-amount outliers from fraud providers ─────────
for i in range(30):
    provider = random.choice(fraud_providers)
    claim_date = rand_date()
    amount = round(random.uniform(8000, 25000), 2)
    rows.append({
        "claim_id":          f"FRD{str(i+1).zfill(6)}",
        "member_id":         random.choice(members),
        "provider_id":       provider,
        "claim_date":        claim_date.strftime("%Y-%m-%d"),
        "service_date":      claim_date.strftime("%Y-%m-%d"),
        "claim_amount":      amount,
        "paid_amount":       round(amount * 0.9, 2),
        "diagnosis_code":    "I10",
        "procedure_code":    "99215",
        "provider_specialty": provider_specialty[provider],
        "member_age":        random.randint(30, 60),
        "member_gender":     "M",
        "claim_type":        "Medical",
        "drug_name":         "",
    })

# ── Inject fraud pattern 2: Duplicate claims ─────────────────────────────────
base_dups = rows[:10]
for dup in base_dups:
    row = dup.copy()
    row["claim_id"] = f"DUP{row['claim_id']}"
    # same member, provider, amount, date → duplicate
    rows.append(row)

# ── Inject fraud pattern 3: Round-number billing ─────────────────────────────
for i in range(25):
    provider = random.choice(fraud_providers)
    claim_date = rand_date()
    amount = random.choice([500, 1000, 1500, 2000, 2500, 5000])
    rows.append({
        "claim_id":          f"RND{str(i+1).zfill(6)}",
        "member_id":         random.choice(members),
        "provider_id":       provider,
        "claim_date":        claim_date.strftime("%Y-%m-%d"),
        "service_date":      claim_date.strftime("%Y-%m-%d"),
        "claim_amount":      float(amount),
        "paid_amount":       float(amount) * 0.85,
        "diagnosis_code":    random.choice(DIAGNOSIS_CODES),
        "procedure_code":    "99215",
        "provider_specialty": provider_specialty[provider],
        "member_age":        45,
        "member_gender":     "F",
        "claim_type":        "Medical",
        "drug_name":         "",
    })

# ── Inject fraud pattern 4: Multi-provider shopping members ──────────────────
many_providers = providers[5:25]  # 20 different providers
for member in fraud_members:
    for j in range(12):
        provider = many_providers[j % len(many_providers)]
        claim_date = rand_date()
        amount = rand_amount(mean=600, std=150)
        rows.append({
            "claim_id":          f"SHP{member[-5:]}{str(j).zfill(3)}",
            "member_id":         member,
            "provider_id":       provider,
            "claim_date":        claim_date.strftime("%Y-%m-%d"),
            "service_date":      claim_date.strftime("%Y-%m-%d"),
            "claim_amount":      amount,
            "paid_amount":       round(amount * 0.88, 2),
            "diagnosis_code":    random.choice(DIAGNOSIS_CODES),
            "procedure_code":    random.choice(PROCEDURE_CODES),
            "provider_specialty": provider_specialty[provider],
            "member_age":        member_ages[member],
            "member_gender":     member_genders[member],
            "claim_type":        "Medical",
            "drug_name":         "",
        })

# ── Inject fraud pattern 5: Same-day multiple claims (one member, one day) ───
same_day_member = members[10]
same_day_date   = datetime(2024, 3, 15).strftime("%Y-%m-%d")
for k in range(6):
    rows.append({
        "claim_id":          f"SDY{str(k).zfill(4)}",
        "member_id":         same_day_member,
        "provider_id":       random.choice(providers[5:15]),
        "claim_date":        same_day_date,
        "service_date":      same_day_date,
        "claim_amount":      rand_amount(mean=300, std=80),
        "paid_amount":       rand_amount(mean=240, std=60),
        "diagnosis_code":    random.choice(DIAGNOSIS_CODES),
        "procedure_code":    random.choice(PROCEDURE_CODES),
        "provider_specialty": random.choice(SPECIALTIES),
        "member_age":        member_ages[same_day_member],
        "member_gender":     member_genders[same_day_member],
        "claim_type":        "Medical",
        "drug_name":         "",
    })

# ── Shuffle and export ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "mock_claims.csv"
df.to_csv(output_path, index=False)

print(f"[OK] Mock dataset generated: {output_path}")
print(f"    Total claims : {len(df):,}")
print(f"    Providers    : {df['provider_id'].nunique()} ({len(fraud_providers)} flagged as fraud providers)")
print(f"    Members      : {df['member_id'].nunique()} ({len(fraud_members)} multi-provider shoppers)")
print(f"    Date range   : {df['claim_date'].min()} → {df['claim_date'].max()}")
print(f"    Fraud patterns injected:")
print(f"      • 30  high-amount outlier claims")
print(f"      • 10  duplicate claims")
print(f"      • 25  round-number billing claims")
print(f"      • {len(fraud_members)*12}  multi-provider shopping claims")
print(f"      •  6  same-day multi-claim (member {same_day_member})")
