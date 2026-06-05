from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

REQUIRED = ["claim_id", "member_id", "provider_id", "claim_date", "claim_amount"]

ALIASES = {
    "claim_id":          ["claimid", "claim_no", "claimno", "claim_number", "id"],
    "member_id":         ["memberid", "patient_id", "patientid", "insured_id", "member_no"],
    "provider_id":       ["providerid", "provider_no", "npi", "doctor_id", "facility_id"],
    "claim_date":        ["claimdate", "date", "submission_date", "received_date"],
    "claim_amount":      ["amount", "billed_amount", "total_amount", "charged_amount", "claimamount"],
    "diagnosis_code":    ["diagcode", "icd_code", "icd10", "dx_code", "diagnosis"],
    "procedure_code":    ["proccode", "cpt_code", "cpt", "hcpcs", "procedure"],
    "service_date":      ["servicedate", "dos", "date_of_service", "service_from"],
    "paid_amount":       ["paidamount", "approved_amount", "allowed_amount", "payment"],
    "provider_specialty":["specialty", "provider_type", "providertype", "speciality"],
    "member_age":        ["age", "patient_age", "age_years"],
    "member_gender":     ["gender", "sex", "patient_gender"],
    "drug_name":         ["drug", "medication", "drug_description", "rx_name"],
    "ndc_code":          ["ndc", "national_drug_code", "drug_code"],
    "claim_type":        ["type", "claim_category", "service_type", "claimtype"],
    "denial_code":       ["denial", "denial_reason", "remark_code"],
}


@dataclass
class ColumnProfile:
    has_diagnosis_code: bool = False
    has_procedure_code: bool = False
    has_service_date: bool = False
    has_paid_amount: bool = False
    has_provider_specialty: bool = False
    has_member_age: bool = False
    has_member_gender: bool = False
    has_drug_name: bool = False
    has_ndc_code: bool = False
    has_claim_type: bool = False
    has_denial_code: bool = False
    missing_features: list = field(default_factory=list)
    column_map: dict = field(default_factory=dict)


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def detect_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnProfile, list[str]]:
    """Map raw columns to canonical names. Returns renamed df, ColumnProfile, and list of missing required cols."""
    raw_cols = {_normalize(c): c for c in df.columns}
    mapping = {}

    for canonical, aliases in ALIASES.items():
        if canonical in raw_cols:
            mapping[raw_cols[canonical]] = canonical
            continue
        for alias in aliases:
            if alias in raw_cols:
                mapping[raw_cols[alias]] = canonical
                break

    df = df.rename(columns=mapping)

    missing_required = [c for c in REQUIRED if c not in df.columns]

    profile = ColumnProfile()
    profile.column_map = mapping
    profile.has_diagnosis_code  = "diagnosis_code"   in df.columns
    profile.has_procedure_code  = "procedure_code"   in df.columns
    profile.has_service_date    = "service_date"     in df.columns
    profile.has_paid_amount     = "paid_amount"      in df.columns
    profile.has_provider_specialty = "provider_specialty" in df.columns
    profile.has_member_age      = "member_age"       in df.columns
    profile.has_member_gender   = "member_gender"    in df.columns
    profile.has_drug_name       = "drug_name"        in df.columns
    profile.has_ndc_code        = "ndc_code"         in df.columns
    profile.has_claim_type      = "claim_type"       in df.columns
    profile.has_denial_code     = "denial_code"      in df.columns

    notes = []
    if not profile.has_diagnosis_code:
        notes.append("diagnosis_code — upcoding analysis skipped")
    if not profile.has_procedure_code:
        notes.append("procedure_code — using overall distribution for cost outliers")
    if not profile.has_service_date:
        notes.append("service_date — same-day duplicate detection skipped")
    if not profile.has_provider_specialty:
        notes.append("provider_specialty — specialty-level peer benchmarking skipped")
    profile.missing_features = notes

    return df, profile, missing_required
