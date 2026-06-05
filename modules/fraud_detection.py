import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .column_detector import ColumnProfile


def run(df: pd.DataFrame, profile: ColumnProfile, contamination: float = 0.05) -> pd.DataFrame:
    """Returns df with columns: anomaly_score (0-100), anomaly_flags (list of strings)."""
    results = pd.DataFrame(index=df.index)
    results["anomaly_score"] = 0.0
    results["anomaly_flags"] = [[] for _ in range(len(df))]

    # --- Isolation Forest ---
    features = ["claim_amount"]
    if profile.has_paid_amount and "paid_amount" in df.columns:
        features.append("paid_amount")
    if profile.has_member_age and "member_age" in df.columns:
        features.append("member_age")

    feat_df = df[features].copy()
    for col in features:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    iso.fit(X)
    raw_scores = iso.score_samples(X)
    iso_score = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    results["anomaly_score"] = (iso_score * 100).clip(0, 100)

    iso_flagged = iso.predict(X) == -1
    for idx in results.index[iso_flagged]:
        results.at[idx, "anomaly_flags"].append("Anomalia detectada por modelo estatístico")

    # --- Z-score on claim_amount per provider ---
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    prov_mean = amounts.groupby(df["provider_id"]).transform("mean")
    prov_std  = amounts.groupby(df["provider_id"]).transform("std").fillna(0)
    z = (amounts - prov_mean) / (prov_std.replace(0, np.nan))
    high_z = z > 3
    for idx in results.index[high_z.fillna(False)]:
        mult = amounts.iloc[idx] / prov_mean.iloc[idx] if prov_mean.iloc[idx] > 0 else 0
        results.at[idx, "anomaly_flags"].append(f"Facturado {mult:.1f}× acima da média do prestador (Z>{z.iloc[idx]:.1f})")
        results.at[idx, "anomaly_score"] = min(100, results.at[idx, "anomaly_score"] + 15)

    # --- Round-number billing ---
    round_mask = (amounts % 100 == 0) & (amounts >= 500)
    for idx in results.index[round_mask.fillna(False)]:
        results.at[idx, "anomaly_flags"].append("Valor facturado suspeito (número redondo)")
        results.at[idx, "anomaly_score"] = min(100, results.at[idx, "anomaly_score"] + 8)

    # --- Duplicate detection ---
    dup_cols = ["member_id", "provider_id", "claim_amount"]
    if profile.has_service_date and "service_date" in df.columns:
        dup_cols.append("service_date")
    dups = df.duplicated(subset=dup_cols, keep=False)
    for idx in results.index[dups]:
        results.at[idx, "anomaly_flags"].append("Possível solicitação duplicada")
        results.at[idx, "anomaly_score"] = min(100, results.at[idx, "anomaly_score"] + 20)

    # --- Upcoding signal (if procedure_code available) ---
    if profile.has_procedure_code and "procedure_code" in df.columns:
        proc_counts = df.groupby(["provider_id", "procedure_code"])["claim_id"].transform("count")
        proc_avg = df.groupby("procedure_code")["claim_id"].transform("count")
        upcoding = proc_counts > proc_avg * 3
        for idx in results.index[upcoding.fillna(False)]:
            results.at[idx, "anomaly_flags"].append("Procedimento de alta frequência vs. prestadores similares")
            results.at[idx, "anomaly_score"] = min(100, results.at[idx, "anomaly_score"] + 10)

    results["anomaly_score"] = results["anomaly_score"].clip(0, 100)
    return results
