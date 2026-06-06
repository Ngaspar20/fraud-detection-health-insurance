import pandas as pd
import numpy as np


def compute(
    df: pd.DataFrame,
    anomaly_results: pd.DataFrame,
    provider_claim_scores: pd.DataFrame,
    member_claim_scores: pd.DataFrame,
    cost_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all module outputs into a unified claim-level risk score.
    Weights: anomaly 35%, provider 25%, member 20%, cost 20%.
    Returns df with: risk_score, risk_level, risk_flags, adjudication columns.
    """
    out = df.copy()

    out["anomaly_score"]       = anomaly_results["anomaly_score"].values
    out["provider_risk_score"] = provider_claim_scores.reindex(df["claim_id"]).values
    out["member_risk_score"]   = member_claim_scores.reindex(df["claim_id"]).values
    out["cost_outlier_score"]  = cost_results["cost_outlier_score"].values
    out["peer_mean"]           = cost_results["peer_mean"].values

    for col in ["anomaly_score", "provider_risk_score", "member_risk_score", "cost_outlier_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(50)

    out["risk_score"] = (
        out["anomaly_score"]       * 0.35 +
        out["provider_risk_score"] * 0.25 +
        out["member_risk_score"]   * 0.20 +
        out["cost_outlier_score"]  * 0.20
    ).clip(0, 100).round(1)

    # Merge flags first
    a_flags  = anomaly_results["anomaly_flags"].tolist()
    co_flags = cost_results["cost_outlier_flags"].tolist()

    all_flags = []
    for i in range(len(out)):
        combined = a_flags[i] + co_flags[i]
        all_flags.append(combined)

    out["risk_flags"] = all_flags

    # If any specific flags exist, the score must be at least 40 (Medio)
    has_flags = out["risk_flags"].apply(lambda f: len(f) > 0)
    out.loc[has_flags & (out["risk_score"] < 40), "risk_score"] = 40.0
    out["risk_score"] = out["risk_score"].clip(0, 100).round(1)

    # Risk level
    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=[-1, 39.99, 69.99, 100],
        labels=["Low", "Medium", "High"]
    )

    # Auto-adjudication recommendation
    def adjudicate(row):
        score = row["risk_score"]
        if score >= 70:
            return "Investigar Urgente"
        elif score >= 40:
            return "Rever Manualmente"
        else:
            return "Aprovar Automaticamente"

    out["adjudication"] = out.apply(adjudicate, axis=1)

    # Convert flags list to readable string
    out["risk_flags"] = out["risk_flags"].apply(
        lambda f: "; ".join(f) if f else "Sem sinais especificos detectados"
    )

    return out
