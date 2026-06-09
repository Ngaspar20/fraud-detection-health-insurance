import pandas as pd
import numpy as np

WEIGHTS = {
    "Detecção de Anomalias":  0.35,
    "Risco do Prestador":     0.25,
    "Risco do Beneficiário":  0.20,
    "Custo Atípico":          0.20,
}
WEIGHT_KEYS = list(WEIGHTS.keys())
WEIGHT_VALS = list(WEIGHTS.values())


def _confidence_interval(row: pd.Series, z: float = 1.0) -> tuple:
    """
    Approximate 68% CI (z=1) for the composite score using variance of
    weighted sub-scores.  Returns (low, high) clipped to [0, 100].
    """
    scores = np.array([
        row["anomaly_score"],
        row["provider_risk_score"],
        row["member_risk_score"],
        row["cost_outlier_score"],
    ])
    weights = np.array(WEIGHT_VALS)
    mean   = (scores * weights).sum()
    # Weighted standard deviation of the sub-scores around the composite mean
    variance = (weights * (scores - mean) ** 2).sum()
    std  = np.sqrt(variance)
    margin = z * std * 0.5   # scale to a reasonable half-width
    return (
        float(np.clip(mean - margin, 0, 100).round(1)),
        float(np.clip(mean + margin, 0, 100).round(1)),
    )


def _top_risk_factors(row: pd.Series) -> str:
    """
    Return the top-3 contributing risk factors as a readable string,
    ranked by weighted contribution to the composite score.
    """
    scores = [
        row["anomaly_score"],
        row["provider_risk_score"],
        row["member_risk_score"],
        row["cost_outlier_score"],
    ]
    contributions = [(WEIGHT_KEYS[i], scores[i] * WEIGHT_VALS[i], scores[i])
                     for i in range(4)]
    top3 = sorted(contributions, key=lambda x: x[1], reverse=True)[:3]
    parts = []
    for label, contrib, raw in top3:
        bar = "█" * int(raw / 20)   # visual bar (0–5 blocks)
        parts.append(f"{label}: {raw:.0f} {bar}")
    return " | ".join(parts)


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
    Returns df with: risk_score, risk_score_low, risk_score_high,
                     risk_level, risk_flags, top_risk_factors, adjudication.
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

    # Merge flags
    a_flags  = anomaly_results["anomaly_flags"].tolist()
    co_flags = cost_results["cost_outlier_flags"].tolist()
    all_flags = [a_flags[i] + co_flags[i] for i in range(len(out))]
    out["risk_flags"] = all_flags

    # Floor: if any flags exist, score must be >= 40
    has_flags = out["risk_flags"].apply(lambda f: len(f) > 0)
    out.loc[has_flags & (out["risk_score"] < 40), "risk_score"] = 40.0
    out["risk_score"] = out["risk_score"].clip(0, 100).round(1)

    # ── Confidence intervals ──────────────────────────────────────────────────
    ci = out.apply(_confidence_interval, axis=1, result_type="expand")
    out["risk_score_low"]  = ci[0]
    out["risk_score_high"] = ci[1]

    # ── Top risk factors ──────────────────────────────────────────────────────
    out["top_risk_factors"] = out.apply(_top_risk_factors, axis=1)

    # Risk level
    out["risk_level"] = pd.cut(
        out["risk_score"],
        bins=[-1, 39.99, 69.99, 100],
        labels=["Low", "Medium", "High"]
    )

    # Auto-adjudication
    def adjudicate(row):
        if row["risk_score"] >= 70:
            return "Investigar Urgente"
        elif row["risk_score"] >= 40:
            return "Rever Manualmente"
        return "Aprovar Automaticamente"

    out["adjudication"] = out.apply(adjudicate, axis=1)

    # Flags → string
    out["risk_flags"] = out["risk_flags"].apply(
        lambda f: "; ".join(f) if f else "Sem sinais especificos detectados"
    )

    return out
