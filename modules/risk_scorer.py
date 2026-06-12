import pandas as pd
import numpy as np

# Default weights of the hybrid architecture. When the rules layer is active
# (reference tables loaded / rules firing) deterministic rules carry the
# largest weight — they are the most reliable signal. Without rules the
# weights renormalise to the original statistical-only profile.
DEFAULT_WEIGHTS = {
    "anomaly":  0.25,
    "provider": 0.20,
    "member":   0.15,
    "cost":     0.15,
    "rules":    0.25,
}

# Legacy statistical-only weights (used when rule layer contributes nothing)
LEGACY_WEIGHTS = {
    "anomaly":  0.35,
    "provider": 0.25,
    "member":   0.20,
    "cost":     0.20,
    "rules":    0.00,
}

LABELS = {
    "anomaly":  "Detecção de Anomalias",
    "provider": "Risco do Prestador",
    "member":   "Risco do Beneficiário",
    "cost":     "Custo Atípico",
    "rules":    "Regras de Negócio",
}

SCORE_COLS = {
    "anomaly":  "anomaly_score",
    "provider": "provider_risk_score",
    "member":   "member_risk_score",
    "cost":     "cost_outlier_score",
    "rules":    "rule_score",
}

# Multiplier applied when 2+ independent layers flag the same claim
# (rule layer + statistical layer + temporal layer converging)
CONVERGENCE_BONUS = 1.15


def _confidence_interval(row: pd.Series, weights: dict, z: float = 1.0) -> tuple:
    """Approximate 68% CI (z=1) for the composite score."""
    scores = np.array([row[SCORE_COLS[k]] for k in DEFAULT_WEIGHTS])
    w = np.array([weights[k] for k in DEFAULT_WEIGHTS])
    mean = (scores * w).sum()
    variance = (w * (scores - mean) ** 2).sum()
    std = np.sqrt(variance)
    margin = z * std * 0.5
    return (
        float(np.clip(mean - margin, 0, 100).round(1)),
        float(np.clip(mean + margin, 0, 100).round(1)),
    )


def _top_risk_factors(row: pd.Series, weights: dict) -> str:
    """Top-3 contributing risk factors ranked by weighted contribution."""
    contributions = []
    for key in DEFAULT_WEIGHTS:
        raw = row[SCORE_COLS[key]]
        contributions.append((LABELS[key], raw * weights[key], raw))
    top3 = sorted(contributions, key=lambda x: x[1], reverse=True)[:3]
    parts = []
    for label, contrib, raw in top3:
        bar = "█" * int(raw / 20)
        parts.append(f"{label}: {raw:.0f} {bar}")
    return " | ".join(parts)


def compute(
    df: pd.DataFrame,
    anomaly_results: pd.DataFrame,
    provider_claim_scores: pd.DataFrame,
    member_claim_scores: pd.DataFrame,
    cost_results: pd.DataFrame,
    rule_results: pd.DataFrame = None,
    temporal_results: pd.DataFrame = None,
    network_results: pd.DataFrame = None,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Merge all module outputs into a unified claim-level risk score.

    Hybrid architecture: deterministic rules + statistical analytics + ML.
    - rule_results / temporal_results are optional: without them the score
      falls back to the statistical-only weight profile (graceful degradation).
    - weights: optional user-configured dict with keys
      anomaly/provider/member/cost/rules (auto-normalised to sum 1).
    - Convergence bonus: claims flagged by 2+ independent layers get a
      multiplier — converging evidence means lower false-positive odds.
    """
    out = df.copy()

    out["anomaly_score"]       = anomaly_results["anomaly_score"].values
    out["provider_risk_score"] = provider_claim_scores.reindex(df["claim_id"]).values
    out["member_risk_score"]   = member_claim_scores.reindex(df["claim_id"]).values
    out["cost_outlier_score"]  = cost_results["cost_outlier_score"].values
    out["peer_mean"]           = cost_results["peer_mean"].values

    if rule_results is not None:
        out["rule_score"] = rule_results["rule_score"].values
        r_flags = rule_results["rule_flags"].tolist()
    else:
        out["rule_score"] = 0.0
        r_flags = [[] for _ in range(len(out))]

    if temporal_results is not None:
        t_flags = temporal_results["temporal_flags"].tolist()
        # Temporal signals reinforce the anomaly component
        out["anomaly_score"] = np.clip(
            out["anomaly_score"].values
            + temporal_results["temporal_score"].values * 0.3, 0, 100)
    else:
        t_flags = [[] for _ in range(len(out))]

    if network_results is not None:
        n_flags = network_results["network_flags"].tolist()
        # Relational signals reinforce provider and member components
        out["provider_risk_score"] = np.clip(
            pd.to_numeric(out["provider_risk_score"], errors="coerce").fillna(50).values
            + network_results["network_score"].values * 0.25, 0, 100)
        out["member_risk_score"] = np.clip(
            pd.to_numeric(out["member_risk_score"], errors="coerce").fillna(50).values
            + network_results["network_score"].values * 0.20, 0, 100)
    else:
        n_flags = [[] for _ in range(len(out))]

    for col in SCORE_COLS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(
            50 if col != "rule_score" else 0)

    # ── Weight selection ──────────────────────────────────────────────────────
    rules_active = float(out["rule_score"].max() or 0) > 0
    if weights:
        w = {k: float(weights.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}
    else:
        w = dict(DEFAULT_WEIGHTS) if rules_active else dict(LEGACY_WEIGHTS)
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    out["risk_score"] = sum(
        out[SCORE_COLS[k]] * w[k] for k in DEFAULT_WEIGHTS
    ).clip(0, 100).round(1)

    # ── Merge flags from all layers ───────────────────────────────────────────
    a_flags  = anomaly_results["anomaly_flags"].tolist()
    co_flags = cost_results["cost_outlier_flags"].tolist()
    all_flags = [a_flags[i] + co_flags[i] + r_flags[i] + t_flags[i] + n_flags[i]
                 for i in range(len(out))]
    out["risk_flags"] = all_flags

    # ── Convergence bonus: multiple independent layers agreeing ─────────────
    def n_layers(i):
        layers = 0
        if a_flags[i] or co_flags[i]:
            layers += 1                      # statistical layer
        if r_flags[i]:
            layers += 1                      # deterministic rules layer
        if t_flags[i]:
            layers += 1                      # temporal layer
        if n_flags[i]:
            layers += 1                      # relational/network layer
        return layers

    layer_counts = np.array([n_layers(i) for i in range(len(out))])
    out["evidence_layers"] = layer_counts
    converged = layer_counts >= 2
    out.loc[converged, "risk_score"] = (
        out.loc[converged, "risk_score"] * CONVERGENCE_BONUS
    ).clip(0, 100).round(1)

    # Floor: if any flags exist, score must be >= 40
    has_flags = out["risk_flags"].apply(lambda f: len(f) > 0)
    out.loc[has_flags & (out["risk_score"] < 40), "risk_score"] = 40.0
    # Hard rule violation (rule_score >= 60) floors the composite at 60
    hard_rule = out["rule_score"] >= 60
    out.loc[hard_rule & (out["risk_score"] < 60), "risk_score"] = 60.0
    out["risk_score"] = out["risk_score"].clip(0, 100).round(1)

    # ── Confidence intervals ──────────────────────────────────────────────────
    ci = out.apply(lambda r: _confidence_interval(r, w), axis=1, result_type="expand")
    out["risk_score_low"]  = ci[0]
    out["risk_score_high"] = ci[1]

    # ── Top risk factors ──────────────────────────────────────────────────────
    out["top_risk_factors"] = out.apply(lambda r: _top_risk_factors(r, w), axis=1)

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


# Backwards-compat exports (other modules/pages may import these)
WEIGHTS = {LABELS[k]: v for k, v in LEGACY_WEIGHTS.items() if v > 0}
WEIGHT_KEYS = list(WEIGHTS.keys())
WEIGHT_VALS = list(WEIGHTS.values())
