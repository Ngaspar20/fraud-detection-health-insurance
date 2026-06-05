import pandas as pd
import numpy as np
from .column_detector import ColumnProfile


def _scale(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)


def run(df: pd.DataFrame, profile: ColumnProfile) -> pd.DataFrame:
    """Returns member-level risk dataframe and solicitação-level member_risk_score."""
    df = df.copy()
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    df["_amount"] = amounts

    grp = df.groupby("member_id")

    metrics = pd.DataFrame(index=grp.groups.keys())
    metrics.index.name = "member_id"

    # Total solicitações
    metrics["solicitação_count"] = grp["claim_id"].count()

    # Total spend
    metrics["total_spend"] = grp["_amount"].sum()

    # Avg solicitação amount
    metrics["avg_amount"] = grp["_amount"].mean()

    # Distinct providers (multi-provider shopping)
    metrics["distinct_providers"] = grp["provider_id"].nunique()

    # Same-day multi-solicitação
    if profile.has_service_date and "service_date" in df.columns:
        df["_sdate"] = pd.to_datetime(df["service_date"], errors="coerce").dt.date
        same_day = df.groupby(["member_id", "_sdate"])["claim_id"].count()
        max_same_day = same_day.groupby(level="member_id").max()
        metrics["max_same_day_solicitaçãos"] = max_same_day.reindex(metrics.index).fillna(1)
    else:
        metrics["max_same_day_solicitaçãos"] = 1.0

    s_count    = _scale(metrics["solicitação_count"])
    s_spend    = _scale(metrics["total_spend"])
    s_avg      = _scale(metrics["avg_amount"])
    s_providers = _scale(metrics["distinct_providers"])
    s_sameday  = _scale(metrics["max_same_day_solicitaçãos"])

    weights = {"count": 0.20, "spend": 0.25, "avg": 0.20, "providers": 0.20, "sameday": 0.15}

    metrics["member_risk_score"] = (
        s_count     * weights["count"]     +
        s_spend     * weights["spend"]     +
        s_avg       * weights["avg"]       +
        s_providers * weights["providers"] +
        s_sameday   * weights["sameday"]
    ).clip(0, 100)

    metrics["member_flags"] = ""
    metrics.loc[metrics["distinct_providers"] > 5, "member_flags"] += "Acesso a múltiplos prestadores (>5); "
    metrics.loc[s_spend > 85, "member_flags"] += "Gasto total muito acima do grupo de referência; "
    if profile.has_service_date:
        metrics.loc[metrics["max_same_day_solicitaçãos"] > 2, "member_flags"] += "Múltiplas solicitações no mesmo dia; "

    metrics = metrics.reset_index()

    score_map = metrics.set_index("member_id")["member_risk_score"].to_dict()
    df["member_risk_score"] = df["member_id"].map(score_map).fillna(50)

    return df[["claim_id", "member_risk_score"]].set_index("claim_id"), metrics
