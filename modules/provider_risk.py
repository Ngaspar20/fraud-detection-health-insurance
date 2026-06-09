import pandas as pd
import numpy as np
from .column_detector import ColumnProfile


def _scale(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)


def run(df: pd.DataFrame, profile: ColumnProfile) -> pd.DataFrame:
    """Returns provider-level risk dataframe with provider_risk_score 0-100."""
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    df = df.copy()
    df["_amount"] = amounts

    grp = df.groupby("provider_id")

    metrics = pd.DataFrame(index=grp.groups.keys())
    metrics.index.name = "provider_id"

    # Solicitação volume
    metrics["claim_count"] = grp["claim_id"].count()

    # Average solicitação amount
    metrics["avg_amount"] = grp["_amount"].mean()

    # Std deviation of amounts (high std = erratic billing)
    metrics["std_amount"] = grp["_amount"].std().fillna(0)

    # Duplicate rate
    dup_cols = ["member_id", "claim_amount"]
    if profile.has_service_date and "service_date" in df.columns:
        dup_cols.append("service_date")
    dup_mask = df.duplicated(subset=["provider_id"] + dup_cols, keep=False)
    dup_rate = df[dup_mask].groupby("provider_id")["claim_id"].count() / metrics["claim_count"]
    metrics["dup_rate"] = dup_rate.fillna(0)

    # Round-number billing rate
    round_mask = (amounts % 100 == 0) & (amounts >= 500)
    round_rate = df[round_mask].groupby("provider_id")["claim_id"].count() / metrics["claim_count"]
    metrics["round_rate"] = round_rate.fillna(0)

    # Weekend/holiday billing ratio
    if profile.has_service_date and "service_date" in df.columns:
        df["_dow"] = pd.to_datetime(df["service_date"], errors="coerce").dt.dayofweek
        weekend = df[df["_dow"] >= 5].groupby("provider_id")["claim_id"].count()
        metrics["weekend_rate"] = (weekend / metrics["claim_count"]).fillna(0)
    else:
        metrics["weekend_rate"] = 0.0

    # Compute scaled scores
    s_volume   = _scale(metrics["claim_count"])
    s_avg      = _scale(metrics["avg_amount"])
    s_std      = _scale(metrics["std_amount"])
    s_dup      = _scale(metrics["dup_rate"])
    s_round    = _scale(metrics["round_rate"])
    s_weekend  = _scale(metrics["weekend_rate"])

    weights = {"volume": 0.10, "avg": 0.25, "std": 0.15, "dup": 0.25, "round": 0.15, "weekend": 0.10}

    metrics["provider_risk_score"] = (
        s_volume  * weights["volume"] +
        s_avg     * weights["avg"]    +
        s_std     * weights["std"]    +
        s_dup     * weights["dup"]    +
        s_round   * weights["round"]  +
        s_weekend * weights["weekend"]
    ).clip(0, 100)

    # Provider-level flags
    metrics["provider_flags"] = ""
    metrics.loc[metrics["dup_rate"] > 0.1, "provider_flags"] += "Taxa elevada de duplicados; "
    metrics.loc[metrics["round_rate"] > 0.3, "provider_flags"] += "Facturação excessiva em valores redondos; "
    metrics.loc[s_avg > 80, "provider_flags"] += "Valor médio muito acima dos pares; "
    if profile.has_service_date:
        metrics.loc[metrics["weekend_rate"] > 0.3, "provider_flags"] += "Facturação elevada ao fim de semana; "

    metrics = metrics.reset_index()

    # Map provider risk score back to solicitação-level
    score_map = metrics.set_index("provider_id")["provider_risk_score"].to_dict()
    df["provider_risk_score"] = df["provider_id"].map(score_map).fillna(50)

    return df[["claim_id", "provider_risk_score"]].set_index("claim_id"), metrics


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-provider-per-month metrics for trend charts.
    Returns a DataFrame with columns:
        provider_id, month, claim_count, avg_amount, total_amount, dup_count
    Only works when claim_date is present; returns empty DF otherwise.
    """
    if "claim_date" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["_amount"] = pd.to_numeric(tmp["claim_amount"], errors="coerce")
    tmp["_date"]   = pd.to_datetime(tmp["claim_date"], errors="coerce")
    tmp["month"]   = tmp["_date"].dt.to_period("M").astype(str)

    dup_mask = tmp.duplicated(subset=["provider_id", "member_id", "_amount"], keep=False)

    grp = tmp.groupby(["provider_id", "month"])
    trends = pd.DataFrame({
        "claim_count":  grp["claim_id"].count(),
        "avg_amount":   grp["_amount"].mean().round(2),
        "total_amount": grp["_amount"].sum().round(2),
        "dup_count":    tmp[dup_mask].groupby(["provider_id", "month"])["claim_id"].count(),
    }).reset_index()

    trends["dup_count"] = trends["dup_count"].fillna(0).astype(int)
    trends = trends.sort_values(["provider_id", "month"])
    return trends
