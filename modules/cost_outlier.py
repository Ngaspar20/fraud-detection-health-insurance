import pandas as pd
import numpy as np
from .column_detector import ColumnProfile


def run(df: pd.DataFrame, profile: ColumnProfile) -> pd.DataFrame:
    """Returns solicitação-level cost_outlier_score (0-100) and cost_outlier_flags."""
    df = df.copy()
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
    df["_amount"] = amounts

    results = pd.DataFrame(index=df.index)
    results["cost_outlier_score"] = 0.0
    results["cost_outlier_flags"] = [[] for _ in range(len(df))]
    results["peer_mean"] = np.nan
    results["peer_std"] = np.nan

    def _score_vs_group(group_key):
        grouped = amounts.groupby(df[group_key])
        peer_mean = grouped.transform("mean")
        peer_std  = grouped.transform("std").fillna(0)
        z = (amounts - peer_mean) / (peer_std.replace(0, np.nan))
        pct_above = ((amounts - peer_mean) / peer_mean.replace(0, np.nan) * 100).fillna(0)

        # Score: how far above the peer mean (0-100)
        score = (z.clip(0, 5) / 5 * 100).fillna(0).clip(0, 100)

        flagged = (z > 2).fillna(False)
        return score, flagged, pct_above, peer_mean, peer_std

    # Choose grouping strategy: procedure > specialty > overall
    if profile.has_procedure_code and "procedure_code" in df.columns:
        score, flagged, pct_above, peer_mean, peer_std = _score_vs_group("procedure_code")
        label = "procedimento"
    elif profile.has_provider_specialty and "provider_specialty" in df.columns:
        score, flagged, pct_above, peer_mean, peer_std = _score_vs_group("provider_specialty")
        label = "especialidade"
    else:
        peer_mean_val = amounts.mean()
        peer_std_val  = amounts.std()
        z = (amounts - peer_mean_val) / (peer_std_val if peer_std_val > 0 else 1)
        pct_above = ((amounts - peer_mean_val) / (peer_mean_val if peer_mean_val > 0 else 1) * 100)
        score = (z.clip(0, 5) / 5 * 100).fillna(0).clip(0, 100)
        flagged = (z > 2).fillna(False)
        peer_mean = pd.Series(peer_mean_val, index=df.index)
        peer_std  = pd.Series(peer_std_val,  index=df.index)
        label = "carteira"

    results["cost_outlier_score"] = score.values
    results["peer_mean"] = peer_mean.values
    results["peer_std"]  = peer_std.values

    for idx in results.index[flagged.values]:
        pct = pct_above.iloc[idx]
        results.at[idx, "cost_outlier_flags"].append(
            f"Custo {pct:.0f}% acima da média de referência ({label})"
        )

    return results
