"""
temporal_analysis.py
--------------------
Expanded analytics layer (client feedback point 2):

  - Benford's Law conformity per provider (fabricated amounts detection)
  - Claim velocity per provider (volume bursts)
  - End-of-year concentration (forced use of annual benefit limits)

Each analysis returns per-claim flags merged into the rule/statistical layer.
All analyses degrade gracefully when required columns are missing.
"""

import pandas as pd
import numpy as np

# Expected leading-digit distribution under Benford's Law
BENFORD = np.log10(1 + 1 / np.arange(1, 10))

MIN_CLAIMS_BENFORD = 30   # below this, the test has no statistical power


def _leading_digit(values: np.ndarray) -> np.ndarray:
    """First significant digit of each positive value (0 for invalid)."""
    out = np.zeros(len(values), dtype=int)
    pos = values > 0
    v = values[pos]
    mag = np.floor(np.log10(v))
    out[pos] = (v / 10 ** mag).astype(int)
    return out


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    results     : DataFrame indexed like df with temporal_flags (list[str])
                  and temporal_score (0-100)
    provider_stats : per-provider summary (benford deviation, velocity, etc.)
                     for dashboard display
    """
    n = len(df)
    flags = [[] for _ in range(n)]
    score = np.zeros(n)

    amounts = pd.to_numeric(df["claim_amount"], errors="coerce").fillna(0).values
    providers = df["provider_id"].astype(str).values
    has_date = "claim_date" in df.columns

    prov_stats = {}

    # ── Benford's Law per provider ────────────────────────────────────────────
    digits = _leading_digit(amounts)
    prov_series = pd.Series(providers)
    benford_dev = {}
    for prov, idx in prov_series.groupby(prov_series).groups.items():
        idx = np.asarray(idx)
        d = digits[idx]
        d = d[d > 0]
        if len(d) < MIN_CLAIMS_BENFORD:
            continue
        observed = np.bincount(d, minlength=10)[1:10] / len(d)
        # Mean absolute deviation vs Benford expectation (MAD)
        mad = float(np.abs(observed - BENFORD).mean())
        benford_dev[prov] = mad
        # Nigrini thresholds: MAD > 0.015 = nonconformity (suspicious)
        if mad > 0.015:
            for i in idx:
                flags[i].append(
                    f"Distribuição de valores do prestador desvia da Lei de "
                    f"Benford (MAD {mad:.3f})")
                score[i] += 25
    for prov, mad in benford_dev.items():
        prov_stats.setdefault(prov, {})["benford_mad"] = round(mad, 4)

    if has_date:
        dates = pd.to_datetime(df["claim_date"], errors="coerce")

        # ── Velocity: provider claims per day vs own median ──────────────────
        day = dates.dt.date
        daily_counts = df.groupby([prov_series.values, day])["claim_id"].transform("count")
        prov_median_daily = daily_counts.groupby(prov_series.values).transform("median")
        burst = (daily_counts >= 5) & (daily_counts > prov_median_daily * 4)
        for i in np.where(burst.values)[0]:
            flags[i].append(
                f"Pico de volume: {int(daily_counts.iloc[i])} actos no dia "
                f"(mediana do prestador: {prov_median_daily.iloc[i]:.0f})")
            score[i] += 20

        # ── End-of-year concentration (last 3 weeks of December) ─────────────
        eoy = (dates.dt.month == 12) & (dates.dt.day >= 10)
        prov_eoy_rate = eoy.groupby(prov_series.values).transform("mean")
        prov_n = df.groupby(prov_series.values)["claim_id"].transform("count")
        # > 25% of a provider's claims compressed into ~3 weeks of the year
        eoy_hit = eoy & (prov_eoy_rate > 0.25) & (prov_n >= 20)
        for i in np.where(eoy_hit.values)[0]:
            flags[i].append(
                f"Concentração de fim de ano: {prov_eoy_rate.iloc[i]:.0%} dos "
                f"claims do prestador em meados de Dezembro")
            score[i] += 15

        for prov in set(providers):
            mask = prov_series.values == prov
            prov_stats.setdefault(prov, {})["eoy_rate"] = round(
                float(eoy.values[mask].mean()), 3)

    results = pd.DataFrame({
        "temporal_score": np.clip(score, 0, 100),
        "temporal_flags": flags,
    }, index=df.index)

    stats_df = (pd.DataFrame.from_dict(prov_stats, orient="index")
                .rename_axis("provider_id").reset_index()
                if prov_stats else pd.DataFrame(columns=["provider_id"]))

    return results, stats_df
