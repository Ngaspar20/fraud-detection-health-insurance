"""
network_analysis.py
-------------------
Relational analytics layer (client feedback point 2 — Entrega 2).

Detects organised-fraud patterns in the provider <-> member bipartite network
that no per-claim statistical model can see:

  1. STAR pattern   : a member whose claims are overwhelmingly concentrated in
                      a single provider (possible captive member / collusion)
  2. CLIQUE pattern : a pair of providers sharing an implausibly large group
                      of common members (possible referral collusion ring)
  3. IMPLAUSIBLE PROCEDURE COMBOS : same member, same day, a pair of
                      procedures that statistically never co-occur in the
                      portfolio (learned from the data itself — no clinical
                      knowledge base required)

Implemented with pandas only — no networkx dependency, keeps the cloud
deploy lightweight.

Outputs per claim: network_score (0-100) + network_flags (list[str]),
plus provider-pair and star tables for the dashboard.
"""

import pandas as pd
import numpy as np
from itertools import combinations

# ── Tunables (conservative defaults) ──────────────────────────────────────────
STAR_MIN_CLAIMS    = 8      # member needs >= this many claims to evaluate
STAR_CONCENTRATION = 0.80   # >= 80% of member's claims with one provider
CLIQUE_MIN_SHARED  = 5      # provider pair must share >= this many members
CLIQUE_LIFT        = 4.0    # shared members >= LIFT x expected under independence
COMBO_MIN_FREQ     = 20     # each procedure must individually appear >= this
                            # often for a never-seen pair to be suspicious


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns
    -------
    results : DataFrame indexed like df with network_score / network_flags
    stats   : {"stars": DataFrame, "cliques": DataFrame, "combos": DataFrame}
              for dashboard display
    """
    n = len(df)
    flags = [[] for _ in range(n)]
    score = np.zeros(n)

    members = df["member_id"].astype(str)
    providers = df["provider_id"].astype(str)

    # ── 1. STAR pattern ───────────────────────────────────────────────────────
    star_rows = []
    mp_counts = df.groupby([members, providers])["claim_id"].count()
    m_totals = df.groupby(members)["claim_id"].count()

    star_pairs = set()
    for (m, p), cnt in mp_counts.items():
        total = m_totals[m]
        if total >= STAR_MIN_CLAIMS and cnt / total >= STAR_CONCENTRATION:
            star_pairs.add((m, p))
            star_rows.append({"member_id": m, "provider_id": p,
                              "claims_member": int(total),
                              "claims_with_provider": int(cnt),
                              "concentration": round(cnt / total, 2)})

    if star_pairs:
        mask = [(m, p) in star_pairs for m, p in zip(members, providers)]
        for i in np.where(mask)[0]:
            conc = mp_counts[(members.iloc[i], providers.iloc[i])] / m_totals[members.iloc[i]]
            flags[i].append(
                f"Padrão estrela: {conc:.0%} dos claims do beneficiário "
                f"concentrados neste prestador")
            score[i] += 35

    # ── 2. CLIQUE pattern (provider pairs sharing too many members) ──────────
    clique_rows = []
    prov_members = df.groupby(providers)["member_id"].apply(
        lambda s: set(s.astype(str)))
    n_members_total = members.nunique()
    flagged_pairs = []

    provs = list(prov_members.index)
    for p1, p2 in combinations(provs, 2):
        s1, s2 = prov_members[p1], prov_members[p2]
        shared = len(s1 & s2)
        if shared < CLIQUE_MIN_SHARED:
            continue
        # Expected overlap under independence
        expected = len(s1) * len(s2) / max(n_members_total, 1)
        lift = shared / expected if expected > 0 else 0
        if lift >= CLIQUE_LIFT:
            flagged_pairs.append((p1, p2, s1 & s2))
            clique_rows.append({"provider_1": p1, "provider_2": p2,
                                "shared_members": shared,
                                "expected": round(expected, 1),
                                "lift": round(lift, 1)})

    for p1, p2, shared_members in flagged_pairs:
        in_pair = (providers.isin([p1, p2])) & (members.isin(shared_members))
        for i in np.where(in_pair.values)[0]:
            flags[i].append(
                f"Par de prestadores com sobreposição anómala de "
                f"beneficiários ({p1} ↔ {p2})")
            score[i] += 25

    # ── 3. Implausible same-day procedure combinations ───────────────────────
    combo_rows = []
    if "procedure_code" in df.columns and "claim_date" in df.columns:
        proc = df["procedure_code"].astype(str)
        day = pd.to_datetime(df["claim_date"], errors="coerce").dt.date
        proc_freq = proc.value_counts()

        # Pair counts across the whole portfolio (same member, same day)
        episode = pd.DataFrame({"m": members, "d": day, "p": proc,
                                "idx": np.arange(n)})
        episode = episode.dropna(subset=["d"])
        pair_counts = {}
        episode_groups = episode.groupby(["m", "d"])
        for _, g in episode_groups:
            codes = sorted(set(g["p"]))
            for a, b in combinations(codes, 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        for (m, d), g in episode_groups:
            codes = sorted(set(g["p"]))
            if len(codes) < 2:
                continue
            for a, b in combinations(codes, 2):
                # Both procedures individually common, but the pair is
                # (almost) never seen elsewhere -> implausible combination
                if (proc_freq.get(a, 0) >= COMBO_MIN_FREQ
                        and proc_freq.get(b, 0) >= COMBO_MIN_FREQ
                        and pair_counts.get((a, b), 0) <= 1):
                    for i in g["idx"]:
                        flags[i].append(
                            f"Combinação implausível no mesmo dia: "
                            f"{a} + {b} (nunca co-ocorrem na carteira)")
                        score[i] += 20
                    combo_rows.append({"member_id": m, "date": str(d),
                                       "proc_a": a, "proc_b": b})

    results = pd.DataFrame({
        "network_score": np.clip(score, 0, 100),
        "network_flags": flags,
    }, index=df.index)

    stats = {
        "stars":   pd.DataFrame(star_rows),
        "cliques": pd.DataFrame(clique_rows),
        "combos":  pd.DataFrame(combo_rows),
    }
    return results, stats
