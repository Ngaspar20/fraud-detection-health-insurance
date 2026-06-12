"""
rules_engine.py
---------------
Deterministic rules layer of the hybrid architecture
(rules + statistical analytics + machine learning).

Each rule inspects the claims DataFrame (plus the client-supplied reference
tables) and flags individual claims. Rules degrade gracefully: when a rule's
data requirement is not met (missing column or missing reference table), it is
skipped and reported as inactive.

Output per claim:
  rule_score  : 0-100 (severity-weighted count of violated rules)
  rule_flags  : list[str] human-readable, prefixed with the rule id

Rule severities: 1 = informative, 2 = suspicious, 3 = hard violation.
"""

import pandas as pd
import numpy as np

from modules import reference_tables

# ── Rule registry ─────────────────────────────────────────────────────────────
# (rule_id, name_pt, severity, default_param, param_label_pt, requirement_desc)

RULES_META = {
    "R01": {"name": "Valor acima do preço de tabela", "severity": 3,
            "param": 15.0, "param_label": "% acima do preço para gerar alerta",
            "requires": "Tabela de preços"},
    "R02": {"name": "Procedimento fora da tabela de preços", "severity": 2,
            "param": None, "param_label": None,
            "requires": "Tabela de preços + procedure_code"},
    "R03": {"name": "Procedimento excluído da cobertura", "severity": 3,
            "param": None, "param_label": None,
            "requires": "Regras de cobertura + procedure_code"},
    "R04": {"name": "Limite por sinistro excedido", "severity": 3,
            "param": None, "param_label": None,
            "requires": "Regras de cobertura"},
    "R05": {"name": "Plafond anual do beneficiário excedido", "severity": 3,
            "param": None, "param_label": None,
            "requires": "Regras de cobertura + claim_date"},
    "R06": {"name": "Procedimento incompatível com género", "severity": 3,
            "param": None, "param_label": None,
            "requires": "member_gender + procedure_code"},
    "R07": {"name": "Procedimento incompatível com idade", "severity": 3,
            "param": None, "param_label": None,
            "requires": "member_age + procedure_code"},
    "R08": {"name": "Volume diário impossível do prestador", "severity": 3,
            "param": 30.0, "param_label": "Máx. de actos por prestador por dia",
            "requires": "claim_date"},
    "R09": {"name": "Acto fora da especialidade do prestador", "severity": 2,
            "param": None, "param_label": None,
            "requires": "provider_specialty + procedure_code"},
    "R10": {"name": "Concentração de claims ao fim-de-semana", "severity": 1,
            "param": 50.0, "param_label": "% de claims do prestador ao fim-de-semana",
            "requires": "claim_date"},
    "R11": {"name": "Beneficiário em múltiplos prestadores no mesmo dia", "severity": 2,
            "param": 3.0, "param_label": "Nº de prestadores distintos no mesmo dia",
            "requires": "claim_date"},
    "R12": {"name": "Valor redondo repetido pelo prestador", "severity": 1,
            "param": 60.0, "param_label": "% de claims do prestador com valores redondos",
            "requires": "claim_amount"},
}

# Procedure-code knowledge used by R06/R07/R09. Deliberately small and
# conservative — extended in partnership with the client's clinical team.
GENDER_RESTRICTED = {
    # code prefix -> allowed gender
    "59":  "F",   # CPT 59xxx obstetrics / delivery
    "58":  "F",   # CPT 58xxx gynecology
    "55":  "M",   # CPT 55xxx male genital surgery
    "54":  "M",   # CPT 54xxx male genital
}
AGE_RESTRICTED = {
    # code prefix -> (min_age, max_age)
    "9938": (0, 18),    # pediatric preventive visits (99381-99385 approx)
    "9939": (0, 18),
}
SPECIALTY_PROCEDURES = {
    # specialty -> procedure-code prefixes typically billed by that specialty
    "Cardiologia":   ["93", "92"],
    "Ortopedia":     ["27", "29", "20"],
    "Oftalmologia":  ["92", "65", "66", "67"],
    "Psiquiatria":   ["908", "909"],
    "Radiologia":    ["7"],
    "Fisioterapia":  ["97"],
}


def _flag(rule_id: str, text: str) -> str:
    return f"[{rule_id}] {text}"


# ── Engine ────────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, profile=None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Evaluate all enabled rules against the claims DataFrame.

    Returns
    -------
    results : DataFrame indexed like df with columns rule_score (0-100) and
              rule_flags (list[str])
    status  : list of dicts per rule: {rule_id, name, active, reason, n_hits}
    """
    n = len(df)
    flags = [[] for _ in range(n)]
    severity_sum = np.zeros(n)

    price = reference_tables.load_price_table()
    coverage = reference_tables.load_coverage_rules()
    settings = reference_tables.load_rule_settings()

    def setting(rule_id):
        meta = RULES_META[rule_id]
        s = settings.get(rule_id, {})
        enabled = s.get("enabled", True)
        param = s.get("param")
        if param is None:
            param = meta["param"]
        return enabled, param

    has_price = not price.empty
    has_coverage = not coverage.empty
    has_proc = "procedure_code" in df.columns
    has_date = "claim_date" in df.columns and df["claim_date"].notna().any()
    has_gender = "member_gender" in df.columns
    has_age = "member_age" in df.columns
    has_spec = "provider_specialty" in df.columns

    status = []
    amounts = pd.to_numeric(df["claim_amount"], errors="coerce").fillna(0).values
    proc = df["procedure_code"].astype(str).str.strip() if has_proc else None

    def record(rule_id, active, reason, n_hits):
        meta = RULES_META[rule_id]
        status.append({"rule_id": rule_id, "name": meta["name"],
                       "severity": meta["severity"], "active": active,
                       "reason": reason, "n_hits": int(n_hits)})

    # R01 — billed above agreed price ─────────────────────────────────────────
    enabled, pct = setting("R01")
    if not enabled:
        record("R01", False, "Desactivada pelo utilizador", 0)
    elif not (has_price and has_proc):
        record("R01", False, "Requer tabela de preços e procedure_code", 0)
    else:
        price_map = dict(zip(price["procedure_code"].astype(str),
                             price["agreed_price"]))
        hits = 0
        for i, (p, amt) in enumerate(zip(proc, amounts)):
            agreed = price_map.get(p)
            if agreed and agreed > 0 and amt > agreed * (1 + pct / 100):
                over = (amt / agreed - 1) * 100
                flags[i].append(_flag("R01",
                    f"Facturado {over:.0f}% acima do preço de tabela "
                    f"({amt:,.0f} vs {agreed:,.0f})"))
                severity_sum[i] += 3
                hits += 1
        record("R01", True, None, hits)

    # R02 — procedure not in price table ──────────────────────────────────────
    enabled, _ = setting("R02")
    if not enabled:
        record("R02", False, "Desactivada pelo utilizador", 0)
    elif not (has_price and has_proc):
        record("R02", False, "Requer tabela de preços e procedure_code", 0)
    else:
        known = set(price["procedure_code"].astype(str))
        hits = 0
        for i, p in enumerate(proc):
            if p and p not in known:
                flags[i].append(_flag("R02",
                    f"Procedimento {p} não consta na tabela de preços"))
                severity_sum[i] += 2
                hits += 1
        record("R02", True, None, hits)

    # R03 — excluded procedure ────────────────────────────────────────────────
    enabled, _ = setting("R03")
    if not enabled:
        record("R03", False, "Desactivada pelo utilizador", 0)
    elif not (has_coverage and has_proc):
        record("R03", False, "Requer regras de cobertura e procedure_code", 0)
    else:
        excluded = set()
        for excl in coverage["excluded_procedures"].dropna():
            for code in str(excl).split(";"):
                code = code.strip()
                if code:
                    excluded.add(code)
        hits = 0
        if excluded:
            for i, p in enumerate(proc):
                if p in excluded:
                    flags[i].append(_flag("R03",
                        f"Procedimento {p} excluído pelas regras de cobertura"))
                    severity_sum[i] += 3
                    hits += 1
        record("R03", True, None, hits)

    # R04 — per-claim limit exceeded ──────────────────────────────────────────
    enabled, _ = setting("R04")
    if not enabled:
        record("R04", False, "Desactivada pelo utilizador", 0)
    elif not has_coverage or coverage["per_claim_limit"].dropna().empty:
        record("R04", False, "Requer regras de cobertura com per_claim_limit", 0)
    else:
        # Without a plan/category mapping per claim, apply the most permissive
        # (max) per-claim limit as a conservative global ceiling
        ceiling = float(coverage["per_claim_limit"].dropna().max())
        hits = 0
        for i, amt in enumerate(amounts):
            if amt > ceiling:
                flags[i].append(_flag("R04",
                    f"Valor {amt:,.0f} excede o limite máximo por sinistro "
                    f"({ceiling:,.0f})"))
                severity_sum[i] += 3
                hits += 1
        record("R04", True, None, hits)

    # R05 — annual cap per member exceeded ────────────────────────────────────
    enabled, _ = setting("R05")
    if not enabled:
        record("R05", False, "Desactivada pelo utilizador", 0)
    elif not has_coverage or not has_date or coverage["annual_limit"].dropna().empty:
        record("R05", False, "Requer regras de cobertura com annual_limit e claim_date", 0)
    else:
        cap = float(coverage["annual_limit"].dropna().max())
        tmp = df[["member_id"]].copy()
        tmp["year"] = pd.to_datetime(df["claim_date"], errors="coerce").dt.year
        tmp["amt"] = amounts
        cum = tmp.groupby(["member_id", "year"])["amt"].cumsum()
        hits = 0
        for i, total in enumerate(cum.values):
            if total > cap:
                flags[i].append(_flag("R05",
                    f"Acumulado anual do beneficiário ({total:,.0f}) excede o "
                    f"plafond ({cap:,.0f})"))
                severity_sum[i] += 3
                hits += 1
        record("R05", True, None, hits)

    # R06 — gender-incompatible procedure ─────────────────────────────────────
    enabled, _ = setting("R06")
    if not enabled:
        record("R06", False, "Desactivada pelo utilizador", 0)
    elif not (has_gender and has_proc):
        record("R06", False, "Requer member_gender e procedure_code", 0)
    else:
        genders = df["member_gender"].astype(str).str.upper().str[0]
        hits = 0
        for i, (p, g) in enumerate(zip(proc, genders)):
            for prefix, allowed in GENDER_RESTRICTED.items():
                if p.startswith(prefix) and g in ("M", "F") and g != allowed:
                    flags[i].append(_flag("R06",
                        f"Procedimento {p} incompatível com género {g}"))
                    severity_sum[i] += 3
                    hits += 1
                    break
        record("R06", True, None, hits)

    # R07 — age-incompatible procedure ────────────────────────────────────────
    enabled, _ = setting("R07")
    if not enabled:
        record("R07", False, "Desactivada pelo utilizador", 0)
    elif not (has_age and has_proc):
        record("R07", False, "Requer member_age e procedure_code", 0)
    else:
        ages = pd.to_numeric(df["member_age"], errors="coerce")
        hits = 0
        for i, (p, a) in enumerate(zip(proc, ages)):
            if pd.isna(a):
                continue
            for prefix, (lo, hi) in AGE_RESTRICTED.items():
                if p.startswith(prefix) and not (lo <= a <= hi):
                    flags[i].append(_flag("R07",
                        f"Procedimento {p} incompatível com idade {a:.0f}"))
                    severity_sum[i] += 3
                    hits += 1
                    break
        record("R07", True, None, hits)

    # R08 — impossible daily volume per provider ──────────────────────────────
    enabled, max_daily = setting("R08")
    if not enabled:
        record("R08", False, "Desactivada pelo utilizador", 0)
    elif not has_date:
        record("R08", False, "Requer claim_date", 0)
    else:
        dates = pd.to_datetime(df["claim_date"], errors="coerce").dt.date
        daily = df.groupby([df["provider_id"], dates])["claim_id"].transform("count")
        hits = 0
        for i, cnt in enumerate(daily.values):
            if cnt > max_daily:
                flags[i].append(_flag("R08",
                    f"Prestador com {int(cnt)} actos no mesmo dia "
                    f"(máx. plausível: {int(max_daily)})"))
                severity_sum[i] += 3
                hits += 1
        record("R08", True, None, hits)

    # R09 — procedure outside provider specialty ──────────────────────────────
    # Conservative: only flag STRONG mismatches — the procedure prefix belongs
    # to another specialty's signature AND not to the provider's own. Generic
    # procedures (consultations etc.) never flag, since any specialty bills them.
    enabled, _ = setting("R09")
    if not enabled:
        record("R09", False, "Desactivada pelo utilizador", 0)
    elif not (has_spec and has_proc):
        record("R09", False, "Requer provider_specialty e procedure_code", 0)
    else:
        specs = df["provider_specialty"].astype(str)
        hits = 0
        for i, (p, s) in enumerate(zip(proc, specs)):
            own = SPECIALTY_PROCEDURES.get(s)
            if not (own and p):
                continue
            if any(p.startswith(pre) for pre in own):
                continue  # within own specialty signature
            # Does it match another specialty's signature?
            foreign = None
            for other_spec, prefixes in SPECIALTY_PROCEDURES.items():
                if other_spec != s and any(p.startswith(pre) for pre in prefixes):
                    foreign = other_spec
                    break
            if foreign:
                flags[i].append(_flag("R09",
                    f"Procedimento {p} típico de {foreign}, facturado por {s}"))
                severity_sum[i] += 2
                hits += 1
        record("R09", True, None, hits)

    # R10 — weekend concentration ─────────────────────────────────────────────
    enabled, wk_pct = setting("R10")
    if not enabled:
        record("R10", False, "Desactivada pelo utilizador", 0)
    elif not has_date:
        record("R10", False, "Requer claim_date", 0)
    else:
        dts = pd.to_datetime(df["claim_date"], errors="coerce")
        is_weekend = dts.dt.dayofweek >= 5
        prov_weekend = is_weekend.groupby(df["provider_id"]).transform("mean") * 100
        prov_count = df.groupby("provider_id")["claim_id"].transform("count")
        hits = 0
        for i, (w, pct_w, cnt) in enumerate(zip(is_weekend, prov_weekend, prov_count)):
            if w and pct_w > wk_pct and cnt >= 10:
                flags[i].append(_flag("R10",
                    f"Prestador com {pct_w:.0f}% dos claims ao fim-de-semana"))
                severity_sum[i] += 1
                hits += 1
        record("R10", True, None, hits)

    # R11 — member in multiple providers same day ─────────────────────────────
    enabled, max_provs = setting("R11")
    if not enabled:
        record("R11", False, "Desactivada pelo utilizador", 0)
    elif not has_date:
        record("R11", False, "Requer claim_date", 0)
    else:
        dates = pd.to_datetime(df["claim_date"], errors="coerce").dt.date
        nprov = df.groupby([df["member_id"], dates])["provider_id"].transform("nunique")
        hits = 0
        for i, cnt in enumerate(nprov.values):
            if cnt >= max_provs:
                flags[i].append(_flag("R11",
                    f"Beneficiário em {int(cnt)} prestadores distintos no mesmo dia"))
                severity_sum[i] += 2
                hits += 1
        record("R11", True, None, hits)

    # R12 — round amounts pattern per provider ────────────────────────────────
    enabled, rnd_pct = setting("R12")
    if not enabled:
        record("R12", False, "Desactivada pelo utilizador", 0)
    else:
        is_round = (np.mod(amounts, 100) == 0) & (amounts > 0)
        prov_round = pd.Series(is_round).groupby(df["provider_id"].values).transform("mean") * 100
        prov_count = df.groupby("provider_id")["claim_id"].transform("count").values
        hits = 0
        for i, (r, pr, cnt) in enumerate(zip(is_round, prov_round, prov_count)):
            if r and pr > rnd_pct and cnt >= 10:
                flags[i].append(_flag("R12",
                    f"Prestador com {pr:.0f}% de valores exactamente redondos"))
                severity_sum[i] += 1
                hits += 1
        record("R12", True, None, hits)

    # ── Score: severity-weighted, saturating at 100 ──────────────────────────
    # One hard violation (sev 3) -> 60; two -> 90; soft signals add less.
    rule_score = np.clip(severity_sum * 20.0, 0, 100)

    results = pd.DataFrame({
        "rule_score": rule_score,
        "rule_flags": flags,
    }, index=df.index)

    return results, status
