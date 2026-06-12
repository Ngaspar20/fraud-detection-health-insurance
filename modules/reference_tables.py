"""
reference_tables.py
-------------------
Client-supplied reference tables that power the deterministic rules engine:

  price_table     : agreed price per procedure (procedure_code -> agreed_price)
  coverage_rules  : benefit coverage rules per plan (limits, waiting periods,
                    exclusions)

Both are uploaded as Excel/CSV on the "Tabelas de Referencia" page, stored in
SQLite (same claims.db) and consumed by modules/rules_engine.py.

Canonical formats (templates downloadable in the app):

price_table:
  procedure_code | procedure_desc | agreed_price | currency
coverage_rules:
  plan_id | benefit_category | annual_limit | per_claim_limit |
  waiting_period_days | excluded_procedures (';'-separated codes)
"""

import io
import pandas as pd
from datetime import datetime

from modules.data_loader import _get_conn

# ── Canonical schemas ─────────────────────────────────────────────────────────

PRICE_COLS = ["procedure_code", "procedure_desc", "agreed_price", "currency"]
COVERAGE_COLS = ["plan_id", "benefit_category", "annual_limit",
                 "per_claim_limit", "waiting_period_days", "excluded_procedures"]

# Fuzzy aliases so client exports map onto canonical names (same approach as
# column_detector.py)
PRICE_ALIASES = {
    "procedure_code": ["proccode", "cpt_code", "cpt", "codigo", "codigo_acto",
                       "cod_procedimento", "act_code", "code"],
    "procedure_desc": ["descricao", "designacao", "description", "desc",
                       "procedimento", "acto", "nome"],
    "agreed_price":   ["preco", "preco_acordado", "valor", "price", "tarifa",
                       "valor_acordado", "unit_price", "preco_tabela"],
    "currency":       ["moeda", "divisa", "curr"],
}
COVERAGE_ALIASES = {
    "plan_id":             ["plano", "plan", "id_plano", "plan_code", "codigo_plano"],
    "benefit_category":    ["categoria", "beneficio", "benefit", "category",
                            "categoria_beneficio", "cobertura"],
    "annual_limit":        ["plafond", "plafond_anual", "limite_anual", "annual_cap",
                            "limite", "yearly_limit"],
    "per_claim_limit":     ["limite_por_sinistro", "limite_acto", "per_claim",
                            "limite_sinistro", "claim_cap"],
    "waiting_period_days": ["carencia", "periodo_carencia", "carencia_dias",
                            "waiting_period", "waiting_days"],
    "excluded_procedures": ["exclusoes", "actos_excluidos", "exclusions",
                            "excluded", "procedimentos_excluidos"],
}


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _map_columns(df: pd.DataFrame, canonical_cols: list, aliases: dict) -> pd.DataFrame:
    raw = {_normalize(c): c for c in df.columns}
    mapping = {}
    for canonical in canonical_cols:
        if canonical in raw:
            mapping[raw[canonical]] = canonical
            continue
        for alias in aliases.get(canonical, []):
            if alias in raw:
                mapping[raw[alias]] = canonical
                break
    return df.rename(columns=mapping)


def _parse_pt_number(series: pd.Series) -> pd.Series:
    """Handle Portuguese number format (1.500,00) as well as plain numerics."""
    if series.dtype == object:
        cleaned = (series.astype(str)
                   .str.replace(r"\s", "", regex=True)
                   .str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False))
        converted = pd.to_numeric(cleaned, errors="coerce")
        # If the PT interpretation failed for most rows, fall back to direct parse
        if converted.notna().mean() < 0.5:
            converted = pd.to_numeric(series, errors="coerce")
        return converted
    return pd.to_numeric(series, errors="coerce")


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_table (
            procedure_code TEXT PRIMARY KEY,
            procedure_desc TEXT,
            agreed_price   REAL,
            currency       TEXT,
            uploaded_at    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coverage_rules (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id             TEXT,
            benefit_category    TEXT,
            annual_limit        REAL,
            per_claim_limit     REAL,
            waiting_period_days INTEGER,
            excluded_procedures TEXT,
            uploaded_at         TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rule_settings (
            rule_id  TEXT PRIMARY KEY,
            enabled  INTEGER NOT NULL DEFAULT 1,
            param    REAL
        )
    """)
    conn.commit()


# ── Price table ───────────────────────────────────────────────────────────────

def save_price_table(df_raw: pd.DataFrame) -> tuple[int, list[str]]:
    """Parse, validate and persist the price table. Replaces previous upload.
    Returns (n_rows_saved, list_of_warnings)."""
    warnings = []
    df = _map_columns(df_raw, PRICE_COLS, PRICE_ALIASES)

    if "procedure_code" not in df.columns or "agreed_price" not in df.columns:
        raise ValueError(
            "Colunas obrigatórias em falta: procedure_code e agreed_price. "
            f"Colunas encontradas: {', '.join(df_raw.columns)}"
        )

    df["procedure_code"] = df["procedure_code"].astype(str).str.strip()
    df["agreed_price"] = _parse_pt_number(df["agreed_price"])

    n_bad = df["agreed_price"].isna().sum()
    if n_bad:
        warnings.append(f"{n_bad} linhas com preço inválido foram ignoradas.")
        df = df.dropna(subset=["agreed_price"])

    dups = df["procedure_code"].duplicated().sum()
    if dups:
        warnings.append(f"{dups} códigos duplicados — mantida a última ocorrência.")
        df = df.drop_duplicates(subset=["procedure_code"], keep="last")

    if "procedure_desc" not in df.columns:
        df["procedure_desc"] = ""
    if "currency" not in df.columns:
        df["currency"] = ""

    df = df[["procedure_code", "procedure_desc", "agreed_price", "currency"]].copy()
    df["uploaded_at"] = datetime.now().isoformat()

    conn = _get_conn()
    _ensure_tables(conn)
    conn.execute("DELETE FROM price_table")
    df.to_sql("price_table", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return len(df), warnings


def load_price_table() -> pd.DataFrame:
    try:
        conn = _get_conn()
        _ensure_tables(conn)
        df = pd.read_sql("SELECT * FROM price_table", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=PRICE_COLS)


def price_table_template() -> bytes:
    """Canonical Excel template the client can fill in."""
    sample = pd.DataFrame({
        "procedure_code": ["99213", "99215", "27447"],
        "procedure_desc": ["Consulta de rotina", "Consulta complexa",
                           "Artroplastia total do joelho"],
        "agreed_price":   [3500.00, 7500.00, 450000.00],
        "currency":       ["KWZ", "KWZ", "KWZ"],
    })
    buf = io.BytesIO()
    sample.to_excel(buf, index=False, sheet_name="tabela_precos")
    return buf.getvalue()


# ── Coverage rules ────────────────────────────────────────────────────────────

def save_coverage_rules(df_raw: pd.DataFrame) -> tuple[int, list[str]]:
    """Parse, validate and persist coverage rules. Replaces previous upload."""
    warnings = []
    df = _map_columns(df_raw, COVERAGE_COLS, COVERAGE_ALIASES)

    if "benefit_category" not in df.columns and "plan_id" not in df.columns:
        raise ValueError(
            "É necessária pelo menos uma das colunas: plan_id ou benefit_category. "
            f"Colunas encontradas: {', '.join(df_raw.columns)}"
        )

    for col in ["annual_limit", "per_claim_limit"]:
        if col in df.columns:
            df[col] = _parse_pt_number(df[col])
        else:
            df[col] = None
    if "waiting_period_days" in df.columns:
        df["waiting_period_days"] = pd.to_numeric(
            df["waiting_period_days"], errors="coerce")
    else:
        df["waiting_period_days"] = None
    for col in ["plan_id", "benefit_category", "excluded_procedures"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[COVERAGE_COLS].copy()
    df["uploaded_at"] = datetime.now().isoformat()

    conn = _get_conn()
    _ensure_tables(conn)
    conn.execute("DELETE FROM coverage_rules")
    df.to_sql("coverage_rules", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return len(df), warnings


def load_coverage_rules() -> pd.DataFrame:
    try:
        conn = _get_conn()
        _ensure_tables(conn)
        df = pd.read_sql("SELECT * FROM coverage_rules", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=COVERAGE_COLS)


def coverage_template() -> bytes:
    sample = pd.DataFrame({
        "plan_id":             ["PLANO_BASE", "PLANO_BASE", "PLANO_PREMIUM"],
        "benefit_category":    ["Consultas", "Cirurgia", "Cirurgia"],
        "annual_limit":        [150000.00, 2000000.00, 5000000.00],
        "per_claim_limit":     [10000.00, 800000.00, 1500000.00],
        "waiting_period_days": [0, 90, 60],
        "excluded_procedures": ["", "27447;27130", ""],
    })
    buf = io.BytesIO()
    sample.to_excel(buf, index=False, sheet_name="regras_cobertura")
    return buf.getvalue()


# ── Rule settings (enable/disable + parameters) ───────────────────────────────

def load_rule_settings() -> dict:
    """Return {rule_id: {'enabled': bool, 'param': float|None}}."""
    try:
        conn = _get_conn()
        _ensure_tables(conn)
        rows = conn.execute("SELECT rule_id, enabled, param FROM rule_settings").fetchall()
        conn.close()
        return {r[0]: {"enabled": bool(r[1]), "param": r[2]} for r in rows}
    except Exception:
        return {}


def save_rule_setting(rule_id: str, enabled: bool, param: float = None) -> None:
    conn = _get_conn()
    _ensure_tables(conn)
    conn.execute(
        "INSERT OR REPLACE INTO rule_settings (rule_id, enabled, param) VALUES (?, ?, ?)",
        (rule_id, int(enabled), param),
    )
    conn.commit()
    conn.close()
