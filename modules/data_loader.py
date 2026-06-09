import sqlite3
import pandas as pd
import hashlib
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "claims.db"


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            filename     TEXT,
            uploaded_at  TEXT,
            row_count    INTEGER,
            columns      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id     TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            verdict      TEXT NOT NULL,
            investigator TEXT,
            notes        TEXT,
            created_at   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ── Feedback CRUD ─────────────────────────────────────────────────────────────

def save_feedback(claim_id: str, session_id: str, verdict: str,
                  investigator: str = "", notes: str = "") -> None:
    """Save or update investigator verdict for a claim.
    verdict must be: 'Fraude Confirmada' | 'Falso Positivo' | 'Em Investigação'
    """
    conn = _get_conn()
    # Upsert: one verdict per (claim_id, session_id)
    conn.execute(
        """INSERT INTO feedback (claim_id, session_id, verdict, investigator, notes, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT DO NOTHING""",
        (claim_id, session_id, verdict, investigator, notes, datetime.now().isoformat())
    )
    # Update if already exists
    conn.execute(
        """UPDATE feedback SET verdict=?, investigator=?, notes=?, created_at=?
           WHERE claim_id=? AND session_id=?""",
        (verdict, investigator, notes, datetime.now().isoformat(), claim_id, session_id)
    )
    conn.commit()
    conn.close()


def load_feedback(session_id: str) -> pd.DataFrame:
    """Return all feedback rows for a session as a DataFrame."""
    try:
        conn = _get_conn()
        df = pd.read_sql(
            "SELECT * FROM feedback WHERE session_id=? ORDER BY created_at DESC",
            conn, params=(session_id,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_feedback_stats() -> pd.DataFrame:
    """Return aggregate feedback counts across all sessions."""
    try:
        conn = _get_conn()
        df = pd.read_sql(
            "SELECT verdict, COUNT(*) as count FROM feedback GROUP BY verdict",
            conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def save_session(df: pd.DataFrame, filename: str) -> str:
    session_id = hashlib.md5(f"{filename}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    conn = _get_conn()
    df.to_sql(f"claims_{session_id}", conn, if_exists="replace", index=False)
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
        (session_id, filename, datetime.now().isoformat(), len(df), ",".join(df.columns.tolist()))
    )
    conn.commit()
    conn.close()
    return session_id


def list_sessions() -> pd.DataFrame:
    try:
        conn = _get_conn()
        df = pd.read_sql("SELECT * FROM sessions ORDER BY uploaded_at DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_session(session_id: str) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql(f"SELECT * FROM claims_{session_id}", conn)
    conn.close()
    return df


def delete_session(session_id: str):
    conn = _get_conn()
    conn.execute(f"DROP TABLE IF EXISTS claims_{session_id}")
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def parse_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, low_memory=False)
    else:
        df = pd.read_excel(uploaded_file)
    return df
